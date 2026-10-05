"""p2-10 serving 本地半场单测：A1 HTTP 服务、A2 endpoint↔本地一致。

口径（与 test_longctx.py 同源）：
  * 全部 CPU 跑完——HTTP 与一致两路都用"纯 python 假引擎"，真 backbone（0.6B 替身）的
    前向凭据走 run 档案（`python -m serving.latency --new-run p210-*`），不在单测里挂
    六秒模型加载；
  * 资源纪律：禁用 MPS（归一阶段训练长跑）；本波一律 CPU 路验证；
  * 用例名按孙任务分关键字：http / parity，与二级 tasks.md 的验证命令 `-k <关键字>`
    一一对应。

假引擎的设计意图：它只承诺"同输入同输出"的确定性，不承诺像真模型。于是
`DecisionService` 的进程内路（= 评测侧 `--model` 本地路的替身）与 HTTP `/v1/systemone`
路（= 评测侧 `--endpoint` 路的替身）走的是同一份权重、同一套 decision 原函数——
parity 用例断言的正是"网络这层皮不改变读数"（argmax 不换人、份额在容差内相同）。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest
import torch

from serving.server import (
    BATCH_PATH,
    HEALTH_PATH,
    STATS_PATH,
    SYSTEMONE_PATH,
    DecisionService,
    spawn,
)
from sys1.decision import RENDER_VERSION, THINK_OFF_SUFFIX


# ------------------------------------------------------------------ 假引擎/假分词器（CPU 替身）
class FakeTokenizer:
    """把消息拼成整段文字并按字符编号；模板尾巴钉死"关闭思考"记号，与真模板同构。"""

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False) -> str:  # noqa: ANN001, ANN003 - 对齐 HF 签名
        text = "\n".join(str(m.get("content", "")) for m in messages)
        return text + THINK_OFF_SUFFIX if add_generation_prompt else text

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [(ord(ch) % 250) + 2 for ch in text]


class FakeEngine:
    """确定性假引擎：`hidden[b,t,j] = f(该行到 t 为止的编号累加, j, t)`，与模型无关。

    白话：只要送进去的编号一样，交出的逐位置数字就一样；26 个字母各占输出层一维
    （head 取单位阵），于是候选分 = 末位数字的前 k 个分量，读得出稳定名次。
    """

    def __init__(self, *, dim: int = 26) -> None:
        self.dim = dim
        self.tokenizer = FakeTokenizer()
        self.head_weight = torch.eye(dim)
        self.letter_ids = tuple(range(dim))
        self.pad_id = 0
        self.dtype = torch.float32
        self.device = "cpu"
        self.calls: list[tuple[int, int]] = []

    def forward_hidden(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        batch, width = int(input_ids.shape[0]), int(input_ids.shape[1])
        out = torch.zeros(batch, width, self.dim, dtype=torch.float32)
        for b in range(batch):
            running = 0
            for t in range(width):
                running += int(input_ids[b, t])
                for j in range(self.dim):
                    out[b, t, j] = float((running * (j + 3) + t) % 13) - 6.0 + 0.01 * j
        self.calls.append((batch, width))
        return out

    def describe(self) -> dict:
        return {
            "engine": "fake-deterministic", "loader": "minimal", "device": "cpu",
            "dtype": "float32", "backend": "torch_eager", "head_shape": [self.dim, self.dim],
            "pad_id": self.pad_id, "render_version": RENDER_VERSION, "provenance": {},
        }


@pytest.fixture(scope="module")
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture(scope="module")
def service(engine: FakeEngine) -> DecisionService:
    return DecisionService(engine)


@pytest.fixture(scope="module")
def server(service: DecisionService):
    srv = spawn(service)
    try:
        yield srv
    finally:
        srv.stop()


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(url: str, payload: dict) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


REQUEST = {
    "id": "p210-req",
    "state": "weather: clear sky at 07:00, humidity 40%, rising pressure",
    "questions": {
        "q1": {"type": "choice", "instructions": "Which option fits the evidence?",
               "criteria": {"a": "sunny", "b": "rain", "c": "cloudy"}},
        "q2": {"type": "noul", "instructions": "Does the evidence support the claim?"},
        "q3": {"type": "score", "instructions": "Rate the confidence.",
               "criteria": ["low", "medium", "high"]},
    },
}


# ================================================================== A1 HTTP 服务（-k http）
def test_http_health_reports_engine_and_render_version(server):
    """/health 交回 status=ok 与引擎画像、渲染版号——起没起来一眼看清。"""
    body = _get(server.base_url + HEALTH_PATH)
    assert body["status"] == "ok"
    assert body["render_version"] == RENDER_VERSION
    assert body["engine"]["engine"] == "fake-deterministic"
    assert body["engine"]["device"] == "cpu"


def test_http_decide_single_answers_all_three_qtypes(server):
    """/v1/systemone 一次答 choice/noul/score 三型：份额和为 1、ms 字段齐全。"""
    body = _post(server.endpoint, REQUEST)
    assert body["id"] == "p210-req"
    assert set(body["answers"]) == {"q1", "q2", "q3"}
    # 每型候选集与 decision 契约对齐（choice→abc / noul→true,false / score→0,1,2）
    assert set(body["answers"]["q1"]) == {"a", "b", "c"}
    assert set(body["answers"]["q2"]) == {"true", "false"}
    assert set(body["answers"]["q3"]) == {"0", "1", "2"}
    for qid, probs in body["answers"].items():
        assert abs(sum(probs.values()) - 1.0) < 1e-5, qid
    assert isinstance(body["ms"], (int, float)) and body["ms"] >= 0.0
    assert body["ms_semantics"] == "forward_plus_readout_of_this_request"
    assert body["render_version"] == RENDER_VERSION
    assert "engine" in body


def test_http_batch_returns_five_with_ms_fields(server):
    """批量端点承 5 份请求：回 5 组 answers，ms/ms_batch 齐全（spec 批量场景）。"""
    batch = [{"id": f"b{i}", "state": REQUEST["state"],
              "questions": {"q1": {"type": "choice", "instructions": f"Which fits? #{i}",
                                   "criteria": {"a": "sunny", "b": "rain"}}}} for i in range(5)]
    body = _post(server.base_url + BATCH_PATH, {"requests": batch})
    assert len(body["results"]) == 5
    assert [r["id"] for r in body["results"]] == [f"b{i}" for i in range(5)]
    for r in body["results"]:
        assert set(r["answers"]["q1"]) == {"a", "b"}
        assert isinstance(r["ms"], (int, float))
        assert r["ms_batch"] == body["results"][0]["ms_batch"]
        assert r["ms_semantics"] == "amortized_within_batch"


def test_http_rejects_empty_questions_with_400(server):
    """不合协议的请求（空 questions）当场 400 并说清原因，绝不当 500 掩盖。"""
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(server.endpoint, {"id": "bad", "state": "s", "questions": {}})
    assert exc.value.code == 400
    detail = json.loads(exc.value.read().decode("utf-8"))
    assert detail["error"] == "bad_request"


def test_http_stats_endpoint_reports_counters(server):
    """/stats 交回记账字段（requests/rows/forwards/tokens_fed/prefix_*）——B2 验收面。"""
    body = _get(server.base_url + STATS_PATH)
    for key in ("requests", "rows", "forwards", "tokens_fed",
                "prefix_hits", "prefix_misses", "prefix_enabled"):
        assert key in body
    assert body["render_version"] == RENDER_VERSION


# ================================================================== A2 双路一致（-k parity）
def test_parity_endpoint_matches_inprocess(server, service):
    """同一副权重：HTTP `--endpoint` 路 vs 进程内 `--model` 路 —— argmax 不换人、份额同容差。

    白话：服务只是 decision 的网络皮，序列化/合批不该挪动任何一个名次。这里拿同一个
    请求分别问进程内和走 HTTP，逐题比 argmax 与整条份额分布。
    """
    local = service.decide(REQUEST)                 # 进程内路（--model 替身）
    served = _post(server.endpoint, REQUEST)        # HTTP 路（--endpoint 替身）
    assert set(local["answers"]) == set(served["answers"])
    for qid in local["answers"]:
        lo, so = local["answers"][qid], served["answers"][qid]
        assert set(lo) == set(so), qid
        assert max(lo, key=lo.get) == max(so, key=so.get), f"argmax 换人：{qid}"
        for code in lo:
            assert abs(lo[code] - so[code]) < 1e-6, f"{qid}/{code} 份额漂移超容差"


def test_parity_pad_invariance_batched_vs_single(server, service):
    """合批右补空洞不改变名次：同题单问 vs 混进长题批里 argmax 一致（spec"三问一次前向"的稳定性）。"""
    short = {"id": "s", "state": REQUEST["state"], "questions": {
        "q1": {"type": "choice", "instructions": "Short?", "criteria": {"a": "x", "b": "y"}}}}
    solo = service.decide(short)["answers"]["q1"]
    padded = {"id": "L", "state": REQUEST["state"], "questions": {
        "q_long": {"type": "choice", "instructions": "z" * 200,
                   "criteria": {"a": "x", "b": "y", "c": "zz", "d": "ww"}},
        **short["questions"]}}
    mixed = service.decide(padded)["answers"]["q1"]
    assert max(solo, key=solo.get) == max(mixed, key=mixed.get)
    for code in solo:
        assert abs(solo[code] - mixed[code]) < 1e-5
