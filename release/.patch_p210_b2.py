"""p2-10 B2 补丁：给假引擎补最小 KV 复用面 + `-k hit_count` 进程内单测门。

跑法：cd release && .venv/bin/python .patch_p210_b2.py
验证：.venv/bin/python -m pytest tests/test_serving.py -k hit_count -q   exit 0

前置：目标文件须为 git HEAD 状态（本脚本一次性落地 5 处改动，重复跑会因锚点已换而停下）。

设计要点（为什么这样补才测到服务路径本身）：
  1. `KVEngine` 三件能力（forward 带 past 增量 / clone_past / past_bytes）用"已看过的编号列
     当小抄"来充当，末位数字仍交回服务侧那同一个 `forward_hidden` 现算——于是复用路与重算路
     共用一条公式，护栏（verify=once 的对拍）必然判等，不会因为"替身不像真模型"而误判负、
     把键拉黑，命中计数测的就是 `DecisionService._read_prefixed` 那条服务路径；
  2. 不新写第二套数字公式（两套公式各抄一份 = p2-03/p2-09 反复踩的"两处真源"坑），
     FakeKVEngine 直接借 FakeEngine.forward_hidden；
  3. 用例名带 `hit_count` 关键字，与二级 tasks B2 行的验证命令一一对应。
"""
from __future__ import annotations

from pathlib import Path

TARGET = Path(__file__).resolve().parent / "tests" / "test_serving.py"
src = TARGET.read_text(encoding="utf-8")
edits: list[tuple[str, str]] = []

# ── 1. 导入：KV 面要用的 EngineOut ───────────────────────────────────────
edits.append((
    '''from serving.server import (
    BATCH_PATH,''',
    '''from serving.prefix_cache import EngineOut
from serving.server import (
    BATCH_PATH,''',
))

# ── 2. 模块 docstring 的用例关键字清单补 hit_count ──────────────────────
edits.append((
    '''  * 用例名按孙任务分关键字：http / parity，与二级 tasks.md 的验证命令 `-k <关键字>`
    一一对应。''',
    '''  * 用例名按孙任务分关键字：http / parity / hit_count，与二级 tasks.md 的验证命令
    `-k <关键字>` 一一对应（hit_count = B2 前缀复用的进程内门）。
  * B2 门用的是**同一个假引擎**加一副最小 KV 面（`FakeKVEngine`）：末位数字一律回交
    `forward_hidden` 现算，复用路与重算路因此共用一条公式，护栏只会判等——命中计数涨没涨、
    命中题有没有重付前缀符号，测的都是服务路径，不是替身像不像真模型。''',
))

# ── 3. FakeEngine.__init__ 补 KV 面占位（没被取用时应是 None，而非 AttributeError）──
edits.append((
    '''        self.calls: list[tuple[int, int]] = []

    def forward_hidden''',
    '''        self.calls: list[tuple[int, int]] = []
        self.kv: "FakeKVEngine" = None              # kv_engine() 交出的替身面（B2：runner 用它）

    def forward_hidden''',
))

# ── 4. FakeEngine 补 kv_engine()（与真引擎 TorchEngine 同名单）───────────
edits.append((
    '''    def describe(self) -> dict:
        return {
            "engine": "fake-deterministic", "loader": "minimal", "device": "cpu",''',
    '''    def kv_engine(self) -> "FakeKVEngine":
        """交出可续 KV 的替身面（B2）：与真引擎 `TorchEngine.kv_engine()` 同一张接口。"""
        self.kv = FakeKVEngine(self)
        return self.kv

    def describe(self) -> dict:
        return {
            "engine": "fake-deterministic", "loader": "minimal", "device": "cpu",''',
))

# ── 5. 新类 FakeKVEngine：放在 FakeEngine 之后、fixture 之前 ─────────────
edits.append((
    '''@pytest.fixture(scope="module")
def engine() -> FakeEngine:''',
    '''class FakeKVEngine:
    """`KVEngine` 协议的假实现：past = 已看过的编号列，数字回交 `FakeEngine.forward_hidden`。

    白话：缓存层对"模型"的全部要求就三件事——能带着上次的小抄增量前向、能把小抄拷一份、
    能报小抄有多重。这里把"看过哪些字"直接当小抄，算数字时仍走服务侧那同一个
    `forward_hidden`，所以"复用路（只喂问题段）"与"重算路（整段重来）"交出的末位数字必然
    逐位相同。于是护栏判负只剩一种可能：服务路径真算错了——用例断言"命中数 ≥1"时测的是
    记账与付费面，而不是替身像不像真模型。增量前向还顺手核对 offset 必须等于已看过的长度：
    位置抬错（忘抬 offset）在真模型上是旋转角度全错，在这儿会当场炸给测试看，不静默到底。
    """

    def __init__(self, owner: FakeEngine) -> None:
        self.owner = owner
        self.forwards: list[dict] = []          # 逐次前向的开销账（命中/付费断言的独立凭据）

    def forward(self, input_ids, past=None, *, offset: int = 0) -> EngineOut:  # noqa: ANN001 - 协议签名
        seen = list(past["tokens"]) if past is not None else []
        fed = [int(t) for t in input_ids]
        if past is not None and offset != len(seen):
            raise AssertionError(
                f"增量前向的 offset={offset} 与已看过 {len(seen)} 个符号不符——位置抬错即全错")
        ids = seen + fed
        hidden = self.owner.forward_hidden(
            torch.tensor([ids], dtype=torch.long), torch.ones(1, len(ids), dtype=torch.long))
        last = hidden[0, len(ids) - 1].unsqueeze(0)          # (1, d)：读点 = 最后一个真位置
        new_past = {"tokens": ids,
                    "keys": torch.zeros(1, len(ids), 4), "values": torch.zeros(1, len(ids), 4)}
        self.forwards.append({"fed": len(fed), "used_past": past is not None})
        return EngineOut(last, new_past, len(fed))

    def clone_past(self, past):                              # noqa: ANN001 - 协议签名
        """拷小抄：编号列给新表、两张小张量 .clone()——原件被涂改会让后续命中全是脏的。"""
        return {"tokens": list(past["tokens"]),
                "keys": past["keys"].clone(), "values": past["values"].clone()}

    def past_bytes(self, past) -> int:                       # noqa: ANN001 - 协议签名
        return (past["keys"].numel() + past["values"].numel()) * past["keys"].element_size()


@pytest.fixture(scope="module")
def engine() -> FakeEngine:''',
))

for old, new in edits:
    hits = src.count(old)
    if hits != 1:
        raise SystemExit(f"锚点命中 {hits} 次（要求恰好 1 次）：{old[:70]!r}")
    src = src.replace(old, new, 1)

# ── 6. B2 用例：文件末尾追加 ────────────────────────────────────────────
B2_TEST = '''

# ================================================================== B2 前缀复用（-k hit_count）
def test_hit_count_service_path_counts_hits_and_never_repays_prefix():
    """B2 进程内门：同 state 连问 → 服务面命中计数 ≥1，且命中题不再为前缀付一个符号。

    白话：`prefix=True` 起的服务把"证据段"（state）算一次就存进缓存，之后同一段证据的每一问
    只喂问题段。这条用例盯四件事，缺一样 B2 都不算过：① 命中计数真的涨了——记账在服务侧
    `stats()["prefix_hits"]`，不是只有 runner 内部自娱自乐；② 每个命中题的开销账写着
    `prefix_tokens_fed==0` 且为问题段付的恰是问题段长度（"没重付前缀"是硬账，不是"应该差不多"）；
    ③ 跨请求同 state 还接着命中（缓存不是只活一轮）；④ 护栏没被替身的数值噪声误触发（判负 0 次、
    黑名单空、服务侧重算 0 次）——一旦判负，后面全是整段重算，命中数看着对也没有复用效果。
    最后拿 answers 与 prefix=False 的服务逐题比 argmax：复用不改变读数，才是 B2 能进生产面的前提。
    """
    plain = DecisionService(FakeEngine())                  # 对照组：完全不走复用
    ref = plain.decide(REQUEST)["answers"]

    eng = FakeEngine()
    svc = DecisionService(eng, prefix=True, namespace="p210-hitcount")
    assert svc.runner is not None, "prefix=True 没拼出 runner（B2 面根本没挂上）"
    assert eng.kv is not None, "服务的 KV 面没被取用（runner 走的不是这副假引擎）"

    base = svc.stats()
    assert base["prefix_enabled"] is True and base["prefix_hits"] == 0

    body = svc.decide(REQUEST)                             # 一份请求三问：共用同一段证据
    rows = body["prefix"]["rows"]
    assert len(rows) == 3
    st = svc.stats()
    assert st["prefix_hits"] - base["prefix_hits"] >= 1, \\
        f"服务路径没记到命中（三问同 state 至少该命中 1 次）：{st['prefix_hits']}"
    hits = [r for r in rows if r["hit"]]
    assert hits, f"三问同 state 却一次没命中：{[(r['hit'], r['prefix_tokens_fed']) for r in rows]}"
    for r in hits:
        assert r["tokens_are_suffix_only"], f"命中题重付了前缀：{r}"
        assert r["prefix_tokens_fed"] == 0 and r["suffix_tokens_fed"] == r["suffix_len"]
    cold = [r for r in rows if not r["hit"]]
    assert sum(r["prefix_tokens_fed"] for r in rows) == sum(r["prefix_tokens_fed"] for r in cold), \\
        "证据段被付了不止一次——前缀只该在冷启那回付费"

    again = svc.decide({**REQUEST, "id": "p210-req-2"})     # 跨请求同 state：该继续命中
    again_rows = again["prefix"]["rows"]
    st2 = svc.stats()
    assert st2["prefix_hits"] > st["prefix_hits"], \\
        f"跨请求没接着命中：{st['prefix_hits']}->{st2['prefix_hits']}"
    assert all(r["tokens_are_suffix_only"] for r in again_rows), "第二次请求重付了前缀符号"

    run = svc.stats()["runner"]
    assert run["parity_failed"] == 0, f"护栏判负 {run['parity_failed']} 次（替身公式不一致？）"
    assert run["quarantined"] == [], f"键被拉黑，之后全是重算：{run['quarantined']}"
    assert svc.stats()["prefix_recomputes"] == 0, "服务侧记到重算——命中计数再高也没复用效果"
    for qid in ref:
        best_ref = max(ref[qid], key=ref[qid].get)
        best_hit = max(again["answers"][qid], key=again["answers"][qid].get)
        assert best_ref == best_hit, f"复用路改了名次：{qid} {best_ref} -> {best_hit}"
'''

src = src.rstrip("\n") + "\n" + B2_TEST
TARGET.write_text(src, encoding="utf-8")
print(f"[patch-b2] test_serving.py 落笔 {len(edits)} 处改动 + B2 用例 1 个 -> {TARGET}")
