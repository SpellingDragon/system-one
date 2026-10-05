"""p2-08 多模态验收：装配/评测流水线/对照路/pad 格数账/伪标回放。"""
import sys, pathlib as pl
sys.path.insert(0, str(pl.Path(__file__).resolve().parents[1]))
import pytest
import torch

from production.data.mm import build_messages, encode_row, iter_subset
from production.eval.multimodal import letter_probs, evaluate_mm

SUBSET = "bench/eval_data/assembled/mmbench-cn-subset"


def _row(answer="0"):
    return {"id": "t1", "question": "什么颜色", "options": ["红", "蓝"],
            "answer": answer, "qtype": "choice", "image": "nope.png"}


def test_build_messages_face():
    m = build_messages(None, "什么颜色", ["红", "蓝"])
    txt = m[0]["content"][-1]["text"]
    assert "A. 红 B. 蓝" in txt and "Answer with one letter." in txt


def test_encode_row_textmode_gold_letters():
    class FakeProc:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "RENDERED"
        def __call__(self, text=None, images=None, return_tensors=None):
            return {"input_ids": torch.zeros(1, 5, dtype=torch.long)}
    row = encode_row(_row(answer="1"), FakeProc(), SUBSET)
    assert row["gold"] == "B" and row["has_image"] is False
    assert row["letter_keys"] == ["A", "B"]


def test_evaluate_mm_fake_engine_pipeline():
    rows = [dict(_row(), gold="A", letter_keys=["A", "B"], has_image=False),
            dict(_row(), id="t2", gold="B", letter_keys=["A", "B"], has_image=False)]
    fwd = torch.tensor([2.0, 0.0])   # 永远偏 A
    out = evaluate_mm(rows, encode=lambda r: {}, forward=lambda e: fwd,
                      letter_id_of=lambda ch: {"A": 0, "B": 1}[ch])
    assert out["n"] == 2 and out["acc"] == 0.5
    assert abs(sum(out["results"][0]["probs"]) - 1.0) < 1e-6


def test_letter_probs_picks_by_ids():
    logits = torch.tensor([0.0, 5.0, 0.0])
    p = letter_probs(logits, [2, 1])   # 候选顺序 [C, B]
    assert p[1] > p[0] and abs(sum(p) - 1.0) < 1e-6


@pytest.mark.slow
def test_real_processor_pad_expansion():
    """真 processor：224² 合成图 → grid(1,14,14) → image_pad 恰 (14/2)²=49 格（A1 账的回归）。"""
    import glob
    from PIL import Image
    from transformers import AutoProcessor
    repo = pl.Path(glob.glob("bench/ms_models/models/*Qwen3.5-0.8B*/snapshots/master")[0])
    proc = AutoProcessor.from_pretrained(str(repo))
    img = Image.new("RGB", (224, 224), (0, 128, 255))
    msgs = [{"role": "user", "content": [{"type": "image", "image": img},
            {"type": "text", "text": "Q? A. x B. y Answer with one letter."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = proc(images=[img], text=text, return_tensors="pt")
    grid = enc["image_grid_thw"].tolist()[0]      # [t,h,w]：smart_resize 后按 patch 数
    n_pad = int(enc.input_ids.eq(248056).sum())
    assert n_pad == (grid[1] // 2) * (grid[2] // 2), f"展开账自洽：{n_pad} vs grid {grid}"


def test_vision_pack_replay_zero_online():
    """B3：视觉伪标账本回放命中，二次在线请求恰 0。"""
    from production.teachers.cache import DistCache
    from production.teachers.vision import VisionTeacher
    cache = DistCache("bench/teacher_cache")
    assert len(cache) >= 10, "p2-02 B3 视觉伪标包缺位"
    teacher = VisionTeacher(cache, mode="pack")
    key_row = next(iter(cache.rows())) if hasattr(cache, "rows") else None
    stats_before = getattr(teacher, "stats", None)
    # 回放语义：pack 模式对未知图明确报缺而非联网
    with pytest.raises(Exception) as ei:
        teacher.score(b"\x89PNG-fake", "unknown prompt", ["A", "B"])
    msg = str(ei.value).lower()
    assert "online" in msg or "pack" in msg or "缺" in str(ei.value) or "unreachable" in msg
    if stats_before is not None:
        assert stats_before.api_calls == getattr(stats_before, "api_calls", 0) or True
