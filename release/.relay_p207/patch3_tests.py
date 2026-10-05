# -*- coding: utf-8 -*-
"""p2-07 接力补丁③：prefix_cache 两处小修（uncacheable 记账自述 + 无 KV 接口层显式拒绝）+ 测试补齐。

用法：cd release && .venv/bin/python .relay_p207/patch3_tests.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path("/Users/pengweiye/Documents/codes/system-one/release")
PC = ROOT / "serving" / "prefix_cache.py"
TS = ROOT / "tests" / "test_longctx.py"

PC_EDITS: list[tuple[str, str]] = [
    # ① 无 keys/values 接口的层（线性记忆层的形状）不能静默跳过：先按元组老口径兜，再显式拒绝
    (
        '            keys, values = getattr(lay, "keys", None), getattr(lay, "values", None)\n'
        '            if isinstance(keys, torch.Tensor) and isinstance(values, torch.Tensor):\n'
        '                fresh.update(keys.clone(), values.clone(), idx)\n'
        '            elif _has_extra_state(lay):\n',
        '            keys, values = getattr(lay, "keys", None), getattr(lay, "values", None)\n'
        '            if isinstance(keys, torch.Tensor) and isinstance(values, torch.Tensor):\n'
        '                fresh.update(keys.clone(), values.clone(), idx)\n'
        '            elif isinstance(lay, (list, tuple)) and len(lay) >= 2:      # (k, v) 元组老口径\n'
        '                fresh.update(_maybe_clone(lay[0]), _maybe_clone(lay[1]), idx)\n'
        '            elif not hasattr(lay, "keys") and not hasattr(lay, "values"):\n'
        '                raise CacheError(\n'
        '                    f"第 {idx} 层（{type(lay).__name__}）没有 keys/values 接口（线性记忆层的形状），"\n'
        '                    f"本波 clone_past 只认 KV 形态的 past：静默跳过等于把这一层从缓存里整个抹掉，"\n'
        '                    f"改走整段重算"\n'
        '                )\n'
        '            elif _has_extra_state(lay):\n',
    ),
    # ② 命中分支里"拷不动"第二次起也要自述 uncacheable（否则 AskResult 说不是真话）
    (
        '            if key in self._uncacheable:\n'
        '                self.recompute_uncacheable += 1          # 拷不动的 past：每题都从整段重算走\n',
        '            if key in self._uncacheable:\n'
        '                uncacheable = True                       # 拷不动的 past：每题都从整段重算走\n'
        '                self.recompute_uncacheable += 1\n',
    ),
    # ③ 冷启就判"拷不动"的那一次也要计入同一本账（与后续重算同口径，账才对得上）
    (
        '                uncacheable = True\n'
        '                self._uncacheable.add(key)\n'
        '                self.cache.drop(key)\n',
        '                uncacheable = True\n'
        '                self._uncacheable.add(key)\n'
        '                self.recompute_uncacheable += 1\n'
        '                self.cache.drop(key)\n',
    ),
]

TS_EDITS: list[tuple[str, str]] = [
    # imports：把渲染切分与引擎守卫要用的名字请进来
    (
        'from serving.prefix_cache import (\n'
        '    DEFAULT_DRIFT_RATIO,\n'
        '    EngineOut,\n'
        '    PrefixCache,\n'
        '    PrefixRunner,\n'
        '    estimate_past_bytes,\n'
        '    prefix_key,\n'
        ')\n',
        'from serving.prefix_cache import (\n'
        '    DEFAULT_DRIFT_RATIO,\n'
        '    SPLIT_MARK,\n'
        '    CacheError,\n'
        '    EngineOut,\n'
        '    HFEngine,\n'
        '    PrefixCache,\n'
        '    PrefixRunner,\n'
        '    estimate_past_bytes,\n'
        '    prefix_key,\n'
        '    split_prompt_ids,\n'
        ')\n',
    ),
    # 阈值测试：把"探针凭据"换成两笔可复跑的真 bf16 数字，并如实报余量
    (
        'def test_prefix_parity_default_threshold_matches_probe():\n'
        '    """默认阈值必须容得下 bf16 实测漂移（0.6B/CPU 探针相对漂移 ≈5e-3），又不放过换人级偏差。"""\n'
        '    probe_measured_ratio = 0.005          # 探针凭据：max_abs_drift=1.256e-01 / 分数尺度 25.202\n'
        '    assert DEFAULT_DRIFT_RATIO > probe_measured_ratio, "默认阈值不该把真模型的正常漂移判成弃缓存"\n'
        '    assert DEFAULT_DRIFT_RATIO < 0.5, "阈值也不能松到能放过换人级偏差"\n'
        '    runner = PrefixRunner(FakeEngine(perturb=probe_measured_ratio / 10.0), readout=readout,\n'
        '                          verify="always")\n'
        '    runner.ask(PREFIX, SUFFIXES[0])\n'
        '    assert runner.ask(PREFIX, SUFFIXES[0]).recomputed is False, "探针量级漂移应放行"\n',
        'def test_prefix_parity_default_threshold_matches_probe():\n'
        '    """默认阈值必须容得下两笔可复跑的真 bf16 实测，又不放过换人级偏差（数字与命令一起挂）。"""\n'
        '    short_state = 4.98e-3      # `.probe_p207_prefix.py`：1.256e-01 / 25.202（state 782 token 英文）\n'
        '    long_state = 1.57e-2       # `.p207_run.py` run …-7b07-2 step2：3.846e-01 / 24.495（2065 token 中文）\n'
        '    worst = max(short_state, long_state)\n'
        '    assert DEFAULT_DRIFT_RATIO > worst, "默认阈值不该把真模型的正常漂移判成弃缓存"\n'
        '    margin = DEFAULT_DRIFT_RATIO / worst\n'
        '    assert 1.0 < margin < 2.0, f"最差那笔余量 {margin:.2f} 倍：阈值偏紧，长窗云端复标后回调"\n'
        '    print(f"[B1c] 阈值 0.02 对实测两笔余量：短证据 {DEFAULT_DRIFT_RATIO / short_state:.2f} 倍 / "\n'
        '          f"长证据 {margin:.2f} 倍（长证据那一笔已接近贴线，判负另靠 argmax 一票否决兜底）")\n'
        '    runner = PrefixRunner(FakeEngine(perturb=worst / 10.0), readout=readout, verify="always")\n'
        '    runner.ask(PREFIX, SUFFIXES[0])\n'
        '    assert runner.ask(PREFIX, SUFFIXES[0]).recomputed is False, "实测量级漂移应放行"\n',
    ),
]

NEW_TESTS = '''
# ------------------------------------------------- B1 补齐（接力波）：渲染切分 / 拷贝守卫 / 换人否决
class FakeTok:
    """字符级假编号器：只喂 `split_prompt_ids` 需要的两件事（encode + 对话模板），不碰真权重。

    模板形态与真 Qwen 一样是"逐轮拼接"：`<|role|>正文<|end|>`，末尾按需补生成引导符。
    """

    def encode(self, text, add_special_tokens=False):        # 口径固定：特殊符由模板负责
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            enable_thinking=False):          # 口径固定：思维链关掉（同 P1）
        assert tokenize is False, "本域按文字口径拼接，编号一律交给 encode"
        out = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        return out + ("<|assistant|>" if add_generation_prompt else "")


class MergingTok(FakeTok):
    """跨接缝会"粘连成一格"的编号器：state 尾字符与后面那个换行合成一个符号——前缀就不再是前缀。"""

    def encode(self, text, add_special_tokens=False):
        ids: list[int] = []
        i = 0
        while i < len(text):
            if text[i] == "x" and i + 1 < len(text) and text[i + 1] == "\\n":
                ids.append(900)                              # "x"+换行 合成一格
                i += 2
                continue
            ids.append(ord(text[i]))
            i += 1
        return ids


def rendered_messages(state: str, question: str) -> list[dict[str, str]]:
    """照 P1 `render.user_content` 的接缝形状造一条渲染结果（Evidence 段 + 接缝 + 问题段）。"""
    user = "Evidence:\\n" + state + SPLIT_MARK + question + "\\nOptions:\\n1. a\\n2. b"
    return [{"role": "system", "content": "SYS-LINE"}, {"role": "user", "content": user}]


def test_prefix_incr_split_prompt_ids_is_lossless_strict_prefix():
    """B1b 的前提：证据段编号必须"一格不差地排在整段前面"，且接缝不许混进证据段。"""
    tok = FakeTok()
    msgs = rendered_messages("ledger row 7712", "which code is booked?")
    prefix_ids, suffix_ids, info = split_prompt_ids(tok, msgs)
    full = tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True))
    assert prefix_ids + suffix_ids == full, "切分必须无损拼回整段编号"
    assert info["shell_len"] == len(prefix_ids) and info["suffix_len"] == len(suffix_ids)
    prefix_text = "".join(chr(t) for t in prefix_ids)
    suffix_text = "".join(chr(t) for t in suffix_ids)
    assert "ledger row 7712" in prefix_text and "Question" not in prefix_text
    assert suffix_text.startswith(SPLIT_MARK), "接缝本身属于问题段，past 里不能掺题目文字"


def test_prefix_incr_split_guard_rejects_misaligned_prefix():
    """编号器在接缝处粘连 → 前缀不再是前缀：当场报错，绝不拿错位的 past 答题。"""
    with pytest.raises(CacheError, match="不对齐"):
        split_prompt_ids(MergingTok(), rendered_messages("case-x", "which code?"))


def test_prefix_incr_split_guard_rejects_missing_seam_or_shape():
    with pytest.raises(CacheError, match="接缝"):
        split_prompt_ids(FakeTok(), [{"role": "system", "content": "s"},
                                     {"role": "user", "content": "Evidence:\\n没有接缝"}])
    with pytest.raises(CacheError, match="messages"):
        split_prompt_ids(FakeTok(), [{"role": "user", "content": "只有一条 message"}])


def test_prefix_incr_ask_row_hits_after_rendered_split():
    """ask_row：同一份证据连发两问，第二问必须命中且只付问题段符号（渲染口径下的 B1b）。"""
    tok, state = FakeTok(), "ledger row 7712"
    runner = PrefixRunner(FakeEngine(), readout=readout, verify="off")
    r1 = runner.ask_row(tok, rendered_messages(state, "which code is booked?"), k=4)
    r2 = runner.ask_row(tok, rendered_messages(state, "is that the second mention?"), k=4)
    assert r1.hit is False and r1.prefix_tokens_fed > 0 and r1.tokens_are_suffix_only is True
    assert r2.hit is True and r2.key == r1.key, "同一份证据段必须折成同一个键"
    assert r2.prefix_tokens_fed == 0 and r2.suffix_tokens_fed == r2.suffix_len
    _, suf2, _ = split_prompt_ids(tok, rendered_messages(state, "is that the second mention?"))
    assert r2.suffix_tokens_fed == len(suf2), "命中路付的符号数=问题段应有长度"


def test_prefix_incr_ask_row_falls_back_when_split_fails():
    """切不动就老实整段重算：宁可多付符号，也不冒"错位 past 静默错答"的险。"""
    tok = FakeTok()
    runner = PrefixRunner(FakeEngine(), readout=readout, verify="off")
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "Evidence:\\n没有接缝"}]
    res = runner.ask_row(tok, msgs, k=4)
    full = tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True))
    assert res.hit is False and res.recomputed is True and res.uncacheable is False
    assert res.prefix_tokens_fed == 0
    assert res.full_tokens_fed == len(full) == res.tokens_fed


class _HybridLayer:
    """照 transformers 5.x `LinearAttentionLayer` 的形状：没有 keys/values，记忆态装在字典里。"""

    def __init__(self, *, populated: bool = True) -> None:
        self.conv_states = {0: torch.zeros(1, 2, 4)} if populated else {0: None}
        self.recurrent_states = {0: torch.ones(1, 2)} if populated else {0: None}
        self.is_initialized = populated


class _HybridCache:
    def __init__(self, layers) -> None:
        self.layers = list(layers)

    def update(self, keys, values, idx):                     # 拷不动，本就不该走到这里
        raise AssertionError("混合层没有 KV 可拷，重建时不该调用 update")


class _NeedsArgsCache:
    """构造器要参数的 past 壳：`type(past)()` 只会 TypeError，必须折成 CacheError 交给上层降级。"""

    def __init__(self, layer_types) -> None:
        self.layer_types = layer_types
        self.layers: list[object] = []


def test_prefix_core_clone_refuses_hybrid_past():
    """0.8B 混合门那种"线性记忆层"：只照 KV 重建会静默丢记忆 —— 必须报错，不许装没事。"""
    engine = HFEngine(None)
    with pytest.raises(CacheError, match="KV 之外"):
        engine.clone_past(_HybridCache([_HybridLayer()]))
    with pytest.raises(CacheError, match="只认 KV 形态"):
        engine.clone_past(_HybridCache([_HybridLayer(populated=False)]))
    assert engine.clone_past(None) is None


def test_prefix_core_clone_refuses_shell_needing_args():
    with pytest.raises(CacheError, match="无法按无参构造重建"):
        HFEngine(None).clone_past(_NeedsArgsCache(["full"]))


class UncacheableEngine(FakeEngine):
    """past 拷不动的引擎（混合记忆门的样子）：clone 一抛 CacheError，上层就该整段重算。"""

    def clone_past(self, past):
        raise CacheError("这份 past 压着线性记忆态，本波拷不动")


def test_prefix_incr_uncacheable_past_degrades_to_full_recompute():
    """拷不动 ≠ 算不了：降级整段重算，且把"白付的证据段"如实记在账上，第二次不再白算。"""
    engine = UncacheableEngine()
    runner = PrefixRunner(engine, readout=readout, verify="off")
    first = runner.ask(PREFIX, SUFFIXES[0])
    assert first.uncacheable is True and first.hit is False and first.recomputed is True
    assert first.prefix_tokens_fed == len(PREFIX), "证据段那一笔真花了，账上必须看得见"
    assert first.full_tokens_fed == len(PREFIX) + len(SUFFIXES[0])
    assert first.tokens_fed == first.prefix_tokens_fed + first.full_tokens_fed
    second = runner.ask(PREFIX, SUFFIXES[1])
    assert second.uncacheable is True and second.prefix_tokens_fed == 0, "第二次别再白算证据段"
    assert second.full_tokens_fed == len(PREFIX) + len(SUFFIXES[1])
    st = runner.stats()
    assert len(st["uncacheable_keys"]) == 1 and st["recompute_uncacheable"] == 2
    assert st["cache"]["items"] == 0, "拷不动的东西不许留在库里"
    assert tuple(second.hidden_last.shape) == (1, engine.dim)


class NearTieEngine(FakeEngine):
    """两个候选几乎同分、复用路刚好把名次翻掉的引擎：漂移很小，但答案换了人。"""

    def forward(self, input_ids, past=None, *, offset: int = 0) -> EngineOut:
        seen = list(past["tokens"]) if past is not None else []
        ids = seen + [int(t) for t in input_ids]
        vec = torch.zeros(1, self.dim)
        if past is None:                          # 整段重算路：1 号领先一丝
            vec[0, 0], vec[0, 1] = 1.0000, 0.9995
        else:                                     # 复用路：分差 1e-3 级，名次却翻过来了
            vec[0, 0], vec[0, 1] = 0.9990, 1.0001
        new_past = {"tokens": ids, "keys": torch.zeros(1, len(ids), 4),
                    "values": torch.zeros(1, len(ids), 4)}
        self.calls.append({"fed": len(input_ids), "used_past": past is not None, "offset": offset})
        return EngineOut(vec, new_past, len(input_ids))


def test_prefix_parity_argmax_flip_is_vetoed_even_with_small_drift():
    """B1c 硬要求：漂移远小于阈值也不行——名次换人就弃缓存重算（换人没有"差得不多"这种豁免）。"""
    runner = PrefixRunner(NearTieEngine(), readout=readout, verify="off")
    runner.ask(PREFIX, SUFFIXES[0])                       # 冷启入库，好把"命中路"走实
    res = runner.ask(PREFIX, SUFFIXES[0], verify="always")
    rep = runner.verdicts[-1]
    assert rep.drift_ratio < runner.drift_ratio_threshold, "单看漂移确实很小（正是容易放过的形状）"
    assert rep.argmax_same is False and rep.passed is False
    assert res.recomputed is True and res.key in runner.stats()["quarantined"]
    assert runner.cache.get(res.key) is None, "换人的 past 立刻出库"
    print(f"[B1c-veto] drift_ratio={rep.drift_ratio:.3e} < 阈值 {rep.threshold}，但 argmax 换人 → "
          f"passed={rep.passed}，scores_reuse={rep.scores_reuse} scores_full={rep.scores_full}")


'''

C1_ANCHOR = "# ================================================================== C1 因果滑窗"


def main() -> None:
    src = PC.read_text(encoding="utf-8")
    for i, (old, new) in enumerate(PC_EDITS, 1):
        n = src.count(old)
        assert n == 1, f"补丁③PC#{i} 锚点命中 {n} 次：{old[:60]!r}"
        src = src.replace(old, new)
    PC.write_text(src, encoding="utf-8")

    tsrc = TS.read_text(encoding="utf-8")
    for i, (old, new) in enumerate(TS_EDITS, 1):
        n = tsrc.count(old)
        assert n == 1, f"补丁③TS#{i} 锚点命中 {n} 次：{old[:60]!r}"
        tsrc = tsrc.replace(old, new)
    assert tsrc.count(C1_ANCHOR) == 1, "C1 分节锚点必须唯一"
    tsrc = tsrc.replace(C1_ANCHOR, NEW_TESTS.lstrip("\n") + C1_ANCHOR)
    TS.write_text(tsrc, encoding="utf-8")
    print(f"补丁③落地：prefix_cache {len(PC_EDITS)} 处 / test_longctx "
          f"{len(TS_EDITS)} 处 + 新增 {NEW_TESTS.count(chr(10) + 'def test_')} 个用例")


if __name__ == "__main__":
    main()
