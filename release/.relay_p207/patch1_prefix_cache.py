# -*- coding: utf-8 -*-
"""p2-07 接力补丁①：serving/prefix_cache.py —— 数字溯源 + argmax 一票否决 + 混合 past 拷贝守卫。

用法：cd release && .venv/bin/python .relay_p207/patch1_prefix_cache.py
每条替换都断言锚点恰好命中 1 次（大块改写前先数锚点，防错位粘贴）。
"""
from __future__ import annotations

from pathlib import Path

TARGET = Path("/Users/pengweiye/Documents/codes/system-one/release/serving/prefix_cache.py")

PAIRS: list[tuple[str, str]] = [
    # ① 模块头【为什么】：漂移凭据从"一笔"改成"两笔可复跑"
    (
        '    差一点点没关系"——bf16 下 past 复用的漂移是可测的（0.6B 探针 max_abs_drift≈1.3e-1，\n'
        '    分数尺度≈25，相对漂移≈5e-3），一旦某条 state 漂移大到换人，静默复用就是静默错答。\n',
        '    差一点点没关系"——bf16 下 past 复用的漂移是可复跑的两笔账：`.probe_p207_prefix.py`\n'
        '    （782 token 英文证据）相对漂移 4.98e-3；`.p207_run.py` 落到 run\n'
        '    1006-p207-longctx-local-half-7b07-2 step2（2065 token 中文证据）1.57e-2 —— 后者已贴着\n'
        '    0.02 阈值。一旦某条 state 漂移到换人，静默复用就是静默错答（弃缓存只是慢，错答是事故）。\n',
    ),
    # ② 阈值注释：把"4 倍余量"这种经不起复跑的修辞换成实测余量
    (
        '#: 复用路 vs 重算路的相对漂移阈值（`max_abs_drift / 分数尺度`）。定标凭据见 p2-07 探针：\n'
        '#: 0.6B/bf16/CPU 上实测相对漂移 ≈5e-3，argmax 不换人；0.02 留 4 倍余量又足够揪出换人级偏差。\n',
        '#: 复用路 vs 重算路的相对漂移阈值（`max_abs_drift / 分数尺度`）。两笔真 bf16 凭据（同机可复跑）：\n'
        '#: `.probe_p207_prefix.py` → 0.6B/CPU，state 782 token（英文证据）：1.256e-01 / 25.202 ≈ 4.98e-3；\n'
        '#: `.p207_run.py` → run 1006-p207-longctx-local-half-7b07-2 step2，state 2065 token（中文证据）：\n'
        '#: 3.846e-01 / 24.495 ≈ 1.57e-2。所以 0.02 只包住最差那一笔（余量 1.27 倍，不是 4 倍）：证据越\n'
        '#: 长漂移越大，这个阈值偏紧、可能误弃本可复用的 state —— 留待云端长窗实测再定标。为不把"偏紧"\n'
        '#: 变成"漏判"，判负另加 argmax 换人一票否决（见 `compare_parity`）。\n',
    ),
    # ③ _maybe_clone 之后补一个"这层还压着别的东西吗"的探测器
    (
        'def _maybe_clone(value: Any) -> Any:\n'
        '    """张量就 `.clone()`，其他（None、标量、缓存壳）原样交回。"""\n'
        '    return value.clone() if isinstance(value, torch.Tensor) else value\n',
        'def _maybe_clone(value: Any) -> Any:\n'
        '    """张量就 `.clone()`，其他（None、标量、缓存壳）原样交回。"""\n'
        '    return value.clone() if isinstance(value, torch.Tensor) else value\n'
        '\n'
        '\n'
        'def _has_extra_state(lay: Any) -> bool:\n'
        '    """这一层除了 keys/values 还压着别的张量状态吗（线性记忆/卷积态就是这种形状）。\n'
        '\n'
        '    白话：小抄盒子里除了"键值两叠纸"，可能还夹着别的东西——线性记忆那种"读到哪儿就更新到\n'
        '    哪儿"的进度条。进度条没抄走就交下一问，等于让模型拿半截记忆答题：答得漂亮却全错。\n'
        '    """\n'
        '    holder = lay if isinstance(lay, dict) else getattr(lay, "__dict__", {})\n'
        '    for name, val in dict(holder).items():\n'
        '        if name in ("keys", "values"):\n'
        '            continue\n'
        '        if isinstance(val, torch.Tensor):\n'
        '            if val.numel() > 0:\n'
        '                return True\n'
        '        elif isinstance(val, dict):        # conv_states / recurrent_states：{槽号: 张量|None}\n'
        '            if any(isinstance(x, torch.Tensor) and x.numel() > 0 for x in val.values()):\n'
        '                return True\n'
        '        elif isinstance(val, (list, tuple)):\n'
        '            if any(isinstance(x, torch.Tensor) and x.numel() > 0 for x in val):\n'
        '                return True\n'
        '    return False\n',
    ),
    # ④ clone_past：拷不动的两类 past 一律抛 CacheError，交给上层降级，绝不静默少拷
    (
        '    def clone_past(self, past: Any) -> Any:\n'
        '        """深拷一份 past：前向会就地追加，缓存里的原件一旦被污染，后续命中全是脏的。\n'
        '\n'
        '        transformers 5.x 的 `Cache` 没有 `copy()`（实测），所以按层重建：新建一个空\n'
        '        Cache，把每层的 keys/values `.clone()` 后按原层号 `update()` 回去。\n'
        '        """\n'
        '        if past is None:\n'
        '            return None\n'
        '        layers = getattr(past, "layers", None)\n'
        '        if layers is None:                              # 退回 (keys, values) 元组列表的老口径\n'
        '            if isinstance(past, list):\n'
        '                return [(_maybe_clone(k), _maybe_clone(v)) for k, v in past]\n'
        '            return past\n'
        '        fresh = type(past)()\n'
        '        for idx, lay in enumerate(layers):\n'
        '            keys, values = getattr(lay, "keys", None), getattr(lay, "values", None)\n'
        '            if isinstance(keys, torch.Tensor) and isinstance(values, torch.Tensor):\n'
        '                fresh.update(keys.clone(), values.clone(), idx)\n'
        '        return fresh\n',
        '    def clone_past(self, past: Any) -> Any:\n'
        '        """深拷一份 past：前向会就地追加，缓存里的原件一旦被污染，后续命中全是脏的。\n'
        '\n'
        '        白话：小抄要留原件、也要留复件——复件交出去被人涂改，原件才还能给下一问接着用。\n'
        '\n'
        '        transformers 5.x 的 `Cache` 没有 `copy()`（实测），所以按层重建：新建一个空\n'
        '        Cache，把每层的 keys/values `.clone()` 后按原层号 `update()` 回去。重建不了的两类\n'
        '        一律抛 `CacheError`（由 `PrefixRunner.ask` 降级成整段重算），绝不静默少拷：\n'
        '        ① past 壳的构造器要参数（`EncoderDecoderCache` 一类），空壳重建只会 TypeError；\n'
        '        ② 层里带着 KV 之外的状态（`LinearAttentionLayer` 的 conv_states/recurrent_states，\n'
        '        0.8B 混合门正是这种形状）——只照 KV 重建就把递归记忆整层丢了，比没缓存更坏。\n'
        '        """\n'
        '        if past is None:\n'
        '            return None\n'
        '        layers = getattr(past, "layers", None)\n'
        '        if layers is None:                              # 退回 (keys, values) 元组列表的老口径\n'
        '            if isinstance(past, list):\n'
        '                return [(_maybe_clone(k), _maybe_clone(v)) for k, v in past]\n'
        '            return past\n'
        '        try:\n'
        '            fresh = type(past)()\n'
        '        except TypeError as exc:                        # 空壳建不出来：这型 past 本波不认\n'
        '            raise CacheError(\n'
        '                f"past 壳 {type(past).__name__} 无法按无参构造重建（原始错误：{exc}）："\n'
        '                "本波只覆盖 DynamicCache 形态，改走整段重算"\n'
        '            ) from exc\n'
        '        for idx, lay in enumerate(layers):\n'
        '            keys, values = getattr(lay, "keys", None), getattr(lay, "values", None)\n'
        '            if isinstance(keys, torch.Tensor) and isinstance(values, torch.Tensor):\n'
        '                fresh.update(keys.clone(), values.clone(), idx)\n'
        '            elif _has_extra_state(lay):\n'
        '                raise CacheError(\n'
        '                    f"第 {idx} 层（{type(lay).__name__}）压着 KV 之外的状态（线性记忆的 "\n'
        '                    f"conv_states/recurrent_states 一类）：只照 KV 重建会静默丢记忆，本波改走"\n'
        '                    f"整段重算；0.8B 混合门要复用，得等 D7 三件套第二件（GDN 旁路）落成"\n'
        '                )\n'
        '        return fresh\n',
    ),
    # ⑤ AskResult：多一个"这份 past 拷不动"的自述位
    (
        '    argmax_same: bool | None = None\n'
        '    elapsed_s: float = 0.0\n',
        '    argmax_same: bool | None = None\n'
        '    elapsed_s: float = 0.0\n'
        '    uncacheable: bool = False    # 引擎的 past 拷不动（混合记忆态）→ 本题只能整段重算\n',
    ),
    (
        '            "drift": self.drift, "drift_ratio": self.drift_ratio,\n'
        '            "argmax_same": self.argmax_same, "elapsed_s": round(self.elapsed_s, 4),\n',
        '            "drift": self.drift, "drift_ratio": self.drift_ratio,\n'
        '            "argmax_same": self.argmax_same, "elapsed_s": round(self.elapsed_s, 4),\n'
        '            "uncacheable": self.uncacheable,\n',
    ),
    # ⑥ Runner 记账：拷不动的键单独一本账（能力边界 ≠ 数值事故，两本账不许混）
    (
        '        self._quarantine: set[str] = set()\n'
        '        self._checked: set[str] = set()\n'
        '        self.verdicts: list[ParityReport] = []\n'
        '        self.recompute_forced = 0\n',
        '        self._quarantine: set[str] = set()\n'
        '        self._uncacheable: set[str] = set()\n'
        '        self._checked: set[str] = set()\n'
        '        self.verdicts: list[ParityReport] = []\n'
        '        self.recompute_forced = 0\n'
        '        self.recompute_uncacheable = 0\n',
    ),
    # ⑦ ask()：命中路包一层降级（拷不动 → 整段重算，已付的证据段符号照实记账）
    (
        '        same: bool | None = None\n'
        '        hit = False\n'
        '\n'
        '        if force_full or key in self._quarantine:\n'
        '            out = self._forward_full(list(prefix_ids) + list(suffix_ids))\n'
        '            full_fed = out.tokens\n'
        '            self.recompute_forced += 1\n'
        '            hidden, past = out.hidden_last, None\n'
        '        else:\n'
        '            past = self.cache.get(key)\n'
        '            hit = past is not None\n'
        '            if past is None:                            # 冷启：证据段算一次并入库\n'
        '                pout = self._forward_prefix(list(prefix_ids))\n'
        '                prefix_fed = pout.tokens\n'
        '                self.cache.put(key, self.engine.clone_past(pout.past))\n'
        '                past = pout.past\n'
        '            out = self._forward_suffix(list(suffix_ids), self.engine.clone_past(past), len(prefix_ids))\n'
        '            suffix_fed = out.tokens\n'
        '            hidden = out.hidden_last\n'
        '\n'
        '        # B1c 护栏：与整段重算比对；判负即拉黑名单 + 用重算结果覆盖本次答案\n'
        '        used_past = (not force_full) and (key not in self._quarantine)\n',
        '        same: bool | None = None\n'
        '        hit = False\n'
        '        uncacheable = False\n'
        '\n'
        '        if force_full or key in self._quarantine or key in self._uncacheable:\n'
        '            if key in self._uncacheable:\n'
        '                self.recompute_uncacheable += 1          # 拷不动的 past：每题都从整段重算走\n'
        '            out = self._forward_full(list(prefix_ids) + list(suffix_ids))\n'
        '            full_fed = out.tokens\n'
        '            self.recompute_forced += 1\n'
        '            hidden = out.hidden_last\n'
        '        else:\n'
        '            try:\n'
        '                past = self.cache.get(key)\n'
        '                hit = past is not None\n'
        '                if past is None:                        # 冷启：证据段算一次并入库\n'
        '                    pout = self._forward_prefix(list(prefix_ids))\n'
        '                    prefix_fed = pout.tokens\n'
        '                    self.cache.put(key, self.engine.clone_past(pout.past))\n'
        '                    past = pout.past\n'
        '                out = self._forward_suffix(list(suffix_ids), self.engine.clone_past(past),\n'
        '                                           len(prefix_ids))\n'
        '                suffix_fed = out.tokens\n'
        '                hidden = out.hidden_last\n'
        '            except CacheError:\n'
        '                # 引擎说"这份 past 拷不动"：证据段那一笔已经花了，照实记账，再补一次整段重算，\n'
        '                # 并把键记进 uncacheable——下一次别再白算一遍证据段\n'
        '                uncacheable = True\n'
        '                self._uncacheable.add(key)\n'
        '                self.cache.drop(key)\n'
        '                suffix_fed = 0\n'
        '                out = self._forward_full(list(prefix_ids) + list(suffix_ids))\n'
        '                full_fed = out.tokens\n'
        '                hit = False\n'
        '                hidden = out.hidden_last\n'
        '\n'
        '        # B1c 护栏：与整段重算比对；判负即拉黑名单 + 用重算结果覆盖本次答案\n'
        '        used_past = (not force_full) and (not uncacheable) and (key not in self._quarantine)\n',
    ),
    (
        '        else:\n'
        '            recomputed = bool(force_full) or key in self._quarantine\n',
        '        else:\n'
        '            recomputed = bool(force_full) or uncacheable or key in self._quarantine\n',
    ),
    (
        '            suffix_len=len(suffix_ids), drift=drift, drift_ratio=ratio, argmax_same=same,\n'
        '            elapsed_s=time.perf_counter() - t0,\n',
        '            suffix_len=len(suffix_ids), drift=drift, drift_ratio=ratio, argmax_same=same,\n'
        '            elapsed_s=time.perf_counter() - t0, uncacheable=uncacheable,\n',
    ),
    # ⑧ stats()：把 uncacheable 一起交出去
    (
        '            "quarantined": sorted(self._quarantine),\n'
        '            "verified_keys": len(self._checked),\n'
        '            "parity_failed": sum(1 for r in self.verdicts if not r.passed),\n'
        '            "recompute_forced": self.recompute_forced,\n',
        '            "quarantined": sorted(self._quarantine),\n'
        '            "uncacheable_keys": sorted(self._uncacheable),\n'
        '            "verified_keys": len(self._checked),\n'
        '            "parity_failed": sum(1 for r in self.verdicts if not r.passed),\n'
        '            "recompute_forced": self.recompute_forced,\n'
        '            "recompute_uncacheable": self.recompute_uncacheable,\n',
    ),
    # ⑨ compare_parity：判负加"换人一票否决"（spec 的 argmax 同是硬要求，不是参考项）
    (
        '    白话：分差要除以分数本身的尺度才可比——25 分量级上差 0.12 是小事，1 分量级上差\n'
        '    0.12 就够换人了。所以这里既报绝对差也报相对差，判负只看相对差（阈值可配）。\n',
        '    白话：分差要除以分数本身的尺度才可比——25 分量级上差 0.12 是小事，1 分量级上差\n'
        '    0.12 就够换人了。判负两条并排走：①相对漂移超阈值（阈值可配）；②名次换人——一票否决。\n'
        '    第二条不看幅度，因为"答成了另一个选项"没有"差得不多"这种豁免。\n',
    ),
    (
        '    drift = (s_reuse - s_full).abs().max().item()\n'
        '    scale = max(s_full.abs().mean().item(), 1e-6)\n'
        '    ratio = drift / scale\n'
        '    return ParityReport(\n'
        '        key=key, max_abs_drift=float(drift), score_scale=float(scale),\n'
        '        drift_ratio=float(ratio),\n'
        '        argmax_same=bool(int(s_reuse.argmax()) == int(s_full.argmax())),\n'
        '        passed=bool(ratio <= threshold), threshold=float(threshold),\n',
        '    drift = (s_reuse - s_full).abs().max().item()\n'
        '    scale = max(s_full.abs().mean().item(), 1e-6)\n'
        '    ratio = drift / scale\n'
        '    same = bool(int(s_reuse.argmax()) == int(s_full.argmax()))\n'
        '    return ParityReport(\n'
        '        key=key, max_abs_drift=float(drift), score_scale=float(scale),\n'
        '        drift_ratio=float(ratio),\n'
        '        argmax_same=same,\n'
        '        passed=bool(same and ratio <= threshold), threshold=float(threshold),\n',
    ),
]


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    for i, (old, new) in enumerate(PAIRS, 1):
        n = src.count(old)
        assert n == 1, f"补丁①#{i} 锚点命中 {n} 次（需恰好 1 次）：{old[:70]!r}"
        src = src.replace(old, new)
    TARGET.write_text(src, encoding="utf-8")
    print(f"补丁①落地：{TARGET.name} 共 {len(PAIRS)} 处替换")


if __name__ == "__main__":
    main()
