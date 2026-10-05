"""p2-07 tests/test_longctx.py 第二轮修订：假引擎的位置通道口径 + 两处测试自身的错。"""
from pathlib import Path

P = Path("tests/test_longctx.py")
s = P.read_text()


def rep(old, new, label):
    global s
    assert s.count(old) == 1, f"锚点失配[{label}]: 命中 {s.count(old)} 次"
    s = s.replace(old, new, 1)
    print("ok", label)


# ① 假引擎：hidden 各通道都压到 O(0.1) 量级（阈值才好讲），位置通道改成"末位的全局位置"
rep('''    def forward(self, input_ids, past=None, *, offset: int = 0) -> EngineOut:
        seen = list(past["tokens"]) if past is not None else []
        ids = seen + [int(t) for t in input_ids]
        vec = torch.zeros(1, self.dim)
        vec[0, 0] = sum(ids) / 1000.0
        vec[0, 1] = float(len(ids))
        vec[0, 2] = float(max(ids)) if ids else 0.0
        vec[0, 3] = float(offset)
        if past is not None and self.perturb:
            vec[0, 0] += self.perturb''',
    '''    def forward(self, input_ids, past=None, *, offset: int = 0) -> EngineOut:
        seen = list(past["tokens"]) if past is not None else []
        fed = [int(t) for t in input_ids]
        ids = seen + fed
        vec = torch.zeros(1, self.dim)
        vec[0, 0] = sum(ids) / 1000.0
        vec[0, 1] = len(ids) / 1000.0
        vec[0, 2] = (max(ids) / 1000.0) if ids else 0.0
        # 末位的全局位置：带 past 时是 offset+len-1，整段重算时是 len-1 —— 两条路必须同值，
        # 位置传错（增量前向忘了抬 offset）就会在这里露出来
        vec[0, 3] = ((offset + len(fed) - 1) if past is not None else len(fed) - 1) / 1000.0
        if past is not None and self.perturb:
            vec[0, 0] += self.perturb''',
    "fake_engine_position_channel")

rep('''def readout(hidden: torch.Tensor, k: int | None = None) -> torch.Tensor:
    """假读出：取前 k 个通道再放大 10 倍（够把"相对漂移"这件事演成一个可断言的数）。"""
    n = hidden.size(-1) if k is None else int(k)
    return hidden[:, :n].float() * 10.0''',
    '''def readout(hidden: torch.Tensor, k: int | None = None) -> torch.Tensor:
    """假读出：取前 k 个通道当候选分（量级 ≈0.1，漂移阈值讲起来直观）。"""
    n = hidden.size(-1) if k is None else int(k)
    return hidden[:, :n].float()''',
    "readout_scale")

# ② force_full 对照用例：第一问是冷启（要付证据段），得先热一次再比
rep('''    engine = FakeEngine()
    runner = PrefixRunner(engine, readout=readout, verify="off")
    warm = runner.ask(PREFIX, SUFFIXES[1])
    runner.ask(PREFIX, SUFFIXES[1])
    direct = runner.ask(PREFIX, SUFFIXES[1], force_full=True)
    assert warm.prefix_tokens_fed == 0''',
    '''    engine = FakeEngine()
    runner = PrefixRunner(engine, readout=readout, verify="off")
    cold = runner.ask(PREFIX, SUFFIXES[1])            # 冷启：证据段算一次并入库
    warm = runner.ask(PREFIX, SUFFIXES[1])            # 命中：只喂问题段
    direct = runner.ask(PREFIX, SUFFIXES[1], force_full=True)
    assert cold.prefix_tokens_fed == len(PREFIX) and not cold.hit
    assert warm.prefix_tokens_fed == 0 and warm.hit''',
    "force_full_baseline")

# ③ 长度凭据用例：units 是"口径单位"数（est-chars 档就是字数），不该拿 token 档位去等值断言
rep('''    assert nd["len_units"] == "est-chars", "无编号器时口径必须写 est-chars"
    assert nd["target_units"] == 8192 and nd["units"] > 0
    assert abs(nd["units"] - nd["target_units"]) / nd["target_units"] < 0.10, "正文长度须在档位 ±10% 内"
    assert nd["est_tokens"] > 0 and nd["chars"] > 0''',
    '''    assert nd["len_units"] == "est-chars", "无编号器时口径必须写 est-chars"
    assert nd["target_units"] == round(8192 * longctx.CHARS_PER_TOKEN_EST)   # 档位×折算率
    assert abs(nd["units"] - nd["target_units"]) / nd["target_units"] < 0.10, "正文长度须在档位 ±10% 内"
    assert abs(nd["est_tokens"] - 8192) / 8192 < 0.10, "折算回 token 口径也要在 ±10% 内"
    assert nd["chars"] == nd["units"] == len(row["sample"]["state"])
    assert "tokens" not in nd, "没有编号器就不许冒出真 token 字段（不虚报）"''',
    "length_ledger_assert")

P.write_text(s)
print("written", P, len(s.splitlines()), "lines")
