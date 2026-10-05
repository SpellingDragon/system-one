"""p2-01 风险闸门探针：真分词器上的字母单 token + 边界 + 思考关闭 + 多模态编号四道接缝。

跑法：cd release && .venv/bin/python .probe_p201_seams2.py
预期：26 个字母编号与 StartLux letter_token_ids=[32..57] 同段。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from production.assets import (  # noqa: E402
    CANONICAL_ROW_SPEC,
    CANONICAL_STATE,
    SeamCheckError,
    canonical_prompt,
    check_letter_boundary,
    check_letters,
    check_mm_token_ids,
    check_think_off,
    letter_ids_from_mapping,
)
from sys1.decision.render import from_systemone, render  # noqa: E402

SNAP = Path("bench/ms_models/models/Qwen--Qwen3.5-0.8B/snapshots/master")


def main() -> int:
    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(SNAP))
    cfg = AutoConfig.from_pretrained(str(SNAP))
    out: dict = {"repo": str(SNAP)}
    try:
        mapping = check_letters(tok)
    except SeamCheckError as exc:
        print(json.dumps({"letters_ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    ids = letter_ids_from_mapping(mapping)
    out["letters_ok"] = True
    out["letter_ids_26"] = list(ids)
    out["lower_sample"] = {k: mapping[k] for k in "aeiz"}
    row = from_systemone(CANONICAL_STATE, dict(CANONICAL_ROW_SPEC))
    messages, order = render(row)
    _, suffix = check_think_off(tok, messages)
    out["think_off_suffix"] = suffix
    text, ntok, sha = canonical_prompt(tok)
    out["prompt_tokens"] = ntok
    out["prompt_sha16"] = sha
    check_letter_boundary(tok, text, ids)
    out["boundary_ok"] = True
    out["mm_token_ids"] = check_mm_token_ids(cfg, tok, ids)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
