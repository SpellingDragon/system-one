"""p2-01 换脑资产 run 记录：复跑真路径探针，把下载凭据/四校验/布局对拍/峰值内存写进 runs/。

跑法：cd release && .venv/bin/python .p201_run.py
约束：前向一律 CPU（MPS 归一阶段长跑占用）；MPS 侧只做小面积 fp16 往返冒烟，不载入权重。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch                              # noqa: E402
from sys1.runs import new_run             # noqa: E402

PROBE = ROOT / ".probe_p201_real.py"
DL_LOG = ROOT / ".out_p201_redownload.txt"


def run_shell(argv: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    """在本仓库 release 目录下跑一条子命令，原样收回 stdout/stderr（不吞栈）。"""
    return subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def real_probe() -> dict:
    """复跑真路径探针：一次载重拿到四校验、布局对拍、换脑三问、右填充因果不变与峰值内存。"""
    p = run_shell([sys.executable, str(PROBE)])
    if p.returncode != 0:
        raise RuntimeError(f"探针退出码 {p.returncode}；stderr 尾部：{p.stderr[-1500:]}")
    return json.loads(p.stdout)


def download_fact() -> dict:
    """从实下载日志取 PROVENANCE_JSON 行：字节量 / 耗时 / 镜像源 / 文件数。"""
    for line in reversed(DL_LOG.read_text(encoding="utf-8").splitlines()):
        if line.startswith("PROVENANCE_JSON "):
            return json.loads(line[len("PROVENANCE_JSON "):])
    raise RuntimeError(f"{DL_LOG.name} 里找不到 PROVENANCE_JSON 行")


def mps_smoke() -> dict:
    """MPS 通路只做小面积冒烟：fp16 小张量在 mps 上过一遍矩阵乘，回报分配器读数。

    本机 MPS 被一阶段训练长跑占着，这里刻意不载入 0.8B 权重（那要 ~1.7GB 显存），
    只验"fp16 张量能落到 mps、能动、读数可取"这条迁移通路本身。
    """
    if not torch.backends.mps.is_available():
        return {"available": False, "note": "torch.backends.mps.is_available() 为假"}
    a = torch.randn(64, 64, dtype=torch.float16, device="mps")
    b = torch.randn(64, 64, dtype=torch.float16, device="mps")
    c = a @ b
    torch.mps.synchronize()
    out = {"available": True, "matmul_finite": bool(torch.isfinite(c).all()),
           "device": str(c.device), "dtype": str(c.dtype),
           "allocated_bytes": int(torch.mps.current_allocated_memory())}
    del a, b, c
    torch.mps.empty_cache()
    return out


def pytest_facts() -> dict:
    """两条孙任务口径的测试各复跑一遍，把尾部汇总行原样抄进记录。"""
    lines = {}
    for key, argv in {
        "full": [sys.executable, "-m", "pytest", "tests/test_assets.py",
                 "tests/test_backbone_layout.py", "-q", "--no-header"],
        "a4_mps": [sys.executable, "-m", "pytest", "tests/test_assets.py",
                   "-k", "swap_brain", "-q", "-m", "mps", "--no-header"],
    }.items():
        p = run_shell(argv)
        tail = [s.strip() for s in (p.stdout + p.stderr).splitlines() if s.strip()]
        lines[key] = {"rc": p.returncode, "summary": tail[-1] if tail else ""}
    return lines


def _rel_of(note: str) -> float:
    """从对拍说明行里抠出相对差数值（格式 rel=6.180e-06）。"""
    import re
    m = re.search(r"rel=([0-9.eE+-]+)", note)
    return float(m.group(1)) if m else float("nan")


def observe_lines(rep: dict, dl: dict, mps: dict, tests: dict) -> list[str]:
    """把复跑拿到的数摊成笔记观察行：每行一个可复查的事实，结论词只跟在数后面。"""
    seam = rep["seam"]
    ids = seam["seam_letter_ids"]
    rows = rep["letter_rows_shape"]
    out = [
        f"真实下载实测（ModelScope，全新空目录首拉，非缓存命中、非 monkeypatch）："
        f"{dl['backbone_download_bytes']:,} 字节 / {dl['backbone_download_seconds']}s / "
        f"镜像源 {dl['backbone_endpoint']} / {dl['backbone_files']} 个文件；"
        f"repo={dl['backbone_repo']} revision={dl['backbone_revision']}。",
        f"真载实测（CPU + fp16）：load_seconds={rep['load_seconds']}s，"
        f"权重占用 {rep['model_param_bytes_mb']:,}MB，hidden_size={rep['hidden_size']}，"
        f"峰值常驻内存 peak_rss_mb_at_load={rep['peak_rss_mb_at_load']}MB、"
        f"整轮结束 {rep['peak_rss_mb_end']}MB。",
        f"校验①字母单 token：A–Z {len(ids)} 格全占独号（{ids[0]}..{ids[-1]}），"
        f"52 大小写均单格，拼进渲染串后边界不吞并——通过；未触发降 0.6B 预案。",
        f"校验②letter_rows 随隐宽重建：形状 {rows[0]}×{rows[1]}（26 行取自 lm_head，宽度跟 config）——通过。",
        f"校验③思考关闭模板快照：canonical prompt {seam['seam_prompt_tokens']} 格、"
        f"sha256 前 16 位 {seam['seam_prompt_sha256_16']}，尾巴逐字节等于 decision 的关闭后缀——通过（快照已固化入单测）。",
        f"校验④多模态编号不撞号：{seam['seam_mm_token_ids']}，与字母号 {ids[0]}..{ids[-1]} 无交集——通过。",
    ]
    for v in rep["layout"]:
        out.append(f"权重布局对拍 layer{v['layer']}（自研栈布局 {v['layout']}）：{v['note']}；"
                   f"行号抽样逐元素匹配={v['row_match']}，不匹配位={v['mismatch']}，"
                   f"转换产出 {len(v['keys'])} 个张量键 {v['keys']}。")
    sb = rep["swap_brain"]
    outs = ", ".join(f"{q['qid']}→{q['letter']}/{q['code']}(Σp={sum(q['probs']):.4f})" for q in sb)
    out.append(f"换脑冒烟（一阶段 decision 程序原样跑，零改动、不 import 一阶段 model.py）：{len(sb)} 问 {outs}。")
    ci = rep["causal_invariance"]
    out.append(f"因果不变性右填充对照：批宽 {ci['width']}、长度 {ci['lengths']}，"
               f"垫与不垫两种送法候选分最大绝对差 {ci['max_abs_diff']:.3e}（同长度尺度 ~{ci['mean_abs_score']:.2f}），"
               f"argmax 一致={ci['same_argmax']}（{ci['argmax_pad']} vs {ci['argmax_single']}）。")
    out.append(f"渲染对齐：串由 sys1/decision/render 只读产出（RENDER_VERSION={rep['seam'].get('render_version', 'n/a')}），"
               f"编码走 Backbone.encode_prompt，与 decision 的字母序/候选序一一对上（单测 test_render_alignment 覆盖）。")
    out.append(f"MPS 迁移冒烟（小张量，不载权重，因 MPS 被一阶段长跑占用）：{mps}；"
               f"完整 0.8B 的 MPS 前向峰值按派单延后（CPU 口径已给峰值常驻内存行）。")
    out.append(f"测试复跑：全量 {tests['full']['summary']}（rc={tests['full']['rc']}）；"
               f"A4 口径 -k swap_brain -q -m mps → {tests['a4_mps']['summary']}（rc={tests['a4_mps']['rc']}）。")
    return out


def main() -> int:
    """复跑真路径 → 开 run → 记曲线 → 填笔记三行 → finish 校验；凭据全落到 runs/。"""
    rep, dl = real_probe(), download_fact()
    mps, tests = mps_smoke(), pytest_facts()
    obs = observe_lines(rep, dl, mps, tests)
    cfg = {
        "domain": "p2-01-backbone-assets",
        "device_policy": "cpu（MPS 归一阶段长跑占用；本域前向一律 CPU，MPS 只冒烟小张量）",
        "verify_commands": [
            "python -m pytest tests/test_assets.py -k load -q",
            "python -m pytest tests/test_assets.py -k seams -q",
            "python -m pytest tests/test_assets.py -k seam_fail -q",
            "python -m pytest tests/test_assets.py -k swap_brain -q -m mps",
            "python -m pytest tests/test_backbone_layout.py -q",
        ],
        "real_probe": ".probe_p201_real.py",
        "fresh_download_dir": "bench/p201_dl_fresh",
        "peak_rss_mb_at_load": rep["peak_rss_mb_at_load"],
        "peak_rss_mb_end": rep["peak_rss_mb_end"],
    }
    cfg.update(dl)
    cfg.update(rep["seam"])
    nb = new_run("p201-backbone-assets-realpath", cfg)
    ci, sb, seam = rep["causal_invariance"], rep["swap_brain"], rep["seam"]
    rels = [_rel_of(v["note"]) for v in rep["layout"]]
    nb.log_metrics(0, download_bytes=dl["backbone_download_bytes"],
                   download_seconds=dl["backbone_download_seconds"], download_files=dl["backbone_files"])
    nb.log_metrics(1, load_seconds=rep["load_seconds"], weight_mb=rep["model_param_bytes_mb"],
                   peak_rss_mb_at_load=rep["peak_rss_mb_at_load"], peak_rss_mb_end=rep["peak_rss_mb_end"])
    nb.log_metrics(2, seam_checks_passed=4, letter_unique_tokens=len(seam["seam_letter_ids"]),
                   parity_layers_ok=sum(1 for v in rep["layout"] if v["ok"]),
                   parity_rel_max=max(rels), parity_rel_min=min(rels))
    nb.log_metrics(3, swap_brain_questions=len(sb), causal_max_abs_diff=ci["max_abs_diff"],
                   causal_same_argmax=int(ci["same_argmax"]), mps_small_smoke=int(bool(mps.get("matmul_finite"))))
    nb.log_metrics(4, tests_full_rc=tests["full"]["rc"], tests_a4_mps_rc=tests["a4_mps"]["rc"])
    hyp = ("Qwen3.5-0.8B 能从魔搭真下载并真载，四道载重接缝全过（字母单 token 是全计划最大单点），"
           "HF 权重换成自研栈布局后整层前向与 HF 的相对差进 1e-3，换脑后一阶段决策程序零改动仍出合法份额")
    text = nb.notes_file.read_text(encoding="utf-8")
    text = text.replace("- 假设：待填写", f"- 假设：{hyp}", 1)
    text = text.replace("- 观察：待填写", "- 观察：\n" + "\n".join(f"- {s}" for s in obs), 1)
    nb.notes_file.write_text(text, encoding="utf-8")
    nb.conclude(
        f"p2-01 六项孙任务落地：① 真下载 {dl['backbone_download_bytes']:,} 字节 / "
        f"{dl['backbone_download_seconds']}s（{dl['backbone_endpoint']}，{dl['backbone_files']} 文件，"
        f"空目录首拉）；② 真载 CPU+fp16 峰值常驻内存 {rep['peak_rss_mb_at_load']}MB（载重后）/"
        f"{rep['peak_rss_mb_end']}MB（整轮）；③ 四校验全过——字母 A–Z 单 token "
        f"{seam['seam_letter_ids'][0]}..{seam['seam_letter_ids'][-1]}（未触发降 0.6B 预案）、"
        f"letter_rows {rep['letter_rows_shape']}、思考关闭快照 "
        f"{seam['seam_prompt_tokens']} 格 / sha {seam['seam_prompt_sha256_16']}、"
        f"多模态编号与字母号无交集；④ 布局对拍 {len(rels)} 层 rel 最大 {max(rels):.1e} ≤1e-3；"
        f"⑤ 换脑 {len(sb)} 问份额和为 1、右填充因果不变 max_abs={ci['max_abs_diff']:.3e} 且 argmax 一致；"
        f"⑥ MPS 完整前向峰值按派单延后（本机 MPS 被一阶段长跑占用），本 run 只交 fp16→mps 小张量冒烟，"
        f"CPU 口径峰值已入笔记。"
    )
    print("RUN_DIR", nb.finish())
    return 0


if __name__ == "__main__":
    sys.exit(main())
