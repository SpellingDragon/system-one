"""p2-10 盘点探针：把已存在的 serving/server.py 真跑一遍（/health、/v1/systemone、批量、/stats）。

口径：CPU、禁 MPS、loader=minimal（p2-05 确立的 0.6B 替身侧门）。
只读盘点，不写任何产物；判"server.py 是真是假"的第一手凭据。
"""
from __future__ import annotations

import json
import time
import urllib.request

from serving.server import DecisionService, build_engine, spawn, verify_think_tail
from sys1.eval import run as E

STATE = "weather: clear sky at 07:00, humidity 40%, rising pressure"
REQ = {
    "id": "probe-1",
    "state": STATE,
    "questions": {
        "q1": {"type": "choice", "instructions": "Which option fits the evidence?",
               "criteria": {"a": "sunny", "b": "rain", "c": "cloudy"}},
        "q2": {"type": "noul", "instructions": "Does the evidence support the claim?"},
        "q3": {"type": "score", "instructions": "Rate the confidence.",
               "criteria": ["low", "medium", "high"]},
    },
}


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def post(url: str, payload: dict) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> None:
    t0 = time.time()
    eng = build_engine("0.6b", loader="minimal", device="cpu", dtype="float32")
    print(f"[load] {time.time() - t0:.1f}s describe={json.dumps(eng.describe(), ensure_ascii=False)}")
    print(f"[tail] ok={verify_think_tail(eng)!r}")

    srv = spawn(DecisionService(eng))
    print(f"[listen] {srv.base_url}")
    print("[health] " + json.dumps(get(srv.base_url + "/health"), ensure_ascii=False)[:400])

    r1 = post(srv.endpoint, REQ)
    print("[decide] keys=" + json.dumps(sorted(r1.keys())))
    print("[decide] answers=" + json.dumps(r1["answers"], ensure_ascii=False))
    print(f"[decide] ms={r1['ms']} semantics={r1['ms_semantics']} "
          f"uncal={r1.get('uncalibrated_qids')}")

    batch = [{"id": f"b{i}", "state": STATE, "questions": {
        "q1": {"type": "choice", "instructions": f"Which fits? #{i}",
               "criteria": {"a": "sunny", "b": "rain"}}}} for i in range(5)]
    rb = post(srv.base_url + "/v1/systemone/batch", {"requests": batch})
    print(f"[batch] n={len(rb['results'])} ms_list={[x['ms'] for x in rb['results']]} "
          f"ms_batch={rb['results'][0].get('ms_batch')}")
    print("[batch] answers0=" + json.dumps(rb["results"][0]["answers"], ensure_ascii=False))

    print("[stats] " + json.dumps(get(srv.base_url + "/stats"), ensure_ascii=False)[:400])

    try:
        post(srv.endpoint, {"id": "bad", "state": "s", "questions": {}})
    except urllib.error.HTTPError as exc:
        print(f"[400-check] code={exc.code} body={exc.read().decode('utf-8')[:160]}")

    recs = E.load_axis_records("quality", limit=2)
    print(f"[data] axis records n={len(recs)} ids={[r.get('id') for r in recs]}")
    for r in recs:
        s = r["sample"]
        print("   qtypes:", {k: v.get("type") for k, v in s["questions"].items()},
              "state_len:", len(str(s["state"])))
    srv.stop()
    print("[done]")


if __name__ == "__main__":
    main()
