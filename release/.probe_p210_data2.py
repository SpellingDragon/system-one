from sys1.eval import run as E
recs = E.load_axis_records("quality", limit=3)
print("n=", len(recs))
for r in recs:
    s = r["sample"]
    qs = s["questions"]
    for qid, spec in qs.items():
        print(r["id"], "|", qid, "|", type(spec).__name__, "|", spec if not isinstance(spec, dict) else sorted(spec.keys()))
    print("   state type:", type(s["state"]).__name__, "len", len(str(s["state"])))
