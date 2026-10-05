from sys1.eval import run as E
recs = E.load_axis_records("quality", limit=2)
print("n=", len(recs))
for r in recs[:2]:
    print("keys:", sorted(r.keys()))
    print("id:", r.get("id"))
    print("sample keys:", sorted((r.get("sample") or {}).keys()))
    s = r.get("sample") or {}
    print("state:", repr(str(s.get("state"))[:120]))
    print("questions:", {k: str(v)[:100] for k, v in (s.get("questions") or {}).items()})
