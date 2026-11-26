"""Experiment harness: runs the v7 notebook cells up to the cell that builds the identity tables (pools and SIFT come from the cache),
optionally pickles nothing, then exec's an experiment file in the same namespace.  usage: py exp_run.py <experiment.py> [stop_marker]"""
import sys, json, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: plt.close("all")
nb = json.load(open("submission10_v7.ipynb", encoding="utf-8"))
marker = sys.argv[2] if len(sys.argv) > 2 else "TB = {s: identity_table(POOL[s], TCODE[s]) for s in ALLS}"
skip = ["failure_taxonomy(sch)", "slice_report(sch)", "permutation_importance(m_id_tmp"]
ns = {"__name__": "__main__", "display": print}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    s = "".join(c["source"])
    if any(k in s for k in skip): continue
    exec(compile(s, f"<cell {i}>", "exec"), ns)
    if marker in s: break
print("=== notebook state ready, running", sys.argv[1], flush=True)
exec(compile(open(sys.argv[1], encoding="utf-8").read(), sys.argv[1], "exec"), ns)
