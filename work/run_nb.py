"""Minimal notebook executor (jupyter is not installed here): runs every code cell of a .ipynb in one namespace,
captures stdout and matplotlib figures per cell, and writes the executed notebook with outputs.
usage: py run_nb.py in.ipynb out.ipynb"""
import sys, io, json, base64, traceback, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src_path, out_path = sys.argv[1], sys.argv[2]
nb = json.load(open(src_path, encoding="utf-8"))

class Tee(io.TextIOBase):
    def __init__(self, real): self.real, self.buf = real, io.StringIO()
    def write(self, s): self.real.write(s); self.real.flush(); self.buf.write(s); return len(s)
    def flush(self): self.real.flush()

REAL = sys.stdout
cur_outputs = []
def _flush_text(tee):
    t = tee.buf.getvalue()
    if t: cur_outputs.append({"output_type": "stream", "name": "stdout", "text": t.splitlines(True)})
    tee.buf = io.StringIO()

def _show(*a, **k):
    tee = sys.stdout
    if isinstance(tee, Tee): _flush_text(tee)
    for num in plt.get_fignums():
        fig = plt.figure(num); b = io.BytesIO(); fig.savefig(b, format="png", dpi=90, bbox_inches="tight")
        cur_outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": [f"<Figure {num}>"]}})
    plt.close("all")
plt.show = _show

def _display(x):
    tee = sys.stdout
    if isinstance(tee, Tee): _flush_text(tee)
    txt = x.to_string() if hasattr(x, "to_string") else repr(x)
    cur_outputs.append({"output_type": "display_data", "metadata": {}, "data": {"text/plain": txt.splitlines(True)}})
    REAL.write(txt + "\n")

ns = {"__name__": "__main__", "display": _display}
count = 0
t0 = time.time()
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code": continue
    count += 1; cur_outputs = []
    code = "".join(cell["source"])
    tee = Tee(REAL); sys.stdout = tee
    ok = True
    try:
        exec(compile(code, f"<cell {i}>", "exec"), ns)
    except Exception:
        ok = False; tb = traceback.format_exc(); REAL.write(tb)
    finally:
        _flush_text(tee); sys.stdout = REAL
        if plt.get_fignums(): _show()
    if not ok:
        cur_outputs.append({"output_type": "error", "ename": "Error", "evalue": "see traceback", "traceback": tb.splitlines()})
    cell["outputs"] = cur_outputs; cell["execution_count"] = count
    json.dump(nb, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if not ok:
        print(f"CELL {i} FAILED after {time.time()-t0:.0f}s"); sys.exit(1)
print(f"notebook executed OK in {time.time()-t0:.0f}s -> {out_path}")
