"""Execute a notebook's code cells with THIS python (the animal-det env) and save outputs into the .ipynb.

Why it exists: the animal-det env has no ipykernel/jupyter, so `jupyter nbconvert --execute` cannot use it.
This small runner executes the cells top to bottom in one shared namespace, captures printed text, pandas tables,
matplotlib figures and IPython.display images, writes them as standard nbformat-4 outputs, and stops at the first error.
Once ipykernel is installed in animal-det you can ignore this file and run the notebooks in Jupyter as usual.

Usage: python notebooks/run_notebook.py notebooks/01_baseline_evaluation.ipynb
"""
import ast
import base64
import io
import json
import re
import sys
import time
import traceback
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")  # plt.show() is a no-op here; figures are captured

OUT = []  # outputs of the cell being executed


def _b64(b):
    return base64.b64encode(b).decode()


def _lines(s):
    return s.splitlines(keepends=True)


def add_display(obj):
    """Store one rich output for `obj` (DataFrame, Image, Markdown, text)."""
    if hasattr(obj, "_nb_bundle"):
        OUT.append({"output_type": "display_data", "data": obj._nb_bundle(), "metadata": {}})
    elif hasattr(obj, "to_html") and hasattr(obj, "to_string"):  # pandas DataFrame / Styler-less
        OUT.append({"output_type": "display_data", "metadata": {},
                    "data": {"text/html": _lines(obj.to_html()), "text/plain": _lines(obj.to_string())}})
    else:
        OUT.append({"output_type": "display_data", "metadata": {}, "data": {"text/plain": _lines(repr(obj))}})


class _Image:
    def __init__(self, filename=None, data=None, **kw):
        self.data = Path(filename).read_bytes() if filename else data

    def _nb_bundle(self):
        return {"image/png": _b64(self.data), "text/plain": ["<IPython.core.display.Image object>"]}


class _Markdown:
    def __init__(self, text):
        self.text = text

    def _nb_bundle(self):
        return {"text/markdown": _lines(self.text), "text/plain": _lines(self.text)}


class _HTML(_Markdown):
    def _nb_bundle(self):
        return {"text/html": _lines(self.text), "text/plain": _lines(self.text)}


def install_ipython_stub():
    ip = types.ModuleType("IPython"); disp = types.ModuleType("IPython.display")
    disp.display = lambda *objs, **kw: [add_display(o) for o in objs] and None
    disp.Image, disp.Markdown, disp.HTML = _Image, _Markdown, _HTML
    ip.display = disp
    ip.get_ipython = lambda: None   # libraries (ultralytics) probe for a running IPython; None = "not in IPython"
    ip.version_info, ip.__version__ = (8, 0, 0), "8.0.0"
    sys.modules["IPython"], sys.modules["IPython.display"] = ip, disp


class Capture(io.TextIOBase):
    def __init__(self, name):
        self.name, self.buf = name, []

    def write(self, s):
        self.buf.append(s); return len(s)

    def flush(self):
        pass

    def isatty(self):
        return False


def clean_stream(text):
    """Collapse carriage-return progress bars to their last state and drop ANSI colour codes."""
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
    out = []
    for ln in text.split("\n"):
        out.append(ln.split("\r")[-1] if "\r" in ln else ln)
    return "\n".join(out)


def run_cell(src, ns, count):
    OUT.clear()
    so, se = Capture("stdout"), Capture("stderr")
    old = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = so, se
    err = None
    value = None
    try:
        tree = ast.parse(src)
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        exec(compile(tree, f"<cell {count}>", "exec"), ns)
        if last is not None:
            value = eval(compile(last, f"<cell {count}>", "eval"), ns)
    except BaseException as e:  # noqa
        err = e
        tb = traceback.format_exc()
    finally:
        sys.stdout, sys.stderr = old
    outputs = []
    for s in (so, se):
        t = clean_stream("".join(s.buf))
        if t.strip():
            outputs.append({"output_type": "stream", "name": s.name, "text": _lines(t if t.endswith("\n") else t + "\n")})
    outputs.extend(OUT)
    if value is not None:
        if hasattr(value, "to_html") and hasattr(value, "to_string"):
            outputs.append({"output_type": "execute_result", "execution_count": count, "metadata": {},
                            "data": {"text/html": _lines(value.to_html()), "text/plain": _lines(value.to_string())}})
        elif hasattr(value, "_nb_bundle"):
            outputs.append({"output_type": "display_data", "metadata": {}, "data": value._nb_bundle()})
        else:
            outputs.append({"output_type": "execute_result", "execution_count": count, "metadata": {},
                            "data": {"text/plain": _lines(repr(value))}})
    for n in plt.get_fignums():
        buf = io.BytesIO(); plt.figure(n).savefig(buf, format="png", dpi=100, bbox_inches="tight")
        outputs.append({"output_type": "display_data", "metadata": {},
                        "data": {"image/png": _b64(buf.getvalue()), "text/plain": ["<Figure>"]}})
        plt.close(n)
    if err is not None:
        outputs.append({"output_type": "error", "ename": type(err).__name__, "evalue": str(err),
                        "traceback": _lines(tb)})
    return outputs, err


def main():
    import faulthandler, os
    if os.environ.get("NB_HANG_DUMP"):
        faulthandler.dump_traceback_later(int(os.environ["NB_HANG_DUMP"]), exit=True)
    path = Path(sys.argv[1]).resolve()
    nb = json.loads(path.read_text())
    install_ipython_stub()
    ns = {"__name__": "__main__"}
    sys.path.insert(0, str(path.parent))
    count, t0, failed = 0, time.time(), None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        count += 1
        src = "".join(c["source"])
        t = time.time()
        outs, err = run_cell(src, ns, count)
        c["outputs"], c["execution_count"] = outs, count
        print(f"[{path.name}] cell {i} ok in {time.time() - t:.1f}s" if err is None else f"[{path.name}] cell {i} FAILED: {err!r}",
              file=sys.__stderr__, flush=True)
        if err is not None:
            failed = i
            break
    nb["metadata"].setdefault("execution", {})["runner"] = "notebooks/run_notebook.py"
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"done in {time.time() - t0:.0f}s; {'FAILED at cell ' + str(failed) if failed is not None else 'all cells ran'}",
          file=sys.__stderr__)
    sys.exit(1 if failed is not None else 0)


if __name__ == "__main__":
    main()
