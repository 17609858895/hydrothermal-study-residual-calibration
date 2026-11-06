"""Compile source text and verify unique max-min anchors without model training."""
import ast
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    sources = sorted(ROOT.glob("*.py")) + sorted((ROOT / "scripts").rglob("*.py"))
    for path in sources:
        compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
    path = ROOT / "scripts" / "11_active_anchor.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "greedy_maxmin")
    scope = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    select = scope["greedy_maxmin"]
    C = np.zeros((8, 2)); M = np.full((8, 3), "same"); idx = np.arange(8)
    for k in range(1, 9):
        chosen = select(C, M, idx, k)
        assert len(chosen) == k and len(np.unique(chosen)) == k
    C = np.array([[0., 0.], [0., 0.], [1., 0.], [1., 0.], [2., 0.], [2., 0.]])
    chosen = select(C, np.full((6, 3), "same"), np.arange(6), 5)
    assert len(np.unique(chosen)) == 5
    C = np.arange(30, dtype=float).reshape(15, 2); idx = np.array([2, 4, 8, 11, 14])
    chosen = select(C, np.full((15, 3), "same"), idx, len(idx))
    assert set(chosen) == set(idx)
    print(f"PASS: compiled {len(sources)} Python sources; unique anchors for repeated conditions")


if __name__ == "__main__":
    main()
