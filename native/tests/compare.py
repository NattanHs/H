"""Compara duas saídas .npz (Python x nativo)."""
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
oa, ob = a["out"], b["out"]
ok = True
if oa.shape != ob.shape:
    print("  formas diferentes", oa.shape, ob.shape)
    sys.exit(1)
d = np.abs(oa - ob)
mx = float(d.max())
rms_ref = float(np.sqrt(np.mean(oa.astype(np.float64) ** 2)))
rms_err = float(np.sqrt(np.mean((oa.astype(np.float64) - ob) ** 2)))
sa, sb = a["st"], b["st"]
bad_state = np.argwhere(np.abs(sa - sb) > 1e-6)
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 5e-5
print(f"  máx dif {mx:.2e} | rms sinal {rms_ref:.4f} | rms erro {rms_err:.2e} | estados diferentes: {len(bad_state)}")
if mx > tol:
    i = int(np.argmax(d.max(axis=1)))
    print(f"  !! acima da tolerância {tol:g}: pior no quadro {i} ({i / 48000:.3f}s)")
    ok = False
if len(bad_state):
    r, cidx = bad_state[0]
    print(f"  !! estado diverge: bloco {r} coluna {cidx}: py={sa[r, cidx]} nativo={sb[r, cidx]}")
    ok = False
sys.exit(0 if ok else 1)
