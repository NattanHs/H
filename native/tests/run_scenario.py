"""uso: run_scenario.py PASTA_DO_PACOTE CENARIO SAIDA.npz   — roda um cenário e grava a saída e o estado."""
import importlib
import sys
import time
from pathlib import Path

import numpy as np

pasta, nome, saida = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, pasta)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenarios  # noqa: E402

eng = importlib.import_module("lagotracks.engine")
c = scenarios.Ctx(eng)
t0 = time.perf_counter()
scenarios.SCENARIOS[nome](c)
dt = time.perf_counter() - t0
kind = type(c.e).__name__
st = max(len(s) for s in c.st)
states = np.array([s + [-9] * (st - len(s)) for s in c.st], dtype=np.float64)
np.savez(saida, out=np.concatenate(c.out), st=states, secs=dt, kind=kind)
print(f"{nome}: {kind} {dt:.2f}s")
