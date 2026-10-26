"""Compara o motor Python original com o nativo em vários cenários.
uso: python run_equiv.py DIRETORIO_COM_O__native  [cenário ...]
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_pkgs  # noqa: E402
import scenarios  # noqa: E402

PATCH = '''

# --- motor nativo (C++/JUCE) ---
try:
    from .native_engine import install as _lt_install_native
    Engine = _lt_install_native(Engine, Track)
except Exception as _lt_exc:  # noqa: BLE001
    import sys as _lt_sys
    print("[LagoTracks] motor nativo indisponível:", repr(_lt_exc), file=_lt_sys.stderr)
'''


def main():
    so_dir = Path(sys.argv[1])
    quais = sys.argv[2:] or list(scenarios.SCENARIOS)
    original = HERE / "ref" / "engine_original.py"
    falhas = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ref = make_pkgs.montar(tmp / "ref", False, None, original, None)
        nat = make_pkgs.montar(tmp / "nat", True, so_dir, original, lambda t: t.rstrip("\n") + "\n" + PATCH)
        for nome in quais:
            print(f"== {nome}")
            outs = []
            for rotulo, pasta in (("py", ref), ("nativo", nat)):
                f = tmp / f"{nome}_{rotulo}.npz"
                r = subprocess.run([sys.executable, "-I", str(HERE / "run_scenario.py"), str(pasta), nome, str(f)],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    print(r.stdout, r.stderr)
                    falhas += 1
                    break
                print("  ", r.stdout.strip(), r.stderr.strip()[:200])
                outs.append(f)
            else:
                c = subprocess.run([sys.executable, str(HERE / "compare.py"), str(outs[0]), str(outs[1])],
                                   capture_output=True, text=True)
                print(c.stdout.rstrip())
                if c.returncode != 0:
                    falhas += 1
    print("\nRESULTADO:", "tudo igual" if not falhas else f"{falhas} cenário(s) com diferença")
    sys.exit(1 if falhas else 0)


main()
