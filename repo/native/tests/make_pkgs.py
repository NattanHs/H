"""Monta, numa pasta temporária, dois pacotes `lagotracks`: um com o motor Python original e outro com o nativo."""
import importlib.util
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def montar(destino: Path, nativo: bool, so_dir: Path | None, engine_src: Path, patch_engine) -> Path:
    pkg = destino / "lagotracks"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    for n in ("ministry.py", "synth.py", "limiter.py"):
        shutil.copyfile(HERE / "stubs" / n, pkg / n)
    texto = engine_src.read_text(encoding="utf-8")
    if nativo:
        texto = patch_engine(texto)
        shutil.copyfile(ROOT / "python" / "native_engine.py", pkg / "native_engine.py")
        for so in so_dir.glob("_native*"):
            shutil.copyfile(so, pkg / so.name)
    (pkg / "engine.py").write_text(texto, encoding="utf-8")
    return destino
