#!/usr/bin/env python3
"""
teste_nativo.py - confere o motor nativo (C++/JUCE) no SEU computador.

Rode com o Python do app, dentro de D:\\LagoTracks (depois de aplicar_nativo.py):

    .\\.venv\\Scripts\\python teste_nativo.py                      # informações + igualdade com o motor Python
    .\\.venv\\Scripts\\python teste_nativo.py --audio              # + toca de verdade na placa (30 s) com carga pesada
    .\\.venv\\Scripts\\python teste_nativo.py --audio --motor python   # o mesmo teste com o motor/saída ANTIGOS (para comparar)

Opções do teste de áudio:  --tipo DirectSound|"Windows Audio"|...   --dispositivo "parte do nome"   --bloco 512   --segundos 30
                           --stems PASTA (usa seus arquivos em vez de sons sintéticos)

O teste de áudio toca em silêncio quase total? Não: toca as faixas de verdade na saída escolhida. Abaixe o volume antes.
"""
from __future__ import annotations

import argparse
import gc
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SR = 48000


def audio_sintetico(seed: int, secs: float, stereo: bool, amp: float, tone: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(secs * SR)
    t = np.arange(n) / SR
    base = amp * np.sin(2 * np.pi * tone * t) + 0.1 * amp * rng.standard_normal(n)
    if stereo:
        r = amp * np.sin(2 * np.pi * tone * 1.5 * t + 0.4) + 0.1 * amp * rng.standard_normal(n)
        base = np.stack([base, r], 1)
    return np.clip(np.rint(base * 32767), -32768, 32767).astype(np.int16)


def carregar_stems(pasta: str, n: int):
    from lagotracks import loader
    exts = (".wav", ".flac", ".mp3", ".ogg")
    arqs = sorted(p for p in Path(pasta).rglob("*") if p.suffix.lower() in exts)[:n]
    if not arqs:
        sys.exit(f"Nenhum áudio em {pasta}")
    return [loader.load_audio(str(p), SR) for p in arqs]


def montar(eng, Track, dados):
    tracks = []
    for i, d in enumerate(dados):
        t = Track(f"t{i}", d)
        t.role = "click" if i == 0 else ("guide" if i == 1 else "music")
        tracks.append(t)
    eng.set_tracks(tracks)
    return tracks


def igualdade(Engine, PyEngine, Track, dados) -> bool:
    """Toca o mesmo roteiro nos dois motores e compara a saída bloco a bloco (sem pad/cama: são aleatórios)."""
    print("\n[1] Igualdade com o motor Python (mesmos comandos nos dois):")
    res = []
    for cls in (PyEngine, Engine):
        e = cls(SR)
        e.limiter_on = True
        tr = montar(e, Track, [d.copy() for d in dados])
        e.bpm, e.grid_offset, e.click_mode, e.click_volume = 118.0, 2500, 2, 0.6
        saidas = []

        def r(secs):
            left = int(secs * SR)
            while left > 0:
                n = min(left, (512, 333, 1024, 700)[len(saidas) % 4])
                o = np.zeros((n, 2), dtype=np.float32)
                e.render_into(o)
                saidas.append(o)
                left -= n
        r(0.6)
        e.play()
        r(2.0)
        tr[min(2, len(tr) - 1)].volume = 0.4
        tr[min(3, len(tr) - 1)].mute = True
        r(1.0)
        e.jump(int(1.0 * SR))
        r(2.5)
        e.set_loop(int(0.5 * SR), int(1.5 * SR))
        r(3.0)
        e.clear_loop()
        e.pause()
        r(0.5)
        e.play()
        r(1.5)
        e.stop()
        r(0.5)
        res.append(np.concatenate(saidas))
    d = float(np.abs(res[0] - res[1]).max())
    rms = float(np.sqrt(np.mean(res[0].astype(np.float64) ** 2)))
    ok = res[0].shape == res[1].shape and d < 1e-4
    print(f"    diferença máxima {d:.2e} (sinal rms {rms:.3f})  ->  {'OK, som idêntico' if ok else 'DIFERENTE!'}")
    return ok


def teste_audio(args, Engine, PyEngine, Track, AudioOutput, nativo_ok) -> None:
    from lagotracks import native_engine as ne
    print(f"\n[2] Teste de áudio de verdade ({args.segundos:.0f} s, motor: {args.motor}, tipo: {args.tipo}):")
    if args.stems:
        dados = carregar_stems(args.stems, 16)
    else:
        dados = [audio_sintetico(i, 90.0, i % 3 == 0, 0.12, 110 + 37 * i) for i in range(16)]
    usar_nativo = args.motor == "nativo"
    cls = Engine if usar_nativo else PyEngine
    e = cls(SR)
    e.limiter_on = True
    tracks = montar(e, Track, dados)
    e.bpm = 120.0
    e.click_mode = 0
    e.pad_mode = 1
    e.pad_under = 0.4
    settings_dev = None
    if args.dispositivo or args.tipo:
        try:
            import sounddevice as sd
            api = {"directsound": "directsound", "windows audio": "wasapi"}.get(args.tipo.lower().split(" (")[0], "")
            for i, d in enumerate(sd.query_devices()):
                nome_api = sd.query_hostapis(d["hostapi"])["name"].lower().replace(" ", "")
                if d["max_output_channels"] >= 2 and (not args.dispositivo or args.dispositivo.lower() in d["name"].lower()) \
                        and (not api or api in nome_api) and "primary" not in d["name"].lower():
                    settings_dev = i
                    break
        except Exception as exc:  # noqa: BLE001
            print("    (não consegui listar dispositivos pelo sounddevice:", exc, ")")
    print("    dispositivo escolhido (índice do sounddevice):", settings_dev)
    import os
    os.environ["LAGOTRACKS_JUCE_TIPO"] = args.tipo if usar_nativo else ""
    out = AudioOutput(e, device=settings_dev, blocksize=args.bloco, wasapi_exclusive="exclusive" in args.tipo.lower())
    out.start()
    print("    saída:", getattr(out, "backend", "PortAudio (motor antigo)"), "| latência ~%.0f ms" % out.latency_ms)
    e.play()
    time.sleep(1.0)

    parar = threading.Event()

    def pesado():                      # imita a tela, o carregamento e a transposição disputando a GIL
        x = 0
        while not parar.is_set():
            for i in range(20000):
                x += i * i % 7
            time.sleep(0)
    for _ in range(2):
        threading.Thread(target=pesado, daemon=True).start()

    rng = np.random.default_rng(3)
    t0 = time.perf_counter()
    base_falhas = out.underruns
    if usar_nativo:
        e.native_core.reset_diagnostics()
    ops = 0
    while time.perf_counter() - t0 < args.segundos:
        t = tracks[int(rng.integers(0, len(tracks)))]
        op = int(rng.integers(0, 6))
        if op == 0:
            t.volume = float(rng.uniform(0.3, 1.0))
        elif op == 1:
            t.shifted = audio_sintetico(int(rng.integers(0, 999)), 90.0, t.data.ndim == 2, 0.12, 300.0)
        elif op == 2:
            t.shifted = None
        elif op == 3:
            e.jump(int(rng.integers(0, 80 * SR)))
        elif op == 4:
            gc.collect()
        else:
            e.pad_mode = int(rng.integers(0, 3))
        if usar_nativo:
            e.poll()
        ops += 1
        time.sleep(0.02)
    parar.set()
    falhas = out.underruns - base_falhas
    print(f"    {ops} operações de interface durante o teste")
    if usar_nativo:
        s = e.native_stats()
        print(f"    bloco de áudio: máx {s['maximo_ms']:.2f} ms (limite {512 / SR * 1000:.2f} ms para bloco 512) | blocos lentos: {s['blocos_lentos']}")
        print(f"    pad: falhas {s['pad_falhas']} | cama: {s['cama_falhas']}")
    print(f"    FALHAS DE ÁUDIO relatadas pela saída: {falhas}   <- quanto menor, melhor (0 = perfeito)")
    print(f"    status: {out.last_status or '-'}")
    e.stop()
    out.stop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", action="store_true")
    ap.add_argument("--motor", default="nativo", choices=["nativo", "python"])
    ap.add_argument("--tipo", default="DirectSound")
    ap.add_argument("--dispositivo", default="")
    ap.add_argument("--bloco", type=int, default=512)
    ap.add_argument("--segundos", type=float, default=30.0)
    ap.add_argument("--stems", default="")
    args = ap.parse_args()

    print("Python:", sys.version.split()[0], sys.executable)
    try:
        from lagotracks import native_engine as ne
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Não achei o pacote lagotracks aqui ({exc}). Rode dentro de D:\\LagoTracks.")
    if ne._native is None:
        sys.exit("O módulo nativo (_native*.pyd) não foi encontrado em lagotracks\\ para este Python.\n"
                 "Confira se baixou a versão certa (cp312 = Python 3.12 etc.) e rode aplicar_nativo.py.")
    print("Módulo nativo:", ne._native.build_info())
    print("Tipos de driver:", ne._native.AudioDevice.types())
    for d in ne._native.AudioDevice.list():
        print(f"   [{d.type}] {d.name}")

    from lagotracks import engine as E
    from lagotracks.audio_io import AudioOutput
    Engine, Track = E.Engine, E.Track
    PyEngine = Engine.__mro__[1] if Engine.__mro__[0].__name__ == "NativeEngine" else Engine
    if PyEngine is Engine:
        sys.exit("O app está usando o motor Python (LAGOTRACKS_SEM_NATIVO definido ou aplicar_nativo.py não aplicado).")

    dados = [audio_sintetico(i, 9.0, i % 3 == 0, 0.4, 110 + 37 * i) for i in range(6)]
    ok = igualdade(Engine, PyEngine, Track, dados)
    if args.audio:
        teste_audio(args, Engine, PyEngine, Track, AudioOutput, ok)
    print("\nResultado da igualdade:", "OK" if ok else "FALHOU")


if __name__ == "__main__":
    main()
