"""Teste de tempo real: thread de áudio simulada (C++) + a interface/carregamento brigando pela GIL.
uso: python stress_realtime.py PASTA_DO_PACOTE [segundos] [bloco] [modo: nativo|python]
"""
import gc
import importlib
import sys
import threading
import time
from pathlib import Path

import numpy as np

pasta = sys.argv[1]
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
bloco = int(sys.argv[3]) if len(sys.argv) > 3 else 512
sys.path.insert(0, pasta)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenarios  # noqa: E402

eng = importlib.import_module("lagotracks.engine")
SR = 48000
e = eng.Engine(SR)
NATIVO = type(e).__name__ == "NativeEngine"
if len(sys.argv) > 4 and sys.argv[4] == "py":
    assert not NATIVO
    sys.setswitchinterval(float(sys.argv[5]) if len(sys.argv) > 5 else 0.005)
e.limiter_on = True
tracks = []
for i in range(16):
    t = eng.Track(f"t{i}", scenarios.make_audio(i, 90.0, i % 3 == 0, 0.25, 110 + 37 * i))
    t.role = "click" if i == 0 else ("guide" if i == 1 else "music")
    tracks.append(t)
e.set_tracks(tracks)
e.bpm = 120.0
e.click_mode = 2
e.pad_mode = 1
e.pad_under = 0.4
e.key_pc = 2
if NATIVO:
    from lagotracks import native_engine as ne
    core = e.native_core
    dev = ne._native.AudioDevice(core)
    err = dev.open("test", "", SR, bloco)
    assert not err, err
    e.start_native_audio_helpers()
    e.play()
else:
    stats = {"late": 0, "max_late": 0.0, "max_render": 0.0, "calls": 0}
    halt = threading.Event()

    def audio_py():
        out = np.zeros((bloco, 2), dtype=np.float32)
        per = bloco / SR
        nxt = time.perf_counter() + per
        while not halt.is_set():
            t1 = time.perf_counter()
            e.render_into(out)
            dt = time.perf_counter() - t1
            stats["max_render"] = max(stats["max_render"], dt)
            stats["calls"] += 1
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            late = time.perf_counter() - nxt
            if late > per:
                stats["late"] += 1
                nxt = time.perf_counter()
            stats["max_late"] = max(stats["max_late"], late)
            nxt += per
    th_audio = threading.Thread(target=audio_py, daemon=True)
    e.play()
    th_audio.start()

stop = threading.Event()


def hog():                                  # disputa a GIL como a tela/o carregamento faziam
    x = 0
    while not stop.is_set():
        for i in range(20000):
            x += i * i % 7
        time.sleep(0)


hogs = [threading.Thread(target=hog, daemon=True) for _ in range(3)]
for h in hogs:
    h.start()

rng = np.random.default_rng(7)
t0 = time.perf_counter()
n = 0
if NATIVO:
    core.reset_diagnostics()
while time.perf_counter() - t0 < secs:
    op = int(rng.integers(0, 14))
    t = tracks[int(rng.integers(0, len(tracks)))]
    if op == 0: t.volume = float(rng.uniform(0, 1.2))
    elif op == 1: t.mute = bool(rng.integers(0, 2))
    elif op == 2: t.shifted = scenarios.make_audio(int(rng.integers(0, 999)), 90.0, t.data.ndim == 2, 0.25, 300.0)
    elif op == 3: t.shifted = None
    elif op == 4: e.set_track_data(t, scenarios.make_audio(int(rng.integers(0, 999)), 90.0, t.data.ndim == 2, 0.25, 200.0))
    elif op == 5: e.jump(int(rng.integers(0, 80 * SR)))
    elif op == 6:
        a = int(rng.integers(0, 70 * SR)); e.set_loop(a, a + int(rng.integers(SR, 8 * SR)))
    elif op == 7: e.clear_loop()
    elif op == 8: e.ministry_on = bool(rng.integers(0, 2))
    elif op == 9: e.pad_mode = int(rng.integers(0, 3))
    elif op == 10:
        e.set_tracks(tracks if rng.integers(0, 2) else tracks[:10])
    elif op == 11: gc.collect()
    elif op == 12: e.pause() if rng.integers(0, 2) else e.play()
    else: e.key_pc = int(rng.integers(0, 12))
    if NATIVO:
        e.poll()
    n += 1
    time.sleep(0.01)
stop.set()
if NATIVO:
    st = e.native_stats()
    print(f"[nativo] ops: {n} | posição: {e.position / SR:.1f}s | tocando: {e.playing}")
    print(f"áudio: bloco máx {st['maximo_ms']:.3f} ms | lentos(>80%) {st['blocos_lentos']} | chamadas {st['chamadas']}")
    print(f"falhas de áudio (thread atrasou um bloco): {dev.xruns} | {dev.last_status}")
    print(f"pad subexecuções: {st['pad_falhas']} | cama: {st['cama_falhas']} | produtor erros: {e._producer.errors if e._producer else 0}")
    e.stop_native_audio_helpers()
    dev.close()
else:
    halt.set()
    th_audio.join()
    print(f"[python] ops: {n} | posição: {e.position / SR:.1f}s")
    print(f"áudio: bloco máx {stats['max_render'] * 1000:.3f} ms | chamadas {stats['calls']}")
    print(f"falhas de áudio (thread atrasou um bloco): {stats['late']} | atraso máx do despertar {stats['max_late'] * 1000:.2f} ms")
