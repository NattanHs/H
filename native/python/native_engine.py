"""Motor de áudio nativo (C++) do LagoTracks, ligado ao resto do app.

O núcleo em C++ (`_native`) faz tudo o que o Engine em Python fazia na thread de áudio: mixagem, transporte, click,
limitador. A saída de áudio é do JUCE (`AudioDevice`): a thread de áudio nunca entra no Python, então a tela, o
carregamento de stems e o coletor de lixo não tiram mais tempo do áudio.

O que continua em Python (fora da thread de áudio): a síntese do pad (`PadSynth`) e da cama de ministração
(`MinistryBed`). Uma thread comum as sintetiza um pouco à frente e escreve em filas; o motor só lê as filas.
Se o Python atrasar, só o pad falha; as faixas nunca.

`install()` é chamado no fim de engine.py. Sem o módulo `_native` (ou com LAGOTRACKS_SEM_NATIVO=1) tudo segue em Python.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

try:                                                    # o módulo compilado fica ao lado deste arquivo
    from . import _native
except Exception:                                       # noqa: BLE001
    _native = None

NATIVO = _native is not None and not os.environ.get("LAGOTRACKS_SEM_NATIVO")

_ROLE_CODE = {"music": 0, "click": 1, "guide": 2}
BLOCO_PRODUTOR = 512                                    # quadros por síntese do pad/cama
ALVO_FILA = 9600                                        # ~0,2 s: quanto o produtor tenta manter à frente

# atributos do Engine que viram parâmetros do núcleo (nome no Engine -> nome no núcleo)
_PARAMS = ("master", "bpm", "grid_offset", "beats_per_bar", "quantize", "click_mode", "click_volume", "click_route",
           "pad_mode", "pad_volume", "pad_route", "pad_under", "pad_fade_out", "limiter_on", "ministry_on",
           "ambient_silenced")


# ===================================================================== Track
def _track_property(name, conv):
    key = "_p_" + name

    def get(self):
        return self.__dict__[key]

    def set_(self, v):
        self.__dict__[key] = v
        nt = self.__dict__.get("_nt")
        if nt is not None:
            setattr(nt, name, conv(v))
    return property(get, set_)


def _role_conv(v):
    return _ROLE_CODE.get(v, 3)


def _install_track(Track):
    """Faz os atributos de áudio do Track repassarem para a faixa nativa quando ela está no motor."""
    for n, c in (("volume", float), ("level", float), ("mute", bool), ("solo", bool), ("melodic", bool), ("route", int)):
        setattr(Track, n, _track_property(n, c))
    setattr(Track, "role", _track_property("role", _role_conv))

    def peak_get(self):
        nt = self.__dict__.get("_nt")
        return float(nt.peak) if nt is not None else self.__dict__.get("_p_peak", 0.0)

    def peak_set(self, v):
        self.__dict__["_p_peak"] = v
        nt = self.__dict__.get("_nt")
        if nt is not None:
            nt.peak = float(v)
    Track.peak = property(peak_get, peak_set)

    def data_get(self):
        return self.__dict__.get("_p_data")

    def data_set(self, v):
        self.__dict__["_p_data"] = v
        nt = self.__dict__.get("_nt")
        if nt is not None:
            self.__dict__["_core"].set_data(nt, v)
    Track.data = property(data_get, data_set)

    def sh_get(self):
        return self.__dict__.get("_p_shifted")

    def sh_set(self, v):
        self.__dict__["_p_shifted"] = v
        nt = self.__dict__.get("_nt")
        if nt is not None:
            self.__dict__["_core"].set_shifted(nt, v)
    Track.shifted = property(sh_get, sh_set)


# ===================================================================== produtor do pad/cama
class _Produtor(threading.Thread):
    def __init__(self, engine):
        super().__init__(name="lagotracks-pad", daemon=True)
        self.e = engine
        self.prov = engine._ring
        self._halt = threading.Event()
        self.errors = 0
        self.last_error = ""

    def stop(self):
        self._halt.set()

    def run(self):
        e, prov = self.e, self.prov
        blk = BLOCO_PRODUTOR
        while not self._halt.is_set():
            did = False
            try:
                if prov.pad_needed() and prov.pad_space() >= blk and prov.pad_fill() < ALVO_FILA:
                    e.pad.set_chord(e.key_pc, e.key_minor, e.pad_triad, instant=prov.pad_instant())
                    pl, pr = e.pad.render(blk)
                    prov.pad_write(pl, pr)
                    did = True
                if prov.bed_needed() and prov.bed_space() >= blk and prov.bed_fill() < ALVO_FILA:
                    bed = e.ministry
                    bed.mono = prov.bed_mono()
                    ml, mr, mc = bed.render(blk, prov.bed_want(), prov.bed_click_volume())
                    prov.bed_write(ml, mr, mc, bool(bed.active))
                    did = True
            except Exception as exc:  # noqa: BLE001  (o produtor nunca derruba o app)
                self.errors += 1
                self.last_error = repr(exc)
                time.sleep(0.05)
            if not did:
                time.sleep(0.003)


# ===================================================================== Engine nativo
def _make_engine_class(PyEngine, Track):
    class NativeEngine(PyEngine):
        native_core = None

        def __init__(self, sample_rate: int = 48000):
            object.__setattr__(self, "_core", None)
            super().__init__(sample_rate)
            core = _native.Core(int(sample_rate))
            object.__setattr__(self, "_core", core)
            object.__setattr__(self, "native_core", core)
            core.set_click_waves(np.ascontiguousarray(self._click_accent, dtype=np.float32),
                                 np.ascontiguousarray(self._click_normal, dtype=np.float32))
            for n in _PARAMS:
                core.set(n, float(getattr(self, n)))
            self._ring = _native.RingProvider()
            self._offline = _native.PyProvider(self._cb_pad, self._cb_active, self._cb_bed)
            self._mode = None
            self._use("offline")
            self._producer = None

        # ---- parâmetros: o que é escrito no Engine também vai para o núcleo ----
        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if name in _PARAMS:
                core = self.__dict__.get("_core")
                if core is not None:
                    core.set(name, float(getattr(self, name)))

        # ---- estado publicado pelo núcleo (somente leitura) ----
        def _ro(self, _v):                                 # o Engine em Python atribui estes no __init__
            pass

        position = property(lambda s: s._core.position if s._core else 0, _ro)
        playing = property(lambda s: s._core.playing if s._core else False, _ro)
        armed = property(lambda s: s._core.armed if s._core else False, _ro)
        beat_in_bar = property(lambda s: s._core.beat_in_bar if s._core else 0, _ro)
        beat_counter = property(lambda s: s._core.beat_counter if s._core else 0, _ro)
        ended_count = property(lambda s: s._core.ended_count if s._core else 0, _ro)
        peak_l = property(lambda s: s._core.peak_l if s._core else 0.0, _ro)
        peak_r = property(lambda s: s._core.peak_r if s._core else 0.0, _ro)
        duck = property(lambda s: s._core.duck if s._core else 1.0, _ro)

        @property
        def clock(self) -> int:
            return self._core.clock

        @property
        def total_frames(self) -> int:
            return self._core.total_frames

        @property
        def pending_jump(self):
            return self._core.pending_jump

        @property
        def loop_region(self):
            return self._core.loop_region

        # ---- faixas ----
        def _bind(self, t) -> None:
            if t.__dict__.get("_nt") is not None:
                return
            core = self._core
            nt = core.make_track()
            for n, c in (("volume", float), ("level", float), ("mute", bool), ("solo", bool), ("melodic", bool),
                         ("route", int)):
                setattr(nt, n, c(getattr(t, n)))
            nt.role = _role_conv(t.role)
            nt.peak = float(t.peak)
            t.__dict__["_nt"] = nt
            t.__dict__["_core"] = core
            if t.data is not None:
                core.set_data(nt, t.data)
            if t.shifted is not None:
                core.set_shifted(nt, t.shifted)

        @staticmethod
        def _unbind(t) -> None:
            nt = t.__dict__.get("_nt")
            if nt is not None:
                t.__dict__["_p_peak"] = float(nt.peak)       # o medidor guarda o último valor (como no Python)
            t.__dict__["_nt"] = None

        def add_track(self, track) -> None:
            self._tracks = self._tracks + (track,)
            self._bind(track)
            self._core.add_track(track._nt)

        def set_track_data(self, track, data) -> None:
            track.data = data                              # o Track repassa ao núcleo (e zera o ganho)
            if track.__dict__.get("_nt") is None:
                return
            self._core.collect()

        def remove_track(self, track) -> None:
            nt = track.__dict__.get("_nt")
            self._tracks = tuple(t for t in self._tracks if t is not track)
            if nt is not None:
                self._core.remove_track(nt)
                self._unbind(track)

        def clear_tracks(self) -> None:
            old = self._tracks
            self._tracks = ()
            self._core.clear_tracks()
            for t in old:
                self._unbind(t)

        def set_tracks(self, tracks) -> None:
            tracks = tuple(tracks)
            for t in self._tracks:
                if t not in tracks:
                    self._unbind(t)
            for t in tracks:
                self._bind(t)
            self._tracks = tracks
            self._core.set_tracks([t._nt for t in tracks], True)

        # ---- transporte ----
        def play(self, delay_frames: int = 0) -> None:
            self.ambient_silenced = False
            self._core.play(max(0, int(delay_frames)))

        def cancel_start(self) -> None:
            self._core.cancel_start()

        def pause(self) -> None:
            self._core.pause()

        def stop(self, silence: bool = False) -> None:
            if silence:
                self.ambient_silenced = True
            self._core.stop(bool(silence))

        def seek(self, frames: int) -> None:
            self._core.seek(int(frames))

        def jump(self, frame: int) -> None:
            self._core.jump(int(frame))

        def cancel_jump(self) -> None:
            self._core.cancel_jump()

        def set_loop(self, start: int, end: int) -> None:
            self._core.set_loop(int(start), int(end))

        def clear_loop(self) -> None:
            self._core.clear_loop()

        # ---- pad e cama: fonte Python ----
        def _cb_pad(self, instant, n):
            self.pad.set_chord(self.key_pc, self.key_minor, self.pad_triad, instant=instant)
            return self.pad.render(n)

        def _cb_active(self):
            return bool(self.ministry.active)

        def _cb_bed(self, n, want, vol, mono):
            bed = self.ministry
            bed.mono = mono
            return bed.render(n, want, vol)

        def _use(self, mode: str) -> None:
            if mode == self._mode:
                return
            self._core.set_provider(self._ring if mode == "ring" else self._offline)
            self._mode = mode

        def start_native_audio_helpers(self) -> None:
            """Liga a síntese do pad/cama em segundo plano (chamado ao abrir a saída de áudio)."""
            self._use("ring")
            if self._producer is None or not self._producer.is_alive():
                self._producer = _Produtor(self)
                self._producer.start()

        def stop_native_audio_helpers(self) -> None:
            p, self._producer = self._producer, None
            if p is not None:
                p.stop()
                p.join(timeout=1.0)
            self._use("offline")

        # ---- áudio ----
        def render_into(self, out: np.ndarray) -> None:
            """Renderiza offline (testes) ou pela saída antiga do Python (reserva)."""
            if out.dtype == np.float32 and out.flags.c_contiguous and out.ndim == 2 and out.shape[1] == 2:
                self._core.render(out)
            else:
                tmp = np.zeros((out.shape[0], 2), dtype=np.float32)
                self._core.render(tmp)
                out[:] = tmp

        def poll(self) -> None:
            self._core.collect()

        def native_stats(self) -> dict:
            c = self._core
            d = {"ultimo_ms": c.last_render_us / 1000.0, "maximo_ms": c.max_render_us / 1000.0,
                 "blocos_lentos": c.slow_blocks, "chamadas": c.render_calls}
            r = self._ring
            d["pad_falhas"], d["cama_falhas"] = r.pad_underruns, r.bed_underruns
            return d

    return NativeEngine


def install(PyEngine, Track):
    """Devolve o Engine a usar: o nativo, se o módulo compilado existir; senão o próprio Engine em Python."""
    if not NATIVO:
        return PyEngine
    _install_track(Track)
    return _make_engine_class(PyEngine, Track)


# ===================================================================== saída de áudio
def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _resolve_device(device, wasapi_exclusive: bool):
    """Converte o índice do sounddevice (como o app guarda) em (tipo JUCE, nome do dispositivo)."""
    tipo, nome = "DirectSound", ""
    forcado = os.environ.get("LAGOTRACKS_JUCE_TIPO", "").strip()      # ex.: "Windows Audio" (para testar outro driver)
    if device is None:
        return forcado or tipo, nome
    try:
        import sounddevice as sd
        d = sd.query_devices(device)
        api = sd.query_hostapis(d["hostapi"])["name"].lower()
        nome = str(d["name"])
        if "asio" in api:
            tipo = "ASIO"
        elif "wasapi" in api:
            tipo = "Windows Audio (Exclusive Mode)" if wasapi_exclusive else "Windows Audio"
    except Exception:  # noqa: BLE001
        pass
    if any(k in _norm(nome) for k in ("primary", "primario")):
        nome = ""                                           # "Primary Sound Driver" = padrão do sistema
    return forcado or tipo, nome


def _match(tipo: str, nome: str):
    """Acha o dispositivo do JUCE que corresponde ao nome do PortAudio (nomes diferem um pouco)."""
    if not nome:
        return tipo, ""
    import difflib
    devs = [d for d in _native.AudioDevice.list() if d.type == tipo] or [d for d in _native.AudioDevice.list()
                                                                         if d.type == "DirectSound"]
    if not devs:
        return tipo, ""
    alvo = _norm(nome)
    best, best_r = None, 0.0
    for d in devs:
        n = _norm(d.name)
        if n == alvo:
            return d.type, d.name
        r = difflib.SequenceMatcher(None, alvo, n).ratio()
        if alvo and (alvo in n or n in alvo):
            r = max(r, 0.9)
        if r > best_r:
            best, best_r = d, r
    if best is not None and best_r >= 0.6:
        return best.type, best.name
    return tipo, ""


def make_audio_output_class(PyAudioOutput):
    class AudioOutput(PyAudioOutput):
        """Saída de áudio: JUCE quando o motor é nativo; senão a de sempre (sounddevice)."""

        def __new__(cls, engine, *args, **kwargs):
            if getattr(engine, "native_core", None) is not None:
                return NativeAudioOutput(engine, *args, **kwargs)      # (não é instância de cls: o __init__ de cls não roda)
            return object.__new__(cls)

    class NativeAudioOutput:
        def __init__(self, engine, device=None, blocksize: int = 512, wasapi_exclusive: bool = False,
                     asio_channel: int = 0, asio_native_block: bool = True):
            self.engine = engine
            self.device = device
            self.blocksize = blocksize
            self.wasapi_exclusive = wasapi_exclusive
            self.asio_channel = asio_channel
            self.asio_native_block = asio_native_block
            self._dev = None
            self._fallback = None
            self._prefetch = None
            self.last_status = ""
            self.backend = ""

        # a classe devolvida por __new__ não passa por AudioOutput.__init__: este é o __init__ dela
        def start(self) -> None:
            e = self.engine
            if sys.getswitchinterval() > 0.0005:
                sys.setswitchinterval(0.0005)               # o produtor do pad e a tela trocam de vez com mais agilidade
            tipo, nome = _resolve_device(self.device, self.wasapi_exclusive)
            try:
                tipo, nome = _match(tipo, nome)
                dev = _native.AudioDevice(e.native_core)
                err = dev.open(tipo, nome, int(e.sample_rate), int(self.blocksize))
            except Exception as exc:  # noqa: BLE001
                dev, err = None, f"{type(exc).__name__}: {exc}"
            if err:
                self.last_status = f"JUCE: {err}"
                self._start_fallback(err)
                return
            self._dev = dev
            self.backend = f"JUCE · {dev.description}"
            e.start_native_audio_helpers()
            try:                                            # leitura antecipada do disco (stems mapeadas do cache)
                from .prefetch import Prefetcher
                self._prefetch = Prefetcher(e)
                self._prefetch.start()
            except Exception:  # noqa: BLE001
                self._prefetch = None

        def _start_fallback(self, err: str) -> None:
            """A saída antiga (PortAudio) toca o mesmo motor nativo, só que pelo Python."""
            out = PyAudioOutput(self.engine, self.device, self.blocksize, self.wasapi_exclusive,
                                self.asio_channel, self.asio_native_block)
            out.start()
            self._fallback = out
            self.backend = "PortAudio (reserva)"
            self.last_status = f"JUCE não abriu ({err}); usando PortAudio"

        def stop(self) -> None:
            pf, self._prefetch = self._prefetch, None
            if pf is not None:
                pf.stop()
            fb, self._fallback = self._fallback, None
            if fb is not None:
                fb.stop()
            dev, self._dev = self._dev, None
            if dev is not None:
                dev.close()
            self.engine.stop_native_audio_helpers()

        @property
        def running(self) -> bool:
            if self._fallback is not None:
                return self._fallback.running
            return self._dev is not None and bool(self._dev.running)

        @property
        def latency_ms(self) -> float:
            if self._fallback is not None:
                return self._fallback.latency_ms
            return float(self._dev.latency_ms) if self._dev is not None else 0.0

        @property
        def underruns(self) -> int:
            if self._fallback is not None:
                return self._fallback.underruns
            return int(self._dev.xruns) if self._dev is not None and self._dev.xruns > 0 else 0

        @property
        def rotulo(self) -> str:
            """Texto curto para a barra de status da janela."""
            return "ativo (reserva PortAudio)" if self._fallback is not None else "nativo (JUCE)"

        def diagnostico(self) -> str:
            s = self.engine.native_stats()
            return (f"{self.backend} · bloco máx {s['maximo_ms']:.2f} ms · lentos {s['blocos_lentos']} · "
                    f"pad falhas {s['pad_falhas']}")

    return AudioOutput
