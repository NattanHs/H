"""Motor de áudio do LagoTracks.

Tudo roda em `Engine.render_into`, chamado pela thread de áudio. Não há
dependência de placa de som aqui: o motor só preenche buffers, então pode ser
testado offline.

Conceitos
---------
* Todas as faixas ficam na taxa do motor (sample_rate), em int16 estéreo.
* Existe uma "grade" de tempo do projeto: batida 0 em `grid_offset` (frames),
  com `bpm` e `beats_per_bar`. A grade é a mesma do click e do projeto.
* Com o projeto parado, o click e o pad rodam livres ("relógio ambiente").
  Ao dar play, o projeto é agendado para entrar na próxima batida/compasso
  do relógio ambiente, de modo que a grade da música coincida com a do click.
  Com o projeto tocando, o click acompanha a posição da música.
"""

from __future__ import annotations

import collections
import math
import threading

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from .ministry import MinistryBed
from .synth import PadSynth, make_click

ROUTE_STEREO, ROUTE_LEFT, ROUTE_RIGHT = 0, 1, 2
GATE_OFF, GATE_PAUSED, GATE_ALWAYS = 0, 1, 2
QUANT_IMMEDIATE, QUANT_BEAT, QUANT_BAR = 0, 1, 2

MAX_CHUNK = 4096
FADE_FRAMES = 240          # 5 ms a 48 kHz
XFADE_FRAMES = 2400        # 50 ms: troca entre original e transposta
_INV16 = np.float32(1.0 / 32768.0)

# Pad "eliminador de silêncio": o nível do pad sob a música acompanha o volume das faixas de música.
DUCK_QUIET_DB = -52.0      # música (RMS por canal) abaixo disto = silêncio: o pad sobe até o nível máximo sob a música
DUCK_LOUD_DB = -38.0       # música acima disto = presente: o pad some (fica quase inaudível)
DUCK_FLOOR = 0.02          # o que sobra do pad com a música alta (−34 dB: imperceptível, sem calar de vez)


class MasterLimiter:
    """Limitador de pico com antecipação para a saída (ex.: 2 ms), em blocos de qualquer tamanho.

    O sinal sai atrasado em `lookahead` frames; o ganho desce suave ANTES do pico (sem estalo) e nunca passa do teto.
    Sem nada acima do teto no bloco, só atrasa (custo baixo). Usado para o nivelador poder subir músicas baixas.
    """

    def __init__(self, lookahead: int = 96, ceiling: float = 0.97):
        self.w = int(lookahead) | 1
        self.ceiling = float(ceiling)
        self._prev = np.zeros((2 * self.w, 2), dtype=np.float32)
        self.reduction = 1.0                          # menor ganho aplicado no último bloco (1.0 = nada)

    def process(self, x: np.ndarray) -> np.ndarray:
        w, n = self.w, x.shape[0]
        ext = np.concatenate([self._prev, x], axis=0)
        self._prev = ext[n:].copy()
        peak = np.abs(ext).max(axis=1)
        if float(peak.max()) <= self.ceiling:
            self.reduction = 1.0
            return ext[w:w + n]
        need = np.ones(peak.shape[0], dtype=np.float32)
        over = peak > self.ceiling
        need[over] = (self.ceiling / peak[over]).astype(np.float32)
        g = minimum_filter1d(need, size=w, mode="constant", cval=1.0)
        g = uniform_filter1d(g, size=w, mode="constant", cval=1.0)
        g = np.minimum(g, need)[w:w + n]
        self.reduction = float(g.min())
        return ext[w:w + n] * g[:, None]


class Track:
    """Faixa de áudio. `data` é int16: mono (frames,) ou estéreo (frames, 2); None enquanto carrega ou descarregada."""

    _next_id = 1

    def __init__(self, name: str, data: np.ndarray | None = None, path: str | None = None):
        self.id = Track._next_id
        Track._next_id += 1
        self.name = name
        self.path = path
        self.member: str | None = None              # caminho dentro do ZIP de origem (None = arquivo solto)
        self.data = data
        self.volume = 1.0
        self.level = 1.0                            # ganho do nivelador (volume percebido igual entre as músicas)
        self.mute = False
        self.solo = False
        self.route = ROUTE_STEREO
        self.route_locked = False                   # True = você escolheu o canal (o padrão por papel não mexe mais)
        self.analyze = True
        self.role = "music"                         # "music" | "click" | "guide" (regência)
        self.error: str | None = None
        self.classified = False                     # melódica/percussiva já decidida (não refaz ao recarregar)
        # transposição: só faixas melódicas acompanham o tom escolhido
        self.melodic = True                       # False = percussiva (nunca transpõe)
        self.melodic_locked = False               # True = definido pelo usuário (não reclassificar)
        self.melodic_reason = ""
        self.peak = 0.0                             # pico do último bloco (0..1+), para o medidor da UI
        self.shifted: np.ndarray | None = None    # cópia transposta (mesmo tamanho de `data`)
        self.shifted_semitones = 0
        # estado da thread de áudio
        self._last_gain = 0.0
        self._reset_gain = True
        self._using: np.ndarray | None = None     # fonte tocada no bloco anterior
        self._xf_old: np.ndarray | None = None    # fonte antiga durante o crossfade
        self._xf_left = 0

    @property
    def frames(self) -> int:
        return 0 if self.data is None else int(self.data.shape[0])


def _route_add(out: np.ndarray, left: np.ndarray, right: np.ndarray, route: int) -> None:
    if route == ROUTE_LEFT:
        out[:, 0] += 0.5 * (left + right)
    elif route == ROUTE_RIGHT:
        out[:, 1] += 0.5 * (left + right)
    else:
        out[:, 0] += left
        out[:, 1] += right


class Engine:
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self._tracks: tuple[Track, ...] = ()
        self._tracks_lock = threading.Lock()
        self._total = 0
        self._cmds: collections.deque = collections.deque()

        # ---- parâmetros (escritos pela UI, lidos pela thread de áudio) ----
        self.master = 1.0
        self._bpm = 120.0
        self._grid_offset = 0           # frames
        self._bpb = 4
        self._quantize = QUANT_BAR
        self._grid_version = 0

        self._jump_req: int | None = None      # pulo pedido: acontece no fim do compasso (frame de destino)
        self._loop: tuple[int, int] | None = None   # repetir trecho [início, fim) em frames
        self._split: tuple[int, str] | None = None  # o bloco atual termina no gatilho: (destino, "jump"|"loop")
        self.ambient_silenced = False   # Parar silencia o click e o pad do ambiente até você tocar ou religar
        self._click_mode = GATE_OFF
        self._pad_mode = GATE_OFF
        self.click_volume = 0.7
        self.click_route = ROUTE_STEREO
        self.pad_volume = 0.5
        # Pad sob a música (eliminador de silêncio): fração (0..1) do volume do pad que sobe SÓ quando as faixas de música
        # estão em silêncio (contagem do começo, final) e some sozinho quando a música toca. 0 = como antes (em "Só pausado"
        # o pad some; em "Sempre" ele segue no volume cheio).
        self.pad_under = 0.0
        self._music_power = 0.0                  # potência das faixas de música no bloco (calculada ao mixar)
        self._music_env = 0.0                    # seguidor de envelope dessa potência (ataque rápido, soltura lenta)
        self.duck = 1.0                          # fator atual do pad sob a música (1 = música em silêncio; ~0 = tocando)
        self.pad_route = ROUTE_STEREO
        self.key_pc = 0
        self.key_minor = False
        self.pad_triad = False

        # ---- estado publicado para a UI ----
        self.position = 0               # frames
        self.playing = False
        self.armed = False              # play pedido, aguardando o compasso
        self.beat_in_bar = 0
        self.beat_counter = 0
        self.ended_count = 0                    # quantas vezes o projeto chegou ao fim sozinho (não por parar/pausar)
        self.peak_l = 0.0                       # medidores do master (pico por bloco, queda suave)
        self.peak_r = 0.0

        # ---- estado interno da thread de áudio ----
        self._clock = 0
        self._pos = 0
        self._playing = False
        self._pausing = False
        self._stop_to_start = False
        self._start_pending = False
        self._start_at = 0
        self._start_pos = 0
        self._start_not_before = 0               # relógio mínimo para o início (play com atraso: transições)
        self._fade_in_left = 0
        self._last_grid_version = 0
        self._ambient_beat = 0.0
        self._expected_b = 0.0
        self._last_beat = 0
        self._beat_init = False

        sr = sample_rate
        self._click_accent = make_click(sr, 1760.0, 0.9)
        self._click_normal = make_click(sr, 1175.0, 0.65)
        self._click_voices: list[list] = []      # [onda, índice, atraso]
        self._click_gain = 0.0
        self.pad = PadSynth(sr)
        self.ministry = MinistryBed(sr)          # cama de ministração (relógio próprio)
        self.ministry_on = False
        self._pad_gain = 0.0
        self._pad_decay = np.exp(-(np.arange(1, MAX_CHUNK + 1)) / (0.6 * sr))
        # Quando o projeto entra, o pad não corta: continua e some devagar (curva suave) durante `pad_fade_out` s.
        self.pad_fade_out = 4.0
        self._rel_total = 0                      # frames da descida em curso (0 = nenhuma)
        self._rel_done = 0
        self._rel_start_gain = 0.0

        self.limiter_on = False                  # limitador com antecipação na saída (o app liga; testes usam o simples)
        self._limiter = MasterLimiter()
        self._bus = np.zeros((MAX_CHUNK, 2), dtype=np.float32)
        self._ar = np.arange(1, MAX_CHUNK + 1, dtype=np.float32)

    # ================================================================ controle
    # (chamadas pela thread da UI)

    @property
    def bpm(self) -> float:
        return self._bpm

    @bpm.setter
    def bpm(self, v: float) -> None:
        self._bpm = float(min(400.0, max(20.0, v)))
        self._grid_version += 1

    @property
    def grid_offset(self) -> int:
        return self._grid_offset

    @grid_offset.setter
    def grid_offset(self, frames: int) -> None:
        self._grid_offset = int(frames)
        self._grid_version += 1

    @property
    def beats_per_bar(self) -> int:
        return self._bpb

    @beats_per_bar.setter
    def beats_per_bar(self, v: int) -> None:
        self._bpb = max(1, int(v))
        self._grid_version += 1

    @property
    def quantize(self) -> int:
        return self._quantize

    @quantize.setter
    def quantize(self, v: int) -> None:
        self._quantize = int(v)
        self._grid_version += 1

    @property
    def click_mode(self) -> int:
        return self._click_mode

    @click_mode.setter
    def click_mode(self, mode: int) -> None:
        self._click_mode = int(mode)
        self.ambient_silenced = False                 # escolher o modo do click o religa

    @property
    def pad_mode(self) -> int:
        return self._pad_mode

    @pad_mode.setter
    def pad_mode(self, mode: int) -> None:
        self._pad_mode = int(mode)
        self.ambient_silenced = False

    @property
    def clock(self) -> int:
        """Frames renderizados desde o início (relógio do motor; serve para medir tempo sem depender do relógio do PC)."""
        return self._clock

    @property
    def frames_per_beat(self) -> float:
        return 60.0 * self.sample_rate / self._bpm

    # ---- faixas ----
    @property
    def tracks(self) -> tuple[Track, ...]:
        return self._tracks

    @property
    def total_frames(self) -> int:
        return self._total

    def add_track(self, track: Track) -> None:
        with self._tracks_lock:
            self._tracks = self._tracks + (track,)
            self._recompute_total()

    def set_track_data(self, track: Track, data: np.ndarray) -> None:
        with self._tracks_lock:
            track._reset_gain = True
            track.data = data
            self._recompute_total()

    def remove_track(self, track: Track) -> None:
        with self._tracks_lock:
            self._tracks = tuple(t for t in self._tracks if t is not track)
            self._recompute_total()

    def clear_tracks(self) -> None:
        with self._tracks_lock:
            self._tracks = ()
            self._recompute_total()

    def set_tracks(self, tracks) -> None:
        """Troca o conjunto de faixas tocado (a música "no ar"). Só deve ser chamado com o motor parado."""
        tracks = tuple(tracks)
        with self._tracks_lock:
            for t in tracks:
                t._reset_gain = True               # cada faixa recomeça limpa (sem ganho nem crossfade herdado)
            self._tracks = tracks
            self._recompute_total()

    def _recompute_total(self) -> None:
        self._total = max((t.frames for t in self._tracks), default=0)

    # ---- transporte ----
    def play(self, delay_frames: int = 0) -> None:
        """Pede o início. Com `delay_frames`, o início cai no primeiro ponto da grade DEPOIS desse atraso."""
        self.ambient_silenced = False
        self._cmds.append(("play", max(0, int(delay_frames))))

    def cancel_start(self) -> None:
        """Cancela um início agendado que ainda não aconteceu (não mexe no que já está tocando)."""
        self._cmds.append(("cancel_start", 0))

    def pause(self) -> None:
        self._cmds.append(("pause", 0))

    def stop(self, silence: bool = False) -> None:
        """Para e volta ao início. Com `silence=True` (botão Parar), o click e o pad do ambiente também calam."""
        if silence:
            self.ambient_silenced = True
        self._jump_req = None
        self._loop = None
        self._cmds.append(("stop", 0))

    def seek(self, frames: int) -> None:
        self._cmds.append(("seek", int(frames)))

    # ---- pulos e repetição de trecho (seções) ----
    @property
    def pending_jump(self) -> int | None:
        return self._jump_req

    @property
    def loop_region(self) -> tuple[int, int] | None:
        return self._loop

    def jump(self, frame: int) -> None:
        """Com a música tocando, vai para `frame` no fim do compasso atual (corte limpo, sem estalo)."""
        self._jump_req = max(0, int(frame))

    def cancel_jump(self) -> None:
        self._jump_req = None

    def set_loop(self, start: int, end: int) -> None:
        """Repete o trecho [start, end): ao chegar no fim, volta ao início sem parar."""
        start, end = int(start), int(end)
        self._loop = (max(0, start), end) if end > start else None

    def clear_loop(self) -> None:
        self._loop = None

    # ================================================================== áudio
    def render_into(self, out: np.ndarray) -> None:
        """Preenche `out` (frames, 2) float32. Chamado pela thread de áudio."""
        n = out.shape[0]
        i = 0
        while i < n:
            m = min(MAX_CHUNK, n - i)
            self._split = None
            if self._jump_req is not None or self._loop is not None:
                m, self._split = self._plan_split(m)
            self._render_chunk(out[i:i + m], m)
            i += m

    def _plan_split(self, m: int) -> tuple[int, tuple[int, str] | None]:
        """Pulo/repetição: encurta o bloco para ele terminar exatamente no gatilho (fim do compasso ou do trecho).

        O último bloco antes do corte tem pelo menos FADE_FRAMES, para dar uma descida suave (sem estalo).
        """
        if not self._playing or self._pausing:
            return m, None
        pos = self._pos
        total = self._total
        best: tuple[int, int, str] | None = None
        jump = self._jump_req
        if jump is not None:
            bar = 60.0 * self.sample_rate / self._bpm * self._bpb
            r = (pos - self._grid_offset) % bar
            d = 0 if (r < 1e-6 or bar - r < 1e-6) else math.ceil(bar - r - 1e-6)
            if d < FADE_FRAMES:
                d += int(round(bar))
            best = (d, min(jump, max(0, total - 1)), "jump")
        loop = self._loop
        if loop is not None and pos < loop[1]:
            dl = loop[1] - pos
            if best is None or dl < best[0]:
                best = (dl, min(loop[0], max(0, total - 1)), "loop")
        if best is None:
            return m, None
        d, target, kind = best
        if d <= m:
            return d, (target, kind)
        if d - m < FADE_FRAMES:
            m = max(1, d - FADE_FRAMES)
        return m, None

    def render(self, n: int) -> np.ndarray:
        out = np.zeros((n, 2), dtype=np.float32)
        self.render_into(out)
        return out

    # ------------------------------------------------------------------ interno
    def _schedule(self, b0: float, a: int, b: int, bf: float, bpb: int, events: list) -> None:
        """Acha as batidas inteiras no trecho [a, b) cuja fase inicial é b0 (em batidas)."""
        length = b - a
        if length <= 0:
            return
        if (not self._beat_init) or abs(b0 - self._expected_b) > 1e-3:
            self._last_beat = math.ceil(b0 - 1e-9) - 1
            self._beat_init = True
        while True:
            nb = self._last_beat + 1
            fr = (nb - b0) * bf
            if fr >= length - 1e-9:
                break
            idx = min(b - 1, a + max(0, math.ceil(fr - 1e-6)))
            events.append((idx, nb % bpb))
            self._last_beat = nb
        self._expected_b = b0 + length / bf

    @staticmethod
    def _add_seg(bus: np.ndarray, seg: np.ndarray, n: int, gain, route: int) -> None:
        """Soma `seg` (int16: n mono ou n x 2) no bus; `gain` é escalar ou vetor (n,) float32, já com 1/32768."""
        if seg.ndim == 1:                              # faixa mono: o mesmo sinal nos dois lados
            v = seg * gain
            if route == ROUTE_STEREO:
                bus[:n, 0] += v
                bus[:n, 1] += v
            else:
                bus[:n, 0 if route == ROUTE_LEFT else 1] += v
        elif route == ROUTE_STEREO:
            bus[:n] += seg * (gain[:, None] if isinstance(gain, np.ndarray) else gain)
        else:
            mono = seg.sum(axis=1, dtype=np.float32) * (np.float32(0.5) * gain)
            bus[:n, 0 if route == ROUTE_LEFT else 1] += mono

    def _mix_tracks(self, bus: np.ndarray, frames: int, pos: int, tracks) -> None:
        any_solo = any(t.solo for t in tracks)
        ramp_base = self._ar[:frames] / np.float32(frames)
        for t in tracks:
            d = t.data
            if d is None:
                continue
            if t._reset_gain:
                t._last_gain = 0.0
                t._reset_gain = False
                t._using = None
                t._xf_old = None
                t._xf_left = 0
            # fonte: original ou cópia transposta (só faixas melódicas)
            sh = t.shifted
            cur = sh if (t.melodic and sh is not None and sh.shape == d.shape) else d
            if t._using is None:
                t._using = cur
            elif t._using is not cur:
                t._xf_old = t._using                  # troca de fonte com crossfade curto
                t._xf_left = XFADE_FRAMES
                t._using = cur
            audible = t.solo if any_solo else not t.mute
            target = float(t.volume) * t.level if audible else 0.0
            g0 = t._last_gain
            if g0 == 0.0 and target == 0.0:
                t._xf_old = None                      # silenciada: conclui a troca sem mixar
                t._xf_left = 0
                t.peak = 0.0
                continue
            n = min(frames, max(0, d.shape[0] - pos))
            if n > 0:
                seg_pk = cur[pos:pos + n]
                pk = max(int(seg_pk.max()), -int(seg_pk.min())) * max(g0, target) / 32768.0
                t.peak = pk if pk > t.peak * 0.97 else t.peak * 0.97        # sobe na hora, cai suave
                if t.role == "music":                                        # volume das faixas de música (guia o pad)
                    xf = cur[pos:pos + n:4].astype(np.float32).ravel()      # 1 em cada 4 quadros basta para o nível
                    gm = max(g0, target)
                    self._music_power += float(np.dot(xf, xf)) / max(1, xf.size) * gm * gm * float(_INV16) ** 2
                if g0 == target:
                    gain = np.float32(target) * _INV16
                else:
                    gain = (np.float32(g0) + np.float32(target - g0) * ramp_base[:n]) * _INV16
                old = t._xf_old
                if t._xf_left > 0 and old is not None:
                    m = min(n, t._xf_left)
                    w = (np.arange(1, m + 1, dtype=np.float32) + np.float32(XFADE_FRAMES - t._xf_left)) \
                        / np.float32(XFADE_FRAMES)
                    w_new = np.ones(n, dtype=np.float32)
                    w_new[:m] = w
                    w_old = np.zeros(n, dtype=np.float32)
                    w_old[:m] = 1.0 - w
                    self._add_seg(bus, cur[pos:pos + n], n, gain * w_new, t.route)
                    self._add_seg(bus, old[pos:pos + n], n, gain * w_old, t.route)
                    t._xf_left -= m
                    if t._xf_left <= 0:
                        t._xf_old = None
                else:
                    self._add_seg(bus, cur[pos:pos + n], n, gain, t.route)
            t._last_gain = target

    def _render_chunk(self, out: np.ndarray, n: int) -> None:
        out[:] = 0.0
        tracks = self._tracks
        total = self._total
        sr = self.sample_rate
        bf = 60.0 * sr / self._bpm
        off = self._grid_offset
        bpb = self._bpb

        # ---- comandos ----
        seek = None
        cmd = None
        cmd_arg = 0
        cancel = False
        cmds = self._cmds
        while True:
            try:
                c = cmds.popleft()
            except IndexError:
                break
            if c[0] == "seek":
                seek = c[1]
            elif c[0] == "cancel_start":
                cancel = True
                cmd = None
            else:
                cmd = c[0]
                cmd_arg = c[1]

        recompute = False
        if cancel:
            self._start_pending = False
        if seek is not None:
            self._pos = max(0, min(seek, total))
            if self._playing:
                self._fade_in_left = FADE_FRAMES
            elif self._start_pending:
                recompute = True
        if cmd == "play":
            if not self._playing and not self._start_pending and total > 0:
                if self._pos >= total:
                    self._pos = 0
                self._start_pending = True
                self._start_not_before = self._clock + cmd_arg
                recompute = True
        elif cmd in ("pause", "stop"):
            self._start_pending = False
            if self._playing:
                self._pausing = True
                self._stop_to_start = cmd == "stop"
            elif cmd == "stop":
                self._pos = 0
        if total <= 0:
            self._start_pending = False
            if self._playing:
                self._pausing = True
        gv = self._grid_version
        if self._start_pending and gv != self._last_grid_version:
            recompute = True
        self._last_grid_version = gv

        # ---- agenda o início do projeto, alinhado à grade do click ----
        if self._start_pending and recompute:
            self._start_pos = self._pos
            quant = self._quantize
            q = float(bpb) if quant == QUANT_BAR else (1.0 if quant == QUANT_BEAT else 0.0)
            wait = max(0, self._start_not_before - self._clock)      # atraso mínimo ainda restante
            s = wait
            if q > 0.0:
                target = (self._start_pos - off) / bf
                d = math.fmod(target - (self._ambient_beat + wait / bf), q)
                if d < 0.0:
                    d += q
                if d > q - 1e-6:
                    d = 0.0
                s = wait + max(0, math.ceil(d * bf - 1e-6))
            elif wait == 0:
                s = 0
            self._start_at = self._clock + s

        was_playing = self._playing
        pos_at_start = self._pos
        proj_start = n
        starts_now = False
        if self._start_pending and self._start_at < self._clock + n:
            proj_start = max(0, self._start_at - self._clock)
            starts_now = True
        elif self._playing:
            proj_start = 0
        if starts_now:
            self._playing = True
            self._start_pending = False
            self._pos = self._start_pos
            self._fade_in_left = FADE_FRAMES

        # ---- batidas do click ----
        events: list[tuple[int, int]] = []
        if was_playing:
            self._schedule((pos_at_start - off) / bf, 0, n, bf, bpb, events)
        elif starts_now:
            self._schedule(self._ambient_beat, 0, proj_start, bf, bpb, events)
            self._schedule((self._start_pos - off) / bf, proj_start, n, bf, bpb, events)
        else:
            self._schedule(self._ambient_beat, 0, n, bf, bpb, events)
        self._ambient_beat = self._expected_b

        # ---- projeto ----
        gate_active = was_playing or starts_now
        proj_frames = n - proj_start
        self._music_power = 0.0
        if self._playing and proj_frames > 0:
            bus = self._bus[:proj_frames]
            bus[:] = 0.0
            self._mix_tracks(bus, proj_frames, self._pos, tracks)
            if self._fade_in_left > 0:
                m = min(self._fade_in_left, proj_frames)
                g = 1.0 - (self._fade_in_left - np.arange(m)) / FADE_FRAMES
                bus[:m] *= g.astype(np.float32)[:, None]
                self._fade_in_left -= m
            if self._pausing:
                fo = 1.0 - np.arange(1, proj_frames + 1) / proj_frames
                bus *= fo.astype(np.float32)[:, None]
            split = self._split
            if split is not None and not self._pausing:
                k = min(proj_frames, FADE_FRAMES)
                bus[proj_frames - k:] *= (np.arange(k, 0, -1, dtype=np.float32) / np.float32(k))[:, None]
            out[proj_start:] += bus
            self._pos += proj_frames
            if split is not None and not self._pausing:
                self._pos = split[0]                              # corte limpo: desce, pula, sobe
                self._fade_in_left = FADE_FRAMES
                if split[1] == "jump":
                    self._jump_req = None
            elif self._pausing:
                self._playing = False
                self._pausing = False
                if self._stop_to_start:
                    self._pos = 0
                self._stop_to_start = False
            elif self._pos >= total:
                self._playing = False
                self._pos = 0
                self.ended_count += 1                  # fim natural: o controle decide o que vem depois

        # ---- cama de ministração: toca com o projeto parado; some sozinha quando o projeto toca.
        # Enquanto ela soa, o click e o pad do ambiente do projeto ficam em silêncio (sem click duplo).
        bed = self.ministry
        want_bed = self.ministry_on and not gate_active
        bed_sounds = want_bed or bed.active

        # ---- click ----
        cm = self.click_mode
        click_target = 0.0
        if (cm == GATE_ALWAYS or (cm == GATE_PAUSED and not gate_active)) and not bed_sounds \
                and not self.ambient_silenced:
            click_target = float(self.click_volume)
        if click_target > 0.0 or self._click_gain > 1e-4 or self._click_voices:
            if click_target > 0.0 or self._click_gain > 1e-4:
                for idx, beat in events:
                    wave = self._click_accent if beat == 0 else self._click_normal
                    self._click_voices.append([wave, 0, idx])
            buf = np.zeros(n, dtype=np.float32)
            alive = []
            for v in self._click_voices:
                wave, i0, delay = v
                m = min(len(wave) - i0, n - delay)
                if m > 0:
                    buf[delay:delay + m] += wave[i0:i0 + m]
                    v[1] = i0 + m
                v[2] = 0
                if v[1] < len(wave):
                    alive.append(v)
            self._click_voices = alive
            g0 = self._click_gain
            ramp = np.float32(g0) + np.float32(click_target - g0) * (self._ar[:n] / np.float32(n))
            buf *= ramp
            self._click_gain = click_target
            _route_add(out, buf, buf, self.click_route)

        # ---- volume da música (guia o pad sob a música) ----
        p_now = self._music_power * (proj_frames / n) if n else 0.0
        env = self._music_env
        a = 1.0 - math.exp(-n / (0.10 * self.sample_rate)) if p_now > env else 1.0 - math.exp(-n / (0.60 * self.sample_rate))
        env += (p_now - env) * a
        self._music_env = env
        db = 10.0 * math.log10(env + 1e-12)
        t_loud = min(1.0, max(0.0, (db - DUCK_QUIET_DB) / (DUCK_LOUD_DB - DUCK_QUIET_DB)))
        t_loud = t_loud * t_loud * (3.0 - 2.0 * t_loud)                      # curva suave
        self.duck = DUCK_FLOOR + (1.0 - DUCK_FLOOR) * (1.0 - t_loud)

        # ---- pad ----
        pm = self.pad_mode
        pad_target = 0.0
        under = min(1.0, max(0.0, float(self.pad_under)))
        if pm != GATE_OFF and not bed_sounds and not self.ambient_silenced:
            if not gate_active:
                pad_target = float(self.pad_volume)                    # projeto parado: volume normal
            elif under > 0.0:
                pad_target = float(self.pad_volume) * under * self.duck    # projeto tocando: só preenche o silêncio
            elif pm == GATE_ALWAYS:
                pad_target = float(self.pad_volume)
        # Projeto tocando e o pad ainda soando (sem Parar, sem cama de ministração): descida longa e suave,
        # começando no instante em que as faixas entram (não antes).
        releasing = (pad_target == 0.0 and gate_active and not bed_sounds and not self.ambient_silenced
                     and self.pad_fade_out > 0.05 and (self._pad_gain > 1e-4 or self._rel_total > 0))
        if pad_target > 0.0 or self._pad_gain > 1e-4:
            self.pad.set_chord(self.key_pc, self.key_minor, self.pad_triad,
                               instant=self._pad_gain < 1e-3)
            pl, pr = self.pad.render(n)
            if releasing:
                if self._rel_total == 0:
                    self._rel_total = max(1, int(self.pad_fade_out * self.sample_rate))
                    self._rel_done = 0
                    self._rel_start_gain = self._pad_gain
                t0 = proj_start if starts_now else 0           # antes da entrada, o pad segue inteiro
                k = np.arange(1, n + 1, dtype=np.float64) - t0
                prog = np.clip((self._rel_done + np.maximum(k, 0.0)) / self._rel_total, 0.0, 1.0)
                g = self._rel_start_gain * 0.5 * (1.0 + np.cos(np.pi * prog))
                self._rel_done += max(0, n - t0)
                if self._rel_done >= self._rel_total:
                    g[:] = np.where(prog >= 1.0, 0.0, g)
            else:
                self._rel_total = 0
                decay = self._pad_decay[:n]
                if gate_active and under > 0.0:
                    # sob a música: sobe devagar no silêncio (delicado) e desce no "Some em 2 / 4 / 8 s"
                    if pad_target > self._pad_gain:
                        tau = 1.5
                    else:
                        tau = max(0.5, float(self.pad_fade_out) / 4.0)
                    decay = np.exp(-np.arange(1, n + 1) / (tau * self.sample_rate))
                g = pad_target + (self._pad_gain - pad_target) * decay
            self._pad_gain = float(g[-1])
            gf = g.astype(np.float32)
            _route_add(out, pl * gf, pr * gf, self.pad_route)
        else:
            self._pad_gain = 0.0
            self._rel_total = 0

        # ---- cama de ministração (pad em camadas + click, no tom e andamento próprios) ----
        if bed_sounds:
            bed.mono = self.pad_route != ROUTE_STEREO              # pad em L ou R = um canal só
            ml, mr, mc = bed.render(n, want_bed, float(self.click_volume))
            _route_add(out, ml, mr, self.pad_route)
            _route_add(out, mc, mc, self.click_route)

        # ---- master (com limitador suave) ----
        if self.master != 1.0:
            out *= np.float32(self.master)
        a = np.abs(out)
        pl, pr = float(a[:, 0].max()), float(a[:, 1].max())
        self.peak_l = pl if pl > self.peak_l * 0.97 else self.peak_l * 0.97
        self.peak_r = pr if pr > self.peak_r * 0.97 else self.peak_r * 0.97
        if self.limiter_on:
            out[:] = self._limiter.process(out)
            np.clip(out, -1.0, 1.0, out=out)                # só por segurança: o limitador já segura em 0,97
        elif a.max() > 0.8:
            hot = a > 0.8
            out[hot] = np.sign(out[hot]) * (0.8 + 0.2 * np.tanh((a[hot] - 0.8) / 0.2))

        # ---- publicação para a UI ----
        if events:
            last = events[-1][1]
            self.beat_in_bar = last
            self.beat_counter += len(events)
        self.position = self._pos
        self.playing = self._playing
        self.armed = self._start_pending
        self._clock += n
