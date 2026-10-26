"""Cenários de teste: os MESMOS comandos rodam no motor Python e no nativo; a saída tem de ser igual."""
import numpy as np

SR = 48000
BLOCKS = [512, 512, 333, 1024, 256, 4096, 700, 512, 5000, 128, 512, 2048]


def make_audio(seed, secs, stereo=True, amp=0.5, tone=220.0):
    rng = np.random.default_rng(seed)
    n = int(secs * SR)
    t = np.arange(n) / SR
    base = amp * np.sin(2 * np.pi * tone * t) + 0.1 * amp * rng.standard_normal(n)
    if stereo:
        r = amp * np.sin(2 * np.pi * tone * 1.5 * t + 0.4) + 0.1 * amp * rng.standard_normal(n)
        d = np.stack([base, r], 1)
    else:
        d = base
    return np.clip(np.rint(d * 32767), -32768, 32767).astype(np.int16)


class Ctx:
    def __init__(self, mod):
        self.mod = mod
        self.e = mod.Engine(SR)
        self.e.limiter_on = False
        self.out = []
        self.st = []
        self.bi = 0
        self.tracks = []

    def track(self, name, data, role="music", **kw):
        t = self.mod.Track(name, data)
        t.role = role
        for k, v in kw.items():
            setattr(t, k, v)
        self.tracks.append(t)
        return t

    def r(self, frames):
        """renderiza `frames` quadros em blocos de tamanhos variados"""
        left = frames
        while left > 0:
            n = min(left, BLOCKS[self.bi % len(BLOCKS)])
            self.bi += 1
            o = np.zeros((n, 2), dtype=np.float32)
            self.e.render_into(o)
            self.out.append(o)
            e = self.e
            self.st.append([e.position, int(e.playing), int(e.armed), e.beat_in_bar, e.beat_counter, e.ended_count,
                            e.clock, e.total_frames, -1 if e.pending_jump is None else e.pending_jump,
                            *(e.loop_region or (-1, -1)),
                            *[round(float(t.peak), 2) for t in e.tracks]])
            left -= n

    def sec(self, s):
        self.r(int(s * SR))


def s_click_idle(c):
    e = c.e
    e.bpm = 133.0
    e.beats_per_bar = 3
    e.click_mode = 2
    e.click_volume = 0.8
    c.sec(1.5)
    e.click_route = 1
    e.bpm = 91.5
    c.sec(1.2)
    e.click_mode = 1
    c.sec(0.5)
    e.stop(silence=True)
    c.sec(0.5)
    e.click_mode = 2
    c.sec(0.7)


def _four(c):
    a = c.track("a", make_audio(1, 9.0, True, 0.4, 220.0))
    b = c.track("b", make_audio(2, 9.0, False, 0.5, 330.0))
    k = c.track("click", make_audio(3, 9.0, False, 0.3, 1000.0), role="click")
    g = c.track("guide", make_audio(4, 9.2, True, 0.3, 440.0), role="guide")
    c.e.set_tracks([a, b, k, g])
    return a, b, k, g


def s_play_basic(c):
    e = c.e
    a, b, k, g = _four(c)
    e.bpm = 120.0
    e.grid_offset = 3000
    e.click_mode = 1
    e.pad_mode = 1
    e.pad_volume = 0.6
    c.sec(0.7)
    e.play()
    c.sec(1.5)
    a.volume = 0.5
    b.mute = True
    c.sec(0.8)
    b.mute = False
    k.route = 1
    g.route = 2
    c.sec(0.6)
    a.solo = True
    c.sec(0.5)
    a.solo = False
    a.level = 1.7
    c.sec(0.6)
    e.pause()
    c.sec(0.6)
    e.play()
    c.sec(1.2)
    e.seek(int(4.1 * SR))
    c.sec(0.8)
    e.stop()
    c.sec(0.5)
    e.play(int(1.3 * SR))
    c.sec(2.0)
    e.cancel_start()
    e.stop()
    c.sec(0.3)


def s_jump_loop(c):
    e = c.e
    a, b, k, g = _four(c)
    e.bpm = 100.0
    e.grid_offset = 1234
    e.click_mode = 2
    e.play()
    c.sec(1.5)
    e.jump(int(6.0 * SR))
    c.sec(2.5)
    e.set_loop(int(2.0 * SR), int(3.5 * SR))
    c.sec(5.0)
    e.jump(int(1.0 * SR))
    c.sec(2.0)
    e.clear_loop()
    c.sec(1.0)
    e.set_loop(int(0.5 * SR), int(0.6 * SR))
    c.sec(1.0)
    e.cancel_jump()
    e.clear_loop()
    e.set_loop(int(8.0 * SR), int(8.9 * SR))
    c.sec(4.0)
    e.stop()
    c.sec(0.5)


def s_shifted(c):
    e = c.e
    a, b, k, g = _four(c)
    sa = make_audio(11, 9.0, True, 0.4, 247.0)
    sb = make_audio(12, 9.0, False, 0.5, 370.0)
    e.play()
    c.sec(1.0)
    a.shifted = sa
    a.shifted_semitones = 2
    c.sec(0.3)
    b.shifted = sb
    c.sec(0.5)
    a.melodic = False
    c.sec(0.5)
    a.melodic = True
    c.sec(0.5)
    a.shifted = None
    c.sec(0.5)
    b.shifted = make_audio(13, 9.0, False, 0.5, 400.0)
    c.sec(1.0)
    b.shifted = np.zeros(10, dtype=np.int16)           # formato diferente: não deve ser usado
    c.sec(0.5)
    e.stop()


def s_pad_modes(c):
    e = c.e
    a, b, k, g = _four(c)
    e.bpm = 118.0
    e.key_pc = 4
    e.key_minor = True
    e.pad_mode = 2
    e.pad_volume = 0.7
    e.pad_route = 2
    e.click_mode = 1
    e.click_route = 1
    c.sec(1.0)
    e.pad_under = 0.6
    e.play()
    c.sec(3.0)
    e.pause()
    c.sec(1.0)
    e.pad_under = 0.0
    e.pad_fade_out = 2.0
    e.play()
    c.sec(4.0)
    e.pause()
    c.sec(0.5)
    e.pad_mode = 1
    e.key_pc = 9
    e.pad_triad = True
    c.sec(1.0)
    e.ministry_on = True
    c.sec(2.5)
    e.play()
    c.sec(3.0)
    e.pause()
    c.sec(1.5)
    e.ministry_on = False
    c.sec(2.0)
    e.stop(silence=True)
    c.sec(1.0)
    e.pad_mode = 2
    c.sec(1.0)


def s_limiter(c):
    e = c.e
    a = c.track("a", make_audio(21, 9.0, True, 0.95, 220.0))
    b = c.track("b", make_audio(22, 9.0, True, 0.95, 223.0))
    d = c.track("c", make_audio(23, 9.0, False, 0.95, 226.0))
    e.set_tracks([a, b, d])
    e.limiter_on = True
    e.master = 1.0
    e.play()
    c.sec(2.5)
    e.master = 1.4
    c.sec(1.5)
    e.limiter_on = False
    c.sec(1.5)
    e.master = 0.7
    c.sec(1.0)
    e.limiter_on = True
    c.sec(1.0)
    e.stop()
    c.sec(0.3)


def s_end_replace(c):
    e = c.e
    a = c.track("a", make_audio(31, 2.0, True, 0.4, 220.0))
    b = c.track("b", make_audio(32, 2.6, False, 0.5, 330.0))
    e.set_tracks([a, b])
    e.click_mode = 1
    e.play()
    c.sec(3.2)
    e.play()
    c.sec(1.0)
    n = c.track("n", make_audio(33, 4.0, True, 0.4, 500.0))
    e.add_track(n)
    c.sec(1.0)
    e.set_track_data(a, make_audio(34, 3.0, True, 0.4, 261.0))
    c.sec(1.0)
    e.remove_track(b)
    c.sec(1.0)
    e.pause()
    e.clear_tracks()
    c.sec(0.3)
    e.play()
    c.sec(0.3)
    e.set_tracks([n])
    e.play()
    c.sec(1.5)
    e.stop()


def s_quantize(c):
    e = c.e
    a, b, k, g = _four(c)
    e.bpm = 140.0
    e.grid_offset = 5000
    e.click_mode = 2
    e.quantize = 1
    c.sec(0.45)
    e.play()
    c.sec(1.0)
    e.stop()
    c.sec(0.3)
    e.quantize = 0
    e.play()
    c.sec(0.8)
    e.stop()
    e.quantize = 2
    e.seek(int(2.3 * SR))
    e.play()
    c.sec(0.2)
    e.bpm = 90.0
    c.sec(0.2)
    e.grid_offset = 9000
    c.sec(2.0)
    e.cancel_start()
    c.sec(0.5)
    e.stop()


SCENARIOS = {f.__name__[2:]: f for f in (s_click_idle, s_play_basic, s_jump_loop, s_shifted, s_pad_modes,
                                           s_limiter, s_end_replace, s_quantize)}


# ------------------------------------------------------------------ cenário aleatório (fuzz)
def make_fuzz(seed, steps=120):
    def fuzz(c):
        rng = np.random.default_rng(seed)
        e = c.e
        names = ["a", "b", "k", "g", "m"]
        tr = []
        for i, nm in enumerate(names):
            secs = float(rng.uniform(3.0, 9.0))
            role = "click" if nm == "k" else ("guide" if nm == "g" else "music")
            tr.append(c.track(nm, make_audio(100 + i + seed * 10, secs, bool(rng.integers(0, 2)),
                                             float(rng.uniform(0.2, 0.95)), float(rng.uniform(100, 900))), role=role))
        e.set_tracks(tr[:4])
        e.bpm = float(rng.uniform(60, 160))
        e.grid_offset = int(rng.integers(0, 20000))
        e.beats_per_bar = int(rng.integers(2, 7))
        e.key_pc = int(rng.integers(0, 12))
        e.key_minor = bool(rng.integers(0, 2))
        e.limiter_on = bool(rng.integers(0, 2))
        total = max(t.frames for t in tr[:4])
        for _ in range(steps):
            op = int(rng.integers(0, 30))
            t = tr[int(rng.integers(0, len(tr)))]
            if op == 0: e.play(int(rng.integers(0, 2)) * int(rng.integers(0, 96000)))
            elif op == 1: e.pause()
            elif op == 2: e.stop(bool(rng.integers(0, 2)))
            elif op == 3: e.seek(int(rng.integers(0, total)))
            elif op == 4: e.jump(int(rng.integers(0, total)))
            elif op == 5: e.cancel_jump()
            elif op == 6:
                a = int(rng.integers(0, total - 20000)); e.set_loop(a, a + int(rng.integers(500, 90000)))
            elif op == 7: e.clear_loop()
            elif op == 8: t.volume = float(rng.uniform(0, 1.5))
            elif op == 9: t.mute = bool(rng.integers(0, 2))
            elif op == 10: t.solo = bool(rng.integers(0, 4) == 0)
            elif op == 11: t.route = int(rng.integers(0, 3))
            elif op == 12: t.level = float(rng.uniform(0.3, 2.0))
            elif op == 13: t.melodic = bool(rng.integers(0, 2))
            elif op == 14: t.shifted = make_audio(int(rng.integers(0, 99999)), t.frames / SR, t.data.ndim == 2, 0.5, 300.0) if t.data is not None and rng.integers(0, 3) else None
            elif op == 15: e.click_mode = int(rng.integers(0, 3))
            elif op == 16: e.pad_mode = int(rng.integers(0, 3))
            elif op == 17: e.pad_under = float(rng.choice([0.0, 0.0, 0.5, 1.0]))
            elif op == 18: e.pad_volume = float(rng.uniform(0.1, 1.0))
            elif op == 19: e.ministry_on = bool(rng.integers(0, 2))
            elif op == 20: e.quantize = int(rng.integers(0, 3))
            elif op == 21: e.bpm = float(rng.uniform(60, 160))
            elif op == 22: e.master = float(rng.uniform(0.5, 1.4))
            elif op == 23: e.pad_route = int(rng.integers(0, 3)); e.click_route = int(rng.integers(0, 3))
            elif op == 24: e.cancel_start()
            elif op == 25: e.pad_fade_out = float(rng.choice([0.0, 2.0, 4.0]))
            elif op == 26: e.key_pc = int(rng.integers(0, 12)); e.pad_triad = bool(rng.integers(0, 2))
            elif op == 27: e.limiter_on = bool(rng.integers(0, 2))
            elif op == 28: e.grid_offset = int(rng.integers(0, 20000))
            else: pass
            c.r(int(rng.integers(1, 40000)))
    return fuzz


for _s in range(1, 13):
    SCENARIOS[f"fuzz{_s}"] = make_fuzz(_s)
