"""Cama de ministração de mentira, determinística (só para os testes de equivalência)."""
import numpy as np


class MinistryBed:
    def __init__(self, sr):
        self.sr = sr
        self.bpm = 70.0
        self.key_pc = 0
        self.minor = False
        self.bpb = 4
        self.elements = {}
        self.volume = 0.5
        self.active = False
        self.mono = False
        self._env = 0.0
        self._t = 0

    def prepare(self):
        pass

    def render(self, n, want, click_volume):
        t = (self._t + np.arange(n)) / self.sr
        tgt = 1.0 if want else 0.0
        e0 = self._env
        k = (np.arange(1, n + 1) / n)
        env = e0 + (tgt - e0) * k * (0.02 if want else 0.05) * 10
        env = np.clip(env, 0.0, 1.0)
        self._env = float(env[-1])
        self.active = self._env > 1e-3 or want
        f = 110.0 * (2 ** (self.key_pc / 12))
        ml = (0.2 * np.sin(2 * np.pi * f * t) * env).astype(np.float32)
        mr = (0.2 * np.sin(2 * np.pi * f * 1.005 * t) * env).astype(np.float32)
        if self.mono:
            mr = ml.copy()
        beat = (np.arange(n) + self._t) % int(self.sr * 60 / self.bpm) < 400
        mc = (0.3 * click_volume * beat * env).astype(np.float32)
        self._t += n
        return ml, mr, mc
