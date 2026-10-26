"""Pad e click de mentira, determinísticos (só para os testes de equivalência)."""
import numpy as np


def make_click(sr, freq, amp):
    n = int(0.05 * sr)
    t = np.arange(n) / sr
    return (np.sin(2 * np.pi * freq * t) * np.exp(-t * 60.0) * amp).astype(np.float32)


class PadSynth:
    def __init__(self, sr):
        self.sr = sr
        self._t = 0
        self._f = 130.0
        self._tf = 130.0
        self.calls = 0

    def set_chord(self, key_pc, minor, triad, instant=False):
        self._tf = 130.0 * (2 ** (key_pc / 12)) * (1.2 if minor else 1.0) * (1.1 if triad else 1.0)
        if instant:
            self._f = self._tf

    def render(self, n):
        self.calls += 1
        self._f += (self._tf - self._f) * 0.1
        t = (self._t + np.arange(n)) / self.sr
        self._t += n
        pl = (0.25 * np.sin(2 * np.pi * self._f * t)).astype(np.float32)
        pr = (0.25 * np.sin(2 * np.pi * self._f * 1.01 * t + 0.3)).astype(np.float32)
        return pl, pr
