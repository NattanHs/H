// Fila circular sem travas (um produtor, um consumidor) para blocos de áudio float.
#pragma once
#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

namespace lt {

class Ring {
public:
    Ring(int channels, int capacity_pow2)
        : ch_(channels), cap_(capacity_pow2), mask_(capacity_pow2 - 1),
          buf_(static_cast<size_t>(channels) * static_cast<size_t>(capacity_pow2), 0.0f) {}

    int channels() const { return ch_; }
    int capacity() const { return cap_; }

    // quadros que o consumidor ainda não leu
    int available() const {
        return static_cast<int>(w_.load(std::memory_order_acquire) - r_.load(std::memory_order_acquire));
    }
    // quadros que o produtor ainda pode escrever
    int space() const { return cap_ - available(); }

    // produtor: copia n quadros de cada canal (n <= space())
    void write(const float* const* src, int n) {
        const uint64_t w = w_.load(std::memory_order_relaxed);
        for (int c = 0; c < ch_; ++c) {
            float* dst = &buf_[static_cast<size_t>(c) * cap_];
            for (int i = 0; i < n; ++i) dst[(w + i) & mask_] = src[c][i];
        }
        w_.store(w + n, std::memory_order_release);
    }

    // consumidor: lê até n quadros em cada canal; devolve quantos leu
    int read(float* const* dst, int n) {
        const uint64_t r = r_.load(std::memory_order_relaxed);
        int got = static_cast<int>(w_.load(std::memory_order_acquire) - r);
        if (got > n) got = n;
        for (int c = 0; c < ch_; ++c) {
            const float* src = &buf_[static_cast<size_t>(c) * cap_];
            for (int i = 0; i < got; ++i) dst[c][i] = src[(r + i) & mask_];
        }
        r_.store(r + got, std::memory_order_release);
        return got;
    }

    // consumidor: descarta tudo o que está na fila
    void consumer_flush() { r_.store(w_.load(std::memory_order_acquire), std::memory_order_release); }

private:
    int ch_, cap_;
    uint64_t mask_;
    std::vector<float> buf_;
    std::atomic<uint64_t> w_{0}, r_{0};
};

}  // namespace lt
