// Limitador de pico com antecipação (igual ao MasterLimiter do motor Python).
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace lt {

class MasterLimiter {
public:
    explicit MasterLimiter(int lookahead = 96, float ceiling = 0.97f)
        : w_(lookahead | 1), ceiling_(ceiling), prev_(static_cast<size_t>(4) * w_, 0.0f) {
        const size_t maxn = 4096 + 2 * static_cast<size_t>(w_) + 16;     // sem alocar na thread de áudio
        ext_.reserve(maxn * 2);
        peak_.reserve(maxn); need_.reserve(maxn); gmin_.reserve(maxn); gavg_.reserve(maxn);
        dq_.reserve(maxn + 2 * static_cast<size_t>(w_) + 16);
    }

    float reduction() const { return reduction_; }

    // Processa n quadros intercalados (L,R) no próprio buffer. A saída sai atrasada em w quadros.
    void process(float* x, int n) {
        const int w = w_;
        const int N = 2 * w + n;
        ext_.resize(static_cast<size_t>(N) * 2);
        std::copy(prev_.begin(), prev_.end(), ext_.begin());
        std::copy(x, x + static_cast<size_t>(n) * 2, ext_.begin() + static_cast<size_t>(2 * w) * 2);
        std::copy(ext_.begin() + static_cast<size_t>(n) * 2, ext_.begin() + static_cast<size_t>(N) * 2, prev_.begin());

        peak_.resize(N);
        float pmax = 0.0f;
        for (int i = 0; i < N; ++i) {
            float p = std::max(std::fabs(ext_[2 * i]), std::fabs(ext_[2 * i + 1]));
            peak_[i] = p;
            if (p > pmax) pmax = p;
        }
        if (pmax <= ceiling_) {
            reduction_ = 1.0f;
            std::copy(ext_.begin() + static_cast<size_t>(w) * 2, ext_.begin() + static_cast<size_t>(w + n) * 2, x);
            return;
        }
        need_.assign(N, 1.0f);
        for (int i = 0; i < N; ++i)
            if (peak_[i] > ceiling_) need_[i] = ceiling_ / peak_[i];

        // minimum_filter1d(size=w, constant 1.0): janela centrada
        const int h = w / 2;
        gmin_.resize(N);
        dq_.resize(static_cast<size_t>(N) + 2 * static_cast<size_t>(h) + 4);
        size_t head = 0, tail = 0;                      // fila monotônica de índices (cada índice entra uma vez)
        auto val = [&](int i) { return (i < 0 || i >= N) ? 1.0f : need_[i]; };
        int next = -h;                                   // próximo índice a entrar; janela de i: [i-h, i+h]
        for (int i = 0; i < N; ++i) {
            while (next <= i + h) {
                const float v = val(next);
                while (tail > head && val(dq_[tail - 1]) >= v) --tail;
                dq_[tail++] = next;
                ++next;
            }
            while (dq_[head] < i - h) ++head;
            gmin_[i] = val(dq_[head]);
        }
        // uniform_filter1d(size=w, constant 1.0): média da janela centrada
        gavg_.resize(N);
        double sum = 0.0;
        auto gv = [&](int i) { return (i < 0 || i >= N) ? 1.0 : static_cast<double>(gmin_[i]); };
        for (int k = -h; k <= h; ++k) sum += gv(k);
        for (int i = 0; i < N; ++i) {
            gavg_[i] = static_cast<float>(sum / w);
            sum += gv(i + h + 1) - gv(i - h);
        }
        float red = 1.0f;
        for (int j = 0; j < n; ++j) {
            const int i = w + j;
            const float g = std::min(gavg_[i], need_[i]);
            if (g < red) red = g;
            x[2 * j] = ext_[2 * i] * g;
            x[2 * j + 1] = ext_[2 * i + 1] * g;
        }
        reduction_ = red;
    }

private:
    int w_;
    float ceiling_;
    float reduction_ = 1.0f;
    std::vector<float> prev_, ext_, peak_, need_, gmin_, gavg_;
    std::vector<int> dq_;
};

}  // namespace lt
