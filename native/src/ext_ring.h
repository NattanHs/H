// Provedor do pad e da cama de ministração por filas: uma thread comum do Python sintetiza (fora da thread de áudio)
// e escreve nas filas; a thread de áudio só lê. Se o Python atrasar, só o pad/cama falha, nunca as faixas.
#pragma once
#include <atomic>
#include <cstring>

#include "core.h"
#include "ring.h"

namespace lt {

class RingProvider : public ExtProvider {
public:
    RingProvider() : pad_(2, 16384), bed_(4, 16384), flag_(MAX_CHUNK, 0.0f) {}

    // ------------------------------------------------------------ thread de áudio (consumidor)
    void pad_idle() override { pad_active_.store(false, std::memory_order_release); }

    void pad_render(bool instant, int n, float* l, float* r) override {
        if (!pad_active_.load(std::memory_order_acquire)) {
            pad_.consumer_flush();                       // volta de um período parado: descarta o que ficou velho
            pad_warm_ = false;
            pad_active_.store(true, std::memory_order_release);
        }
        instant_.store(instant, std::memory_order_relaxed);
        float* ch[2] = {l, r};
        const int got = pad_.read(ch, n);
        if (got < n) {
            std::memset(l + got, 0, sizeof(float) * (n - got));
            std::memset(r + got, 0, sizeof(float) * (n - got));
            if (pad_warm_) pad_under_.fetch_add(1, std::memory_order_relaxed);   // falha de verdade
            // (antes da 1ª leitura é só a partida a frio: o produtor ainda está preenchendo a fila)
        }
        if (got > 0) pad_warm_ = true;
    }

    bool bed_active() override { return bed_flag_; }

    void bed_idle() override {
        bed_state_.store(false, std::memory_order_release);
        bed_flag_ = false;
    }

    void bed_render(int n, bool want, float click_volume, bool mono, float* ml, float* mr, float* mc) override {
        if (!bed_state_.load(std::memory_order_acquire)) {
            bed_.consumer_flush();
            bed_flag_ = false;
            bed_warm_ = false;
            bed_state_.store(true, std::memory_order_release);
        }
        want_.store(want, std::memory_order_relaxed);
        vol_.store(click_volume, std::memory_order_relaxed);
        mono_.store(mono, std::memory_order_relaxed);
        if (flag_.size() < static_cast<size_t>(n)) flag_.resize(n);       // (só cresce na 1ª vez com bloco maior)
        float* ch[4] = {ml, mr, mc, flag_.data()};
        const int got = bed_.read(ch, n);
        if (got < n) {
            std::memset(ml + got, 0, sizeof(float) * (n - got));
            std::memset(mr + got, 0, sizeof(float) * (n - got));
            std::memset(mc + got, 0, sizeof(float) * (n - got));
            if (bed_warm_) bed_under_.fetch_add(1, std::memory_order_relaxed);
        }
        if (got > 0) {
            bed_warm_ = true;
            bed_flag_ = flag_[got - 1] > 0.5f;
            starve_ = 0;
        } else {
            starve_ += n;
            if (starve_ > 48000) bed_flag_ = false;                        // produtor parou: não deixa o click calado para sempre
        }
    }

    // ------------------------------------------------------------ thread do produtor (Python)
    bool pad_needed() const { return pad_active_.load(std::memory_order_acquire); }
    int pad_space() const { return pad_.space(); }
    int pad_fill() const { return pad_.available(); }
    bool pad_instant() const { return instant_.load(std::memory_order_relaxed); }
    void pad_write(const float* l, const float* r, int n) {
        const float* ch[2] = {l, r};
        pad_.write(ch, n);
    }
    bool bed_needed() const { return bed_state_.load(std::memory_order_acquire); }
    int bed_space() const { return bed_.space(); }
    int bed_fill() const { return bed_.available(); }
    bool bed_want() const { return want_.load(std::memory_order_relaxed); }
    float bed_click_volume() const { return vol_.load(std::memory_order_relaxed); }
    bool bed_mono() const { return mono_.load(std::memory_order_relaxed); }
    void bed_write(const float* ml, const float* mr, const float* mc, const float* flag, int n) {
        const float* ch[4] = {ml, mr, mc, flag};
        bed_.write(ch, n);
    }
    int64_t pad_underruns() const { return pad_under_.load(std::memory_order_relaxed); }
    int64_t bed_underruns() const { return bed_under_.load(std::memory_order_relaxed); }

private:
    Ring pad_, bed_;
    std::atomic<bool> pad_active_{false}, bed_state_{false};
    std::atomic<bool> instant_{true}, want_{false}, mono_{false};
    std::atomic<float> vol_{0.7f};
    std::atomic<int64_t> pad_under_{0}, bed_under_{0};
    bool bed_flag_ = false;          // (só a thread de áudio)
    int starve_ = 0;
    bool pad_warm_ = false, bed_warm_ = false;   // (só a thread de áudio) já houve 1ª leitura desde a partida?
    std::vector<float> flag_;
};

}  // namespace lt
