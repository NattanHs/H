#include "core.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#if defined(__SSE__) || defined(_M_X64) || defined(_M_AMD64)
#include <xmmintrin.h>
#define LT_HAVE_SSE 1
#endif

namespace lt {

namespace {

constexpr float INV16 = 1.0f / 32768.0f;
constexpr double DUCK_QUIET_DB = -52.0;
constexpr double DUCK_LOUD_DB = -38.0;
constexpr double DUCK_FLOOR = 0.02;
constexpr double PI = 3.14159265358979323846;

// zera números desnormalizados enquanto o render roda (evita picos de CPU nas caudas do pad)
struct NoDenormals {
#ifdef LT_HAVE_SSE
    unsigned int saved;
    NoDenormals() : saved(_mm_getcsr()) { _mm_setcsr(saved | 0x8040u); }
    ~NoDenormals() { _mm_setcsr(saved); }
#endif
};

inline void route_add(float* out, const float* l, const float* r, int n, int route) {
    if (route == ROUTE_LEFT) {
        for (int i = 0; i < n; ++i) out[2 * i] += 0.5f * (l[i] + r[i]);
    } else if (route == ROUTE_RIGHT) {
        for (int i = 0; i < n; ++i) out[2 * i + 1] += 0.5f * (l[i] + r[i]);
    } else {
        for (int i = 0; i < n; ++i) {
            out[2 * i] += l[i];
            out[2 * i + 1] += r[i];
        }
    }
}

inline double pymod(double a, double b) {          // % do Python para floats (sinal do divisor)
    double r = std::fmod(a, b);
    if (r != 0.0 && ((r < 0.0) != (b < 0.0))) r += b;
    return r;
}

}  // namespace

// ============================================================================= construção
Core::Core(int sample_rate) : sr_(sample_rate), limiter_(96, 0.97f) {
    for (auto& p : params_) p.store(0.0);
    params_[P_MASTER] = 1.0;
    params_[P_BPM] = 120.0;
    params_[P_BPB] = 4;
    params_[P_QUANTIZE] = QUANT_BAR;
    params_[P_CLICK_VOLUME] = 0.7;
    params_[P_PAD_VOLUME] = 0.5;
    params_[P_PAD_FADE_OUT] = 4.0;
    bus_.assign(static_cast<size_t>(MAX_CHUNK) * 2, 0.0f);
    tmp_a_.assign(MAX_CHUNK, 0.0f);
    tmp_b_.assign(MAX_CHUNK, 0.0f);
    tmp_c_.assign(MAX_CHUNK, 0.0f);
    click_buf_.assign(MAX_CHUNK, 0.0f);
    ramp_.assign(MAX_CHUNK, 0.0f);
    gain_buf_.assign(MAX_CHUNK, 0.0f);
    w_new_.assign(MAX_CHUNK, 0.0f);
    w_old_.assign(MAX_CHUNK, 0.0f);
    gbuf_.assign(MAX_CHUNK, 0.0);
    decay_.assign(MAX_CHUNK, 0.0);
    for (int i = 0; i < MAX_CHUNK; ++i) decay_[i] = std::exp(-(i + 1.0) / (0.6 * sr_));
    cur_list_ = std::make_shared<TrackList>();
    list_.store(cur_list_.get());
    retired_.reserve(64);
}

Core::~Core() {
    provider_.store(nullptr);
    retired_.clear();
    cur_list_.reset();
}

// ============================================================================= interface
void Core::set_param(Param p, double v) {
    params_[p].store(v, std::memory_order_relaxed);
    if (p == P_BPM || p == P_GRID_OFFSET || p == P_BPB || p == P_QUANTIZE)
        grid_version_.fetch_add(1, std::memory_order_relaxed);
}

double Core::get_param(Param p) const { return params_[p].load(std::memory_order_relaxed); }

void Core::set_click_waves(const float* accent, int na, const float* normal, int nn) {
    waves_.emplace_back(new std::vector<float>(accent, accent + na));
    const std::vector<float>* a = waves_.back().get();
    waves_.emplace_back(new std::vector<float>(normal, normal + nn));
    const std::vector<float>* b = waves_.back().get();
    wave_a_.store(a);
    wave_n_.store(b);
}

void Core::set_provider(std::shared_ptr<ExtProvider> prov) {
    providers_.push_back(prov);             // nunca liberado antes do fim: a thread de áudio pode estar usando
    provider_.store(prov.get());
}

std::shared_ptr<TrackState> Core::make_track() {
    auto t = std::make_shared<TrackState>();
    all_tracks_.push_back(t);
    return t;
}

void Core::retire(std::shared_ptr<const void> keep, const void* ptr) {
    const uint64_t e = epoch_.load(std::memory_order_seq_cst);
    retired_.push_back({std::move(keep), ptr, (e & 1u) ? e + 1 : e});
}

bool Core::referenced(const void* ptr) {
    for (auto& w : all_tracks_) {
        auto t = w.lock();
        if (!t) continue;
        if (t->using_p.load() == ptr || t->xf_old_p.load() == ptr) return true;
    }
    return false;
}

void Core::collect() {
    if (retired_.empty()) {
        return;
    }
    const uint64_t e = epoch_.load(std::memory_order_seq_cst);
    std::vector<Retired> keep;
    std::vector<Retired> drop;
    for (auto& r : retired_) {
        const bool grace = e >= r.stamp;
        if (grace && !referenced(r.ptr)) drop.push_back(std::move(r));
        else keep.push_back(std::move(r));
    }
    retired_ = std::move(keep);
    drop.clear();                        // aqui os arrays do numpy são soltos (interface, com a GIL)
    all_tracks_.erase(std::remove_if(all_tracks_.begin(), all_tracks_.end(),
                                     [](const std::weak_ptr<TrackState>& w) { return w.expired(); }),
                      all_tracks_.end());
}

void Core::refresh_total() {
    int64_t total = 0;
    for (auto& t : ui_tracks_) {
        if (t->data_own) total = std::max(total, t->data_own->frames);
    }
    total_.store(total, std::memory_order_relaxed);
}

void Core::publish_list() {
    auto neu = std::make_shared<TrackList>();
    neu->own = ui_tracks_;
    neu->raw.reserve(neu->own.size());
    for (auto& t : neu->own) neu->raw.push_back(t.get());
    auto old = cur_list_;
    cur_list_ = neu;
    list_.store(neu.get(), std::memory_order_seq_cst);
    retire(old, old.get());
    refresh_total();
    collect();
}

void Core::track_set_data(const std::shared_ptr<TrackState>& t, std::shared_ptr<Samples> s) {
    t->reset_gain.store(true);
    auto old = std::move(t->data_own);
    t->data_own = s;
    t->data.store(s.get(), std::memory_order_seq_cst);
    if (old) retire(old, old.get());
    refresh_total();
    collect();
}

void Core::track_set_shifted(const std::shared_ptr<TrackState>& t, std::shared_ptr<Samples> s) {
    auto old = std::move(t->shifted_own);
    t->shifted_own = s;
    t->shifted.store(s.get(), std::memory_order_seq_cst);
    if (old) retire(old, old.get());
    collect();
}

void Core::set_tracks(const std::vector<std::shared_ptr<TrackState>>& tracks, bool reset) {
    if (reset)
        for (auto& t : tracks) t->reset_gain.store(true);
    ui_tracks_ = tracks;
    publish_list();
}

void Core::add_track(const std::shared_ptr<TrackState>& t) {
    ui_tracks_.push_back(t);
    publish_list();
}

void Core::remove_track(const std::shared_ptr<TrackState>& t) {
    ui_tracks_.erase(std::remove(ui_tracks_.begin(), ui_tracks_.end(), t), ui_tracks_.end());
    publish_list();
}

void Core::clear_tracks() {
    ui_tracks_.clear();
    publish_list();
}

void Core::push_cmd(int type, int64_t arg) {
    const uint32_t w = cmd_w_.load(std::memory_order_relaxed);
    const uint32_t r = cmd_r_.load(std::memory_order_acquire);
    if (w - r >= CMD_CAP) return;           // fila cheia (nunca acontece na prática)
    cmds_[w % CMD_CAP] = {type, arg};
    cmd_w_.store(w + 1, std::memory_order_release);
}

bool Core::pop_cmd(Cmd& c) {
    const uint32_t r = cmd_r_.load(std::memory_order_relaxed);
    if (r == cmd_w_.load(std::memory_order_acquire)) return false;
    c = cmds_[r % CMD_CAP];
    cmd_r_.store(r + 1, std::memory_order_release);
    return true;
}

void Core::play(int64_t delay_frames) {
    params_[P_AMBIENT_SILENCED].store(0.0);
    push_cmd(CMD_PLAY, std::max<int64_t>(0, delay_frames));
}
void Core::cancel_start() { push_cmd(CMD_CANCEL_START, 0); }
void Core::pause() { push_cmd(CMD_PAUSE, 0); }
void Core::stop(bool silence) {
    if (silence) params_[P_AMBIENT_SILENCED].store(1.0);
    jump_req_.store(-1);
    clear_loop();
    push_cmd(CMD_STOP, 0);
}
void Core::seek(int64_t frames) { push_cmd(CMD_SEEK, frames); }
void Core::jump(int64_t frame) { jump_req_.store(std::max<int64_t>(0, frame)); }
void Core::cancel_jump() { jump_req_.store(-1); }

void Core::set_loop(int64_t start, int64_t end) {
    loop_seq_.fetch_add(1, std::memory_order_acq_rel);          // ímpar: escrevendo
    if (end > start) {
        loop_start_.store(std::max<int64_t>(0, start));
        loop_end_.store(end);
        loop_on_.store(true);
    } else {
        loop_on_.store(false);
    }
    loop_seq_.fetch_add(1, std::memory_order_acq_rel);
}
void Core::clear_loop() {
    loop_seq_.fetch_add(1, std::memory_order_acq_rel);
    loop_on_.store(false);
    loop_seq_.fetch_add(1, std::memory_order_acq_rel);
}

bool Core::read_loop(int64_t& s, int64_t& e) const {
    for (int tries = 0; tries < 4; ++tries) {
        const uint32_t a = loop_seq_.load(std::memory_order_acquire);
        if (a & 1u) continue;
        const bool on = loop_on_.load();
        s = loop_start_.load();
        e = loop_end_.load();
        if (loop_seq_.load(std::memory_order_acquire) == a) return on;
    }
    return false;
}

bool Core::loop_region(int64_t& start, int64_t& end) const { return read_loop(start, end); }

void Core::reset_diagnostics() {
    max_us_.store(0.0);
    slow_blocks_.store(0);
    calls_.store(0);
}

// ============================================================================= áudio
void Core::render(float* out, int n) {
    NoDenormals nd;
    const auto t0 = std::chrono::steady_clock::now();
    epoch_.fetch_add(1, std::memory_order_seq_cst);                 // ímpar = renderizando
    int i = 0;
    while (i < n) {
        int m = std::min(MAX_CHUNK, n - i);
        Split sp;
        if (jump_req_.load(std::memory_order_relaxed) >= 0 || loop_on_.load(std::memory_order_relaxed))
            m = plan_split(m, sp);
        render_chunk(out + static_cast<size_t>(2) * i, m, sp);
        i += m;
    }
    epoch_.fetch_add(1, std::memory_order_seq_cst);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    last_us_.store(us, std::memory_order_relaxed);
    if (us > max_us_.load(std::memory_order_relaxed)) max_us_.store(us, std::memory_order_relaxed);
    const double budget = 1e6 * n / sr_;
    if (us > 0.8 * budget) slow_blocks_.fetch_add(1, std::memory_order_relaxed);
    calls_.fetch_add(1, std::memory_order_relaxed);
}

int Core::plan_split(int m, Split& sp) {
    sp.valid = false;
    if (!playing_ || pausing_) return m;
    const int64_t pos = pos_;
    const int64_t total = total_.load(std::memory_order_relaxed);
    bool have = false;
    int64_t bd = 0, bt = 0;
    int bk = 0;
    const int64_t jump = jump_req_.load(std::memory_order_relaxed);
    if (jump >= 0) {
        const double bpm = params_[P_BPM].load(std::memory_order_relaxed);
        const int bpb = static_cast<int>(params_[P_BPB].load(std::memory_order_relaxed));
        const double off = static_cast<double>(static_cast<int64_t>(params_[P_GRID_OFFSET].load(std::memory_order_relaxed)));
        const double bar = 60.0 * sr_ / bpm * bpb;
        const double r = pymod(static_cast<double>(pos) - off, bar);
        int64_t d = (r < 1e-6 || bar - r < 1e-6) ? 0 : static_cast<int64_t>(std::ceil(bar - r - 1e-6));
        if (d < FADE_FRAMES) d += static_cast<int64_t>(std::nearbyint(bar));
        have = true;
        bd = d;
        bt = std::min<int64_t>(jump, std::max<int64_t>(0, total - 1));
        bk = 0;
    }
    int64_t ls, le;
    if (read_loop(ls, le) && pos < le) {
        const int64_t dl = le - pos;
        if (!have || dl < bd) {
            have = true;
            bd = dl;
            bt = std::min<int64_t>(ls, std::max<int64_t>(0, total - 1));
            bk = 1;
        }
    }
    if (!have) return m;
    if (bd <= m) {
        sp.valid = true;
        sp.target = bt;
        sp.kind = bk;
        return static_cast<int>(bd);
    }
    if (bd - m < FADE_FRAMES) m = static_cast<int>(std::max<int64_t>(1, bd - FADE_FRAMES));
    return m;
}

void Core::schedule(double b0, int a, int b, double bf, int bpb, std::array<Event, 256>& ev, int& nev) {
    const int length = b - a;
    if (length <= 0) return;
    if (!beat_init_ || std::fabs(b0 - expected_b_) > 1e-3) {
        last_beat_ = static_cast<int64_t>(std::ceil(b0 - 1e-9)) - 1;
        beat_init_ = true;
    }
    while (true) {
        const int64_t nb = last_beat_ + 1;
        const double fr = (static_cast<double>(nb) - b0) * bf;
        if (fr >= length - 1e-9) break;
        const int idx = std::min(b - 1, a + std::max(0, static_cast<int>(std::ceil(fr - 1e-6))));
        if (nev < static_cast<int>(ev.size())) {
            int64_t bm = nb % bpb;
            if (bm < 0) bm += bpb;
            ev[nev++] = {idx, static_cast<int>(bm)};
        }
        last_beat_ = nb;
    }
    expected_b_ = b0 + length / bf;
}

void Core::mix_tracks(float* bus, int frames, int64_t pos, const TrackList* tl) {
    bool any_solo = false;
    for (auto* t : tl->raw)
        if (t->solo.load(std::memory_order_relaxed)) { any_solo = true; break; }
    // rampa de ganho dentro do bloco: (k+1)/frames
    const float inv_frames = 1.0f / static_cast<float>(frames);

    for (TrackState* t : tl->raw) {
        const Samples* d = t->data.load(std::memory_order_seq_cst);
        if (!d) continue;
        if (t->reset_gain.load(std::memory_order_relaxed)) {
            t->last_gain = 0.0;
            t->reset_gain.store(false);
            t->using_p.store(nullptr);
            t->xf_old_p.store(nullptr);
            t->xf_left = 0;
        }
        const Samples* sh = t->shifted.load(std::memory_order_seq_cst);
        const bool melodic = t->melodic.load(std::memory_order_relaxed);
        const Samples* cur = (melodic && sh && sh->frames == d->frames && sh->channels == d->channels) ? sh : d;
        const Samples* usg = t->using_p.load();
        if (!usg) {
            t->using_p.store(cur);
        } else if (usg != cur) {
            t->xf_old_p.store(usg);
            t->xf_left = XFADE_FRAMES;
            t->using_p.store(cur);
        }
        const bool audible = any_solo ? t->solo.load(std::memory_order_relaxed) : !t->mute.load(std::memory_order_relaxed);
        const double target = audible ? static_cast<double>(t->volume.load(std::memory_order_relaxed)) *
                                            static_cast<double>(t->level.load(std::memory_order_relaxed))
                                      : 0.0;
        const double g0 = t->last_gain;
        if (g0 == 0.0 && target == 0.0) {
            t->xf_old_p.store(nullptr);
            t->xf_left = 0;
            t->peak.store(0.0f, std::memory_order_relaxed);
            continue;
        }
        const int64_t avail = d->frames - pos;
        const int n = static_cast<int>(std::min<int64_t>(frames, std::max<int64_t>(0, avail)));
        if (n > 0) {
            const int ch = cur->channels;
            const int16_t* seg = cur->p + static_cast<size_t>(pos) * ch;
            int mx = -32768, mn = 32767;
            for (int64_t k = 0; k < static_cast<int64_t>(n) * ch; ++k) {
                const int v = seg[k];
                if (v > mx) mx = v;
                if (v < mn) mn = v;
            }
            const double gm = std::max(g0, target);
            const double pk = std::max(mx, -mn) * gm / 32768.0;
            const double oldpk = t->peak.load(std::memory_order_relaxed) * 0.97;
            t->peak.store(static_cast<float>(pk > oldpk ? pk : oldpk), std::memory_order_relaxed);
            if (t->role.load(std::memory_order_relaxed) == ROLE_MUSIC) {
                double acc = 0.0;
                int64_t count = 0;
                for (int k = 0; k < n; k += 4) {
                    for (int c = 0; c < ch; ++c) {
                        const double v = seg[static_cast<size_t>(k) * ch + c];
                        acc += v * v;
                    }
                    count += ch;
                }
                const double inv16 = static_cast<double>(INV16);
                music_power_ += acc / static_cast<double>(std::max<int64_t>(1, count)) * gm * gm * inv16 * inv16;
            }

            const bool ramping = (g0 != target);
            const float fg0 = static_cast<float>(g0);
            const float fdelta = static_cast<float>(target - g0);
            const float fscalar = static_cast<float>(target) * INV16;
            const Samples* old = t->xf_old_p.load();
            const bool xfading = (t->xf_left > 0 && old != nullptr);
            const int route = t->route.load(std::memory_order_relaxed);

            auto gain_at = [&](int k) -> float {
                return ramping ? (fg0 + fdelta * (static_cast<float>(k + 1) * inv_frames)) * INV16 : fscalar;
            };

            auto add_source = [&](const Samples* src, int ncount, const float* wts) {
                const int sch = src->channels;
                const int16_t* sp = src->p + static_cast<size_t>(pos) * sch;
                if (sch == 1) {
                    for (int k = 0; k < ncount; ++k) {
                        float g = gain_at(k);
                        if (wts) g *= wts[k];
                        const float v = static_cast<float>(sp[k]) * g;
                        if (route == ROUTE_STEREO) {
                            bus[2 * k] += v;
                            bus[2 * k + 1] += v;
                        } else {
                            bus[2 * k + (route == ROUTE_LEFT ? 0 : 1)] += v;
                        }
                    }
                } else if (route == ROUTE_STEREO) {
                    for (int k = 0; k < ncount; ++k) {
                        float g = gain_at(k);
                        if (wts) g *= wts[k];
                        bus[2 * k] += static_cast<float>(sp[2 * k]) * g;
                        bus[2 * k + 1] += static_cast<float>(sp[2 * k + 1]) * g;
                    }
                } else {
                    const int dst = route == ROUTE_LEFT ? 0 : 1;
                    for (int k = 0; k < ncount; ++k) {
                        float g = gain_at(k);
                        if (wts) g *= wts[k];
                        const float mono = static_cast<float>(sp[2 * k]) + static_cast<float>(sp[2 * k + 1]);
                        bus[2 * k + dst] += mono * (0.5f * g);
                    }
                }
            };

            if (xfading) {
                const int m = static_cast<int>(std::min<int64_t>(n, t->xf_left));
                for (int k = 0; k < n; ++k) {
                    w_new_[k] = 1.0f;
                    w_old_[k] = 0.0f;
                }
                for (int k = 0; k < m; ++k) {
                    const float w = (static_cast<float>(k + 1) + static_cast<float>(XFADE_FRAMES - t->xf_left)) /
                                    static_cast<float>(XFADE_FRAMES);
                    w_new_[k] = w;
                    w_old_[k] = 1.0f - w;
                }
                add_source(cur, n, w_new_.data());
                const int n_old = static_cast<int>(std::min<int64_t>(n, std::max<int64_t>(0, old->frames - pos)));
                if (n_old > 0) add_source(old, n_old, w_old_.data());
                t->xf_left -= m;
                if (t->xf_left <= 0) t->xf_old_p.store(nullptr);
            } else {
                add_source(cur, n, nullptr);
            }
        }
        t->last_gain = target;
    }
}

void Core::render_chunk(float* out, int n, const Split& split) {
    std::fill(out, out + static_cast<size_t>(2) * n, 0.0f);
    const TrackList* tl = list_.load(std::memory_order_seq_cst);
    const int64_t total = total_.load(std::memory_order_relaxed);
    const int sr = sr_;
    const double bpm = params_[P_BPM].load(std::memory_order_relaxed);
    const double bf = 60.0 * sr / bpm;
    const int64_t off = static_cast<int64_t>(params_[P_GRID_OFFSET].load(std::memory_order_relaxed));
    const int bpb = static_cast<int>(params_[P_BPB].load(std::memory_order_relaxed));

    // ---- comandos ----
    bool have_seek = false, cancel = false;
    int64_t seek = 0;
    int cmd = -1;
    int64_t cmd_arg = 0;
    Cmd c;
    while (pop_cmd(c)) {
        if (c.type == CMD_SEEK) {
            have_seek = true;
            seek = c.arg;
        } else if (c.type == CMD_CANCEL_START) {
            cancel = true;
            cmd = -1;
        } else {
            cmd = c.type;
            cmd_arg = c.arg;
        }
    }

    bool recompute = false;
    if (cancel) start_pending_ = false;
    if (have_seek) {
        pos_ = std::max<int64_t>(0, std::min(seek, total));
        if (playing_) fade_in_left_ = FADE_FRAMES;
        else if (start_pending_) recompute = true;
    }
    if (cmd == CMD_PLAY) {
        if (!playing_ && !start_pending_ && total > 0) {
            if (pos_ >= total) pos_ = 0;
            start_pending_ = true;
            start_not_before_ = clock_ + cmd_arg;
            recompute = true;
        }
    } else if (cmd == CMD_PAUSE || cmd == CMD_STOP) {
        start_pending_ = false;
        if (playing_) {
            pausing_ = true;
            stop_to_start_ = (cmd == CMD_STOP);
        } else if (cmd == CMD_STOP) {
            pos_ = 0;
        }
    }
    if (total <= 0) {
        start_pending_ = false;
        if (playing_) pausing_ = true;
    }
    const uint32_t gv = grid_version_.load(std::memory_order_relaxed);
    if (start_pending_ && gv != last_grid_version_) recompute = true;
    last_grid_version_ = gv;

    // ---- agenda o início do projeto, alinhado à grade do click ----
    if (start_pending_ && recompute) {
        start_pos_ = pos_;
        const int quant = static_cast<int>(params_[P_QUANTIZE].load(std::memory_order_relaxed));
        const double q = quant == QUANT_BAR ? static_cast<double>(bpb) : (quant == QUANT_BEAT ? 1.0 : 0.0);
        const int64_t wait = std::max<int64_t>(0, start_not_before_ - clock_);
        int64_t s = wait;
        if (q > 0.0) {
            const double target = static_cast<double>(start_pos_ - off) / bf;
            double d = std::fmod(target - (ambient_beat_ + static_cast<double>(wait) / bf), q);
            if (d < 0.0) d += q;
            if (d > q - 1e-6) d = 0.0;
            s = wait + std::max<int64_t>(0, static_cast<int64_t>(std::ceil(d * bf - 1e-6)));
        } else if (wait == 0) {
            s = 0;
        }
        start_at_ = clock_ + s;
    }

    const bool was_playing = playing_;
    const int64_t pos_at_start = pos_;
    int proj_start = n;
    bool starts_now = false;
    if (start_pending_ && start_at_ < clock_ + n) {
        proj_start = static_cast<int>(std::max<int64_t>(0, start_at_ - clock_));
        starts_now = true;
    } else if (playing_) {
        proj_start = 0;
    }
    if (starts_now) {
        playing_ = true;
        start_pending_ = false;
        pos_ = start_pos_;
        fade_in_left_ = FADE_FRAMES;
    }

    // ---- batidas do click ----
    std::array<Event, 256> events;
    int nev = 0;
    if (was_playing) {
        schedule(static_cast<double>(pos_at_start - off) / bf, 0, n, bf, bpb, events, nev);
    } else if (starts_now) {
        schedule(ambient_beat_, 0, proj_start, bf, bpb, events, nev);
        schedule(static_cast<double>(start_pos_ - off) / bf, proj_start, n, bf, bpb, events, nev);
    } else {
        schedule(ambient_beat_, 0, n, bf, bpb, events, nev);
    }
    ambient_beat_ = expected_b_;

    // ---- projeto ----
    const bool gate_active = was_playing || starts_now;
    const int proj_frames = n - proj_start;
    music_power_ = 0.0;
    if (playing_ && proj_frames > 0) {
        float* bus = bus_.data();
        std::fill(bus, bus + static_cast<size_t>(2) * proj_frames, 0.0f);
        mix_tracks(bus, proj_frames, pos_, tl);
        if (fade_in_left_ > 0) {
            const int m = std::min(fade_in_left_, proj_frames);
            for (int k = 0; k < m; ++k) {
                const float g = static_cast<float>(1.0 - static_cast<double>(fade_in_left_ - k) / FADE_FRAMES);
                bus[2 * k] *= g;
                bus[2 * k + 1] *= g;
            }
            fade_in_left_ -= m;
        }
        if (pausing_) {
            for (int k = 0; k < proj_frames; ++k) {
                const float g = static_cast<float>(1.0 - static_cast<double>(k + 1) / proj_frames);
                bus[2 * k] *= g;
                bus[2 * k + 1] *= g;
            }
        }
        if (split.valid && !pausing_) {
            const int k = std::min(proj_frames, FADE_FRAMES);
            for (int j = 0; j < k; ++j) {
                const float g = static_cast<float>(k - j) / static_cast<float>(k);
                const int idx = proj_frames - k + j;
                bus[2 * idx] *= g;
                bus[2 * idx + 1] *= g;
            }
        }
        float* dst = out + static_cast<size_t>(2) * proj_start;
        for (int k = 0; k < 2 * proj_frames; ++k) dst[k] += bus[k];
        pos_ += proj_frames;
        if (split.valid && !pausing_) {
            pos_ = split.target;
            fade_in_left_ = FADE_FRAMES;
            if (split.kind == 0) jump_req_.store(-1);
        } else if (pausing_) {
            playing_ = false;
            pausing_ = false;
            if (stop_to_start_) pos_ = 0;
            stop_to_start_ = false;
        } else if (pos_ >= total) {
            playing_ = false;
            pos_ = 0;
            ++ended_s_;
            ended_count_.store(ended_s_, std::memory_order_relaxed);
        }
    }

    ExtProvider* prov = provider_.load(std::memory_order_acquire);

    // ---- cama de ministração ----
    const bool ministry_on = params_[P_MINISTRY_ON].load(std::memory_order_relaxed) != 0.0;
    const bool want_bed = ministry_on && !gate_active;
    const bool bed_sounds = want_bed || (prov && prov->bed_active());
    const bool silenced = params_[P_AMBIENT_SILENCED].load(std::memory_order_relaxed) != 0.0;
    const float click_volume = static_cast<float>(params_[P_CLICK_VOLUME].load(std::memory_order_relaxed));

    // ---- click ----
    const int cm = static_cast<int>(params_[P_CLICK_MODE].load(std::memory_order_relaxed));
    double click_target = 0.0;
    if ((cm == GATE_ALWAYS || (cm == GATE_PAUSED && !gate_active)) && !bed_sounds && !silenced)
        click_target = click_volume;
    if (click_target > 0.0 || click_gain_ > 1e-4 || nvoices_ > 0) {
        const std::vector<float>* wa = wave_a_.load();
        const std::vector<float>* wn = wave_n_.load();
        if (wa && wn) {
            if (click_target > 0.0 || click_gain_ > 1e-4) {
                for (int e = 0; e < nev; ++e) {
                    const std::vector<float>* w = events[e].beat == 0 ? wa : wn;
                    if (nvoices_ < static_cast<int>(voices_.size()))
                        voices_[nvoices_++] = {w->data(), static_cast<int>(w->size()), 0, events[e].idx};
                }
            }
            float* buf = click_buf_.data();
            std::fill(buf, buf + n, 0.0f);
            int alive = 0;
            for (int v = 0; v < nvoices_; ++v) {
                Voice vo = voices_[v];
                const int m = std::min(vo.len - vo.i, n - vo.delay);
                if (m > 0) {
                    for (int k = 0; k < m; ++k) buf[vo.delay + k] += vo.wave[vo.i + k];
                    vo.i += m;
                }
                vo.delay = 0;
                if (vo.i < vo.len) voices_[alive++] = vo;
            }
            nvoices_ = alive;
            const float g0 = static_cast<float>(click_gain_);
            const float dg = static_cast<float>(click_target - click_gain_);
            for (int k = 0; k < n; ++k) buf[k] *= g0 + dg * (static_cast<float>(k + 1) / static_cast<float>(n));
            click_gain_ = click_target;
            route_add(out, buf, buf, n, static_cast<int>(params_[P_CLICK_ROUTE].load(std::memory_order_relaxed)));
        }
    }

    // ---- volume da música (guia o pad sob a música) ----
    const double p_now = n ? music_power_ * (static_cast<double>(proj_frames) / n) : 0.0;
    double env = music_env_;
    const double a_env = p_now > env ? 1.0 - std::exp(-static_cast<double>(n) / (0.10 * sr))
                                      : 1.0 - std::exp(-static_cast<double>(n) / (0.60 * sr));
    env += (p_now - env) * a_env;
    music_env_ = env;
    const double db = 10.0 * std::log10(env + 1e-12);
    double t_loud = std::min(1.0, std::max(0.0, (db - DUCK_QUIET_DB) / (DUCK_LOUD_DB - DUCK_QUIET_DB)));
    t_loud = t_loud * t_loud * (3.0 - 2.0 * t_loud);
    duck_ = DUCK_FLOOR + (1.0 - DUCK_FLOOR) * (1.0 - t_loud);

    // ---- pad ----
    const int pm = static_cast<int>(params_[P_PAD_MODE].load(std::memory_order_relaxed));
    const double pad_volume = params_[P_PAD_VOLUME].load(std::memory_order_relaxed);
    const double under = std::min(1.0, std::max(0.0, params_[P_PAD_UNDER].load(std::memory_order_relaxed)));
    const double pad_fade_out = params_[P_PAD_FADE_OUT].load(std::memory_order_relaxed);
    const int pad_route = static_cast<int>(params_[P_PAD_ROUTE].load(std::memory_order_relaxed));
    double pad_target = 0.0;
    if (pm != GATE_OFF && !bed_sounds && !silenced) {
        if (!gate_active) pad_target = pad_volume;
        else if (under > 0.0) pad_target = pad_volume * under * duck_;
        else if (pm == GATE_ALWAYS) pad_target = pad_volume;
    }
    const bool releasing = (pad_target == 0.0 && gate_active && !bed_sounds && !silenced && pad_fade_out > 0.05 &&
                            (pad_gain_ > 1e-4 || rel_total_ > 0));
    if (pad_target > 0.0 || pad_gain_ > 1e-4) {
        float* pl = tmp_a_.data();
        float* pr = tmp_b_.data();
        if (prov) {
            prov->pad_render(pad_gain_ < 1e-3, n, pl, pr);
        } else {
            std::fill(pl, pl + n, 0.0f);
            std::fill(pr, pr + n, 0.0f);
        }
        double* g = gbuf_.data();
        if (releasing) {
            if (rel_total_ == 0) {
                rel_total_ = std::max<int64_t>(1, static_cast<int64_t>(pad_fade_out * sr));
                rel_done_ = 0;
                rel_start_gain_ = pad_gain_;
            }
            const int t0 = starts_now ? proj_start : 0;
            for (int k = 0; k < n; ++k) {
                const double kk = static_cast<double>(k + 1 - t0);
                double prog = (static_cast<double>(rel_done_) + std::max(kk, 0.0)) / static_cast<double>(rel_total_);
                prog = std::min(1.0, std::max(0.0, prog));
                g[k] = rel_start_gain_ * 0.5 * (1.0 + std::cos(PI * prog));
                if (rel_done_ + std::max(0, n - t0) >= rel_total_ && prog >= 1.0) g[k] = 0.0;
            }
            rel_done_ += std::max(0, n - t0);
        } else {
            rel_total_ = 0;
            if (gate_active && under > 0.0) {
                const double tau = pad_target > pad_gain_ ? 1.5 : std::max(0.5, pad_fade_out / 4.0);
                for (int k = 0; k < n; ++k)
                    g[k] = pad_target + (pad_gain_ - pad_target) * std::exp(-(k + 1.0) / (tau * sr));
            } else {
                for (int k = 0; k < n; ++k) g[k] = pad_target + (pad_gain_ - pad_target) * decay_[k];
            }
        }
        pad_gain_ = g[n - 1];
        for (int k = 0; k < n; ++k) {
            const float gf = static_cast<float>(g[k]);
            pl[k] *= gf;
            pr[k] *= gf;
        }
        route_add(out, pl, pr, n, pad_route);
    } else {
        pad_gain_ = 0.0;
        rel_total_ = 0;
        if (prov) prov->pad_idle();
    }

    // ---- cama de ministração (pad em camadas + click próprio) ----
    if (bed_sounds && prov) {
        float* ml = tmp_a_.data();
        float* mr = tmp_b_.data();
        float* mc = tmp_c_.data();
        prov->bed_render(n, want_bed, click_volume, pad_route != ROUTE_STEREO, ml, mr, mc);
        route_add(out, ml, mr, n, pad_route);
        route_add(out, mc, mc, n, static_cast<int>(params_[P_CLICK_ROUTE].load(std::memory_order_relaxed)));
    } else if (prov) {
        prov->bed_idle();
    }

    // ---- master (com limitador suave) ----
    const float master = static_cast<float>(params_[P_MASTER].load(std::memory_order_relaxed));
    if (master != 1.0f)
        for (int k = 0; k < 2 * n; ++k) out[k] *= master;
    float pl_max = 0.0f, pr_max = 0.0f, amax = 0.0f;
    for (int k = 0; k < n; ++k) {
        const float a0 = std::fabs(out[2 * k]);
        const float a1 = std::fabs(out[2 * k + 1]);
        if (a0 > pl_max) pl_max = a0;
        if (a1 > pr_max) pr_max = a1;
    }
    amax = std::max(pl_max, pr_max);
    peak_l_s_ = pl_max > peak_l_s_ * 0.97 ? pl_max : peak_l_s_ * 0.97;
    peak_r_s_ = pr_max > peak_r_s_ * 0.97 ? pr_max : peak_r_s_ * 0.97;
    if (params_[P_LIMITER_ON].load(std::memory_order_relaxed) != 0.0) {
        limiter_.process(out, n);
        for (int k = 0; k < 2 * n; ++k) out[k] = std::min(1.0f, std::max(-1.0f, out[k]));
    } else if (amax > 0.8f) {
        for (int k = 0; k < 2 * n; ++k) {
            const float a = std::fabs(out[k]);
            if (a > 0.8f) {
                const float s = out[k] < 0.0f ? -1.0f : 1.0f;
                out[k] = s * (0.8f + 0.2f * std::tanh((a - 0.8f) / 0.2f));
            }
        }
    }

    // ---- publicação para a UI ----
    if (nev > 0) {
        beat_in_bar_s_ = events[nev - 1].beat;
        beat_in_bar_.store(beat_in_bar_s_, std::memory_order_relaxed);
        beat_counter_s_ += nev;
        beat_counter_.store(beat_counter_s_, std::memory_order_relaxed);
    }
    position_.store(pos_, std::memory_order_relaxed);
    playing_pub_.store(playing_, std::memory_order_relaxed);
    armed_pub_.store(start_pending_, std::memory_order_relaxed);
    peak_l_.store(static_cast<float>(peak_l_s_), std::memory_order_relaxed);
    peak_r_.store(static_cast<float>(peak_r_s_), std::memory_order_relaxed);
    duck_pub_.store(static_cast<float>(duck_), std::memory_order_relaxed);
    pad_gain_pub_.store(pad_gain_, std::memory_order_relaxed);
    clock_ += n;
    clock_pub_.store(clock_, std::memory_order_relaxed);
}

}  // namespace lt
