// Núcleo do motor de áudio do LagoTracks (C++ puro, sem JUCE): mixagem, transporte, click, pad, limitador.
// É a tradução fiel do Engine de engine.py. Tudo o que a thread de áudio faz está em Core::render().
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "limiter.h"

namespace lt {

constexpr int MAX_CHUNK = 4096;
constexpr int FADE_FRAMES = 240;       // 5 ms a 48 kHz
constexpr int XFADE_FRAMES = 2400;     // 50 ms: troca entre original e transposta

enum Route { ROUTE_STEREO = 0, ROUTE_LEFT = 1, ROUTE_RIGHT = 2 };
enum Gate { GATE_OFF = 0, GATE_PAUSED = 1, GATE_ALWAYS = 2 };
enum Quant { QUANT_IMMEDIATE = 0, QUANT_BEAT = 1, QUANT_BAR = 2 };
enum Role { ROLE_MUSIC = 0, ROLE_CLICK = 1, ROLE_GUIDE = 2, ROLE_OTHER = 3 };

// Áudio de uma faixa: int16 mono (frames) ou estéreo (frames,2). Só leitura; `owner` mantém vivo o array do numpy.
struct Samples {
    const int16_t* p = nullptr;
    int64_t frames = 0;
    int channels = 1;
    std::shared_ptr<void> owner;
};

struct TrackState {
    // escritos pela interface, lidos pela thread de áudio
    std::atomic<float> volume{1.0f};
    std::atomic<float> level{1.0f};
    std::atomic<bool> mute{false};
    std::atomic<bool> solo{false};
    std::atomic<bool> melodic{true};
    std::atomic<int> route{ROUTE_STEREO};
    std::atomic<int> role{ROLE_MUSIC};
    std::atomic<const Samples*> data{nullptr};
    std::atomic<const Samples*> shifted{nullptr};
    std::atomic<bool> reset_gain{true};
    // escrito pela thread de áudio, lido pela interface
    std::atomic<float> peak{0.0f};
    // estado da thread de áudio (a interface só lê `using_p`/`xf_old_p` para saber se uma fonte ainda é usada)
    std::atomic<const Samples*> using_p{nullptr};
    std::atomic<const Samples*> xf_old_p{nullptr};
    double last_gain = 0.0;
    int64_t xf_left = 0;
    // dono das fontes (só a interface mexe)
    std::shared_ptr<Samples> data_own, shifted_own;
};

struct TrackList {
    std::vector<std::shared_ptr<TrackState>> own;
    std::vector<TrackState*> raw;
};

// Fonte externa do pad e da cama de ministração (sintetizadas em Python, fora da thread de áudio).
class ExtProvider {
public:
    virtual ~ExtProvider() = default;
    virtual void pad_idle() {}
    virtual void pad_render(bool instant_chord, int n, float* l, float* r) = 0;
    virtual bool bed_active() = 0;
    virtual void bed_idle() {}
    virtual void bed_render(int n, bool want, float click_volume, bool mono, float* ml, float* mr, float* mc) = 0;
};

enum Param {
    P_MASTER, P_BPM, P_GRID_OFFSET, P_BPB, P_QUANTIZE, P_CLICK_MODE, P_CLICK_VOLUME, P_CLICK_ROUTE,
    P_PAD_MODE, P_PAD_VOLUME, P_PAD_ROUTE, P_PAD_UNDER, P_PAD_FADE_OUT, P_LIMITER_ON, P_MINISTRY_ON,
    P_AMBIENT_SILENCED, P_COUNT
};

class Core {
public:
    explicit Core(int sample_rate);
    ~Core();
    Core(const Core&) = delete;
    Core& operator=(const Core&) = delete;

    int sample_rate() const { return sr_; }

    // ---------------------------------------------------------------- interface (thread da UI)
    void set_param(Param p, double v);
    double get_param(Param p) const;
    void set_click_waves(const float* accent, int na, const float* normal, int nn);
    void set_provider(std::shared_ptr<ExtProvider> prov);

    std::shared_ptr<TrackState> make_track();
    void track_set_data(const std::shared_ptr<TrackState>& t, std::shared_ptr<Samples> s);
    void track_set_shifted(const std::shared_ptr<TrackState>& t, std::shared_ptr<Samples> s);
    void set_tracks(const std::vector<std::shared_ptr<TrackState>>& tracks, bool reset);
    void add_track(const std::shared_ptr<TrackState>& t);
    void remove_track(const std::shared_ptr<TrackState>& t);
    void clear_tracks();
    void collect();               // libera o que a thread de áudio já não usa (chamar de vez em quando, com a GIL)

    void play(int64_t delay_frames);
    void cancel_start();
    void pause();
    void stop(bool silence);
    void seek(int64_t frames);
    void jump(int64_t frame);
    void cancel_jump();
    void set_loop(int64_t start, int64_t end);
    void clear_loop();
    int64_t pending_jump() const { return jump_req_.load(std::memory_order_relaxed); }   // -1 = nenhum
    bool loop_region(int64_t& start, int64_t& end) const;

    // estado publicado
    int64_t position() const { return position_.load(std::memory_order_relaxed); }
    bool playing() const { return playing_pub_.load(std::memory_order_relaxed); }
    bool armed() const { return armed_pub_.load(std::memory_order_relaxed); }
    int beat_in_bar() const { return beat_in_bar_.load(std::memory_order_relaxed); }
    int64_t beat_counter() const { return beat_counter_.load(std::memory_order_relaxed); }
    int64_t ended_count() const { return ended_count_.load(std::memory_order_relaxed); }
    float peak_l() const { return peak_l_.load(std::memory_order_relaxed); }
    float peak_r() const { return peak_r_.load(std::memory_order_relaxed); }
    float duck() const { return duck_pub_.load(std::memory_order_relaxed); }
    int64_t clock() const { return clock_pub_.load(std::memory_order_relaxed); }
    int64_t total_frames() const { return total_.load(std::memory_order_relaxed); }
    double pad_gain() const { return pad_gain_pub_.load(std::memory_order_relaxed); }
    // diagnóstico
    double last_render_us() const { return last_us_.load(std::memory_order_relaxed); }
    double max_render_us() const { return max_us_.load(std::memory_order_relaxed); }
    int64_t slow_blocks() const { return slow_blocks_.load(std::memory_order_relaxed); }
    int64_t render_calls() const { return calls_.load(std::memory_order_relaxed); }
    void reset_diagnostics();

    // ---------------------------------------------------------------- thread de áudio
    // Preenche n quadros intercalados (L,R). Qualquer n.
    void render(float* out, int n);

private:
    struct Cmd { int type; int64_t arg; };
    enum { CMD_PLAY, CMD_CANCEL_START, CMD_PAUSE, CMD_STOP, CMD_SEEK };
    struct Voice { const float* wave; int len; int i; int delay; };
    struct Split { bool valid = false; int64_t target = 0; int kind = 0; };   // kind 0 = pulo, 1 = repetição
    struct Event { int idx; int beat; };
    struct Retired { std::shared_ptr<const void> keep; const void* ptr; uint64_t stamp; };

    void push_cmd(int type, int64_t arg);
    bool pop_cmd(Cmd& c);
    bool read_loop(int64_t& s, int64_t& e) const;
    int plan_split(int m, Split& sp);
    void render_chunk(float* out, int n, const Split& split);
    void schedule(double b0, int a, int b, double bf, int bpb, std::array<Event, 256>& ev, int& nev);
    void mix_tracks(float* bus, int frames, int64_t pos, const TrackList* tl);
    void retire(std::shared_ptr<const void> keep, const void* ptr);
    bool referenced(const void* ptr);
    void refresh_total();
    void publish_list();

    int sr_;
    // parâmetros
    std::array<std::atomic<double>, P_COUNT> params_;
    std::atomic<uint32_t> grid_version_{0};
    // comandos
    static constexpr int CMD_CAP = 1024;
    std::array<Cmd, CMD_CAP> cmds_;
    std::atomic<uint32_t> cmd_w_{0}, cmd_r_{0};
    // pulo e repetição
    std::atomic<int64_t> jump_req_{-1};
    std::atomic<uint32_t> loop_seq_{0};
    std::atomic<int64_t> loop_start_{0}, loop_end_{0};
    std::atomic<bool> loop_on_{false};
    // faixas
    std::atomic<const TrackList*> list_{nullptr};
    std::vector<std::shared_ptr<TrackState>> ui_tracks_;      // lista vista pela interface
    std::shared_ptr<TrackList> cur_list_;                      // lista publicada para a thread de áudio
    std::vector<std::weak_ptr<TrackState>> all_tracks_;
    std::vector<Retired> retired_;
    std::atomic<int64_t> total_{0};
    std::atomic<uint64_t> epoch_{0};
    // provedor do pad/cama
    std::atomic<ExtProvider*> provider_{nullptr};
    std::vector<std::shared_ptr<ExtProvider>> providers_;
    // click
    std::vector<std::unique_ptr<std::vector<float>>> waves_;     // nunca liberadas até o fim (os cliques em andamento apontam para elas)
    std::atomic<const std::vector<float>*> wave_a_{nullptr}, wave_n_{nullptr};

    // ---- estado da thread de áudio ----
    int64_t clock_ = 0, pos_ = 0;
    bool playing_ = false, pausing_ = false, stop_to_start_ = false, start_pending_ = false;
    int64_t start_at_ = 0, start_pos_ = 0, start_not_before_ = 0;
    int fade_in_left_ = 0;
    uint32_t last_grid_version_ = 0;
    double ambient_beat_ = 0.0, expected_b_ = 0.0;
    int64_t last_beat_ = 0;
    bool beat_init_ = false;
    std::array<Voice, 32> voices_;
    int nvoices_ = 0;
    double click_gain_ = 0.0;
    double pad_gain_ = 0.0;
    int64_t rel_total_ = 0, rel_done_ = 0;
    double rel_start_gain_ = 0.0;
    double music_power_ = 0.0, music_env_ = 0.0, duck_ = 1.0;
    double peak_l_s_ = 0.0, peak_r_s_ = 0.0;
    int64_t ended_s_ = 0, beat_counter_s_ = 0;
    int beat_in_bar_s_ = 0;
    MasterLimiter limiter_;
    // buffers
    std::vector<float> bus_, tmp_a_, tmp_b_, tmp_c_, click_buf_, ramp_;
    std::vector<double> gbuf_, decay_;
    std::vector<float> gain_buf_, w_new_, w_old_;
    // publicados
    std::atomic<int64_t> position_{0}, beat_counter_{0}, ended_count_{0}, clock_pub_{0};
    std::atomic<bool> playing_pub_{false}, armed_pub_{false};
    std::atomic<int> beat_in_bar_{0};
    std::atomic<float> peak_l_{0.0f}, peak_r_{0.0f}, duck_pub_{1.0f};
    std::atomic<double> pad_gain_pub_{0.0};
    std::atomic<double> last_us_{0.0}, max_us_{0.0};
    std::atomic<int64_t> slow_blocks_{0}, calls_{0};
};

}  // namespace lt
