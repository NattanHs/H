// Versão sem JUCE: usada nos testes em Linux. O tipo "test" roda uma thread de áudio simulada (tempo real),
// para testar o caminho completo (filas do pad, troca de faixas) sem placa de som.
#include "device.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

namespace lt {

struct AudioDevice::Impl {
    Core& core;
    explicit Impl(Core& c) : core(c) {}
    std::thread th;
    std::atomic<bool> run{false};
    std::atomic<int> late{0};
    std::atomic<double> late_max_ms{0.0};
    int buf = 512;
    double sr = 48000.0;
    std::string desc;
};

AudioDevice::AudioDevice(Core& core) : impl_(new Impl(core)) {}
AudioDevice::~AudioDevice() {
    close();
    delete impl_;
}
bool AudioDevice::available() { return false; }
std::vector<std::string> AudioDevice::types() { return {"test"}; }
std::vector<DeviceInfo> AudioDevice::list() { return {DeviceInfo{"test", "Simulado", 2, 48000.0}}; }

std::string AudioDevice::open(const std::string& type, const std::string&, int sample_rate, int buffer_size) {
    if (type != "test") return "Esta compilação não tem a saída de áudio JUCE.";
    close();
    impl_->buf = buffer_size > 0 ? buffer_size : 512;
    impl_->sr = sample_rate;
    impl_->late = 0;
    impl_->late_max_ms = 0.0;
    impl_->desc = "teste simulado " + std::to_string(impl_->buf) + " quadros";
    impl_->run = true;
    Impl* im = impl_;
    im->th = std::thread([im]() {
        using clk = std::chrono::steady_clock;
        std::vector<float> out(static_cast<size_t>(im->buf) * 2);
        const auto period = std::chrono::duration_cast<clk::duration>(std::chrono::duration<double>(im->buf / im->sr));
        auto next = clk::now() + period;
        while (im->run.load()) {
            im->core.render(out.data(), im->buf);
            std::this_thread::sleep_until(next);
            const auto now = clk::now();
            const double lateness = std::chrono::duration<double, std::milli>(now - next).count();
            if (lateness > 1e3 * im->buf / im->sr) {            // atrasou um bloco inteiro = falha de áudio
                im->late.fetch_add(1);
                next = now;
            }
            if (lateness > im->late_max_ms.load()) im->late_max_ms.store(lateness);
            next += period;
        }
    });
    return "";
}

void AudioDevice::close() {
    if (impl_->run.exchange(false) && impl_->th.joinable()) impl_->th.join();
}
bool AudioDevice::running() const { return impl_->run.load(); }
double AudioDevice::latency_ms() const { return 1e3 * impl_->buf / impl_->sr; }
int AudioDevice::buffer_size() const { return impl_->buf; }
double AudioDevice::sample_rate() const { return impl_->sr; }
int AudioDevice::xruns() const { return impl_->late.load(); }
std::string AudioDevice::description() const { return impl_->desc; }
std::string AudioDevice::last_status() const {
    char b[96];
    std::snprintf(b, sizeof b, "atraso máx do despertar %.2f ms", impl_->late_max_ms.load());
    return b;
}

}  // namespace lt
