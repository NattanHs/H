// Saída de áudio pelo JUCE. A thread de áudio é do próprio driver (DirectSound / WASAPI / ASIO): ela chama
// Core::render() direto, sem passar pelo Python.
#include "device.h"

#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_events/juce_events.h>

#include <algorithm>
#include <cmath>
#include <memory>

namespace lt {

namespace {

void ensure_juce() {
    // O JUCE precisa de um MessageManager vivo. Criado uma vez e nunca destruído (vive até o fim do processo).
    static juce::ScopedJuceInitialiser_GUI* init = new juce::ScopedJuceInitialiser_GUI();
    (void) init;
}

const char* const kNames[] = {"DirectSound", "Windows Audio", "Windows Audio (Exclusive Mode)",
                              "Windows Audio (Low Latency Mode)", "ASIO", "ALSA"};

std::unique_ptr<juce::AudioIODeviceType> make_type(const std::string& name) {
    juce::AudioIODeviceType* t = nullptr;
    if (name == "DirectSound") t = juce::AudioIODeviceType::createAudioIODeviceType_DirectSound();
    else if (name == "Windows Audio") t = juce::AudioIODeviceType::createAudioIODeviceType_WASAPI(juce::WASAPIDeviceMode::shared);
    else if (name == "Windows Audio (Exclusive Mode)")
        t = juce::AudioIODeviceType::createAudioIODeviceType_WASAPI(juce::WASAPIDeviceMode::exclusive);
    else if (name == "Windows Audio (Low Latency Mode)")
        t = juce::AudioIODeviceType::createAudioIODeviceType_WASAPI(juce::WASAPIDeviceMode::sharedLowLatency);
    else if (name == "ASIO") t = juce::AudioIODeviceType::createAudioIODeviceType_ASIO();
    else if (name == "ALSA") t = juce::AudioIODeviceType::createAudioIODeviceType_ALSA();
    return std::unique_ptr<juce::AudioIODeviceType>(t);
}

class Callback : public juce::AudioIODeviceCallback {
public:
    Core* core = nullptr;
    std::vector<float> scratch;
    std::atomic<int> errors{0};
    juce::String last_error;

    void audioDeviceAboutToStart(juce::AudioIODevice* d) override {
        const int n = std::max(16384, d != nullptr ? d->getCurrentBufferSizeSamples() * 2 : 0);
        scratch.assign(static_cast<size_t>(n) * 2, 0.0f);          // aqui pode alocar: ainda não está no tempo real
    }
    void audioDeviceStopped() override {}
    void audioDeviceError(const juce::String& message) override {
        last_error = message;
        errors.fetch_add(1);
    }

    void audioDeviceIOCallbackWithContext(const float* const*, int, float* const* out, int num_out, int n,
                                          const juce::AudioIODeviceCallbackContext&) override {
        juce::ScopedNoDenormals no_denormals;
        const int cap = static_cast<int>(scratch.size() / 2);
        int done = 0;
        while (done < n) {
            const int m = std::min(cap, n - done);
            core->render(scratch.data(), m);
            const float* s = scratch.data();
            if (num_out >= 2) {
                float* l = out[0] + done;
                float* r = out[1] + done;
                for (int i = 0; i < m; ++i) {
                    l[i] = s[2 * i];
                    r[i] = s[2 * i + 1];
                }
            } else if (num_out == 1) {
                float* l = out[0] + done;
                for (int i = 0; i < m; ++i) l[i] = 0.5f * (s[2 * i] + s[2 * i + 1]);
            }
            done += m;
        }
        for (int c = 2; c < num_out; ++c) std::fill(out[c], out[c] + n, 0.0f);
    }
};

}  // namespace

struct AudioDevice::Impl {
    explicit Impl(Core& c) : core(c) {}
    Core& core;
    std::unique_ptr<juce::AudioIODeviceType> type;
    std::unique_ptr<juce::AudioIODevice> dev;
    Callback cb;
    std::string desc;
    double rate = 0.0;
};

AudioDevice::AudioDevice(Core& core) : impl_(new Impl(core)) { ensure_juce(); }

AudioDevice::~AudioDevice() {
    close();
    delete impl_;
}

bool AudioDevice::available() { return true; }

std::vector<std::string> AudioDevice::types() {
    ensure_juce();
    std::vector<std::string> out;
    for (const char* n : kNames)
        if (make_type(n)) out.emplace_back(n);
    return out;
}

std::vector<DeviceInfo> AudioDevice::list() {
    ensure_juce();
    std::vector<DeviceInfo> out;
    for (const char* n : kNames) {
        auto t = make_type(n);
        if (!t) continue;
        t->scanForDevices();
        const auto names = t->getDeviceNames(false);
        for (const auto& name : names) {
            DeviceInfo di;
            di.type = n;
            di.name = name.toStdString();
            di.out_channels = 2;
            di.default_rate = 0.0;
            out.push_back(di);
        }
    }
    return out;
}

std::string AudioDevice::open(const std::string& type, const std::string& name, int sample_rate, int buffer_size) {
    close();
    ensure_juce();
    auto t = make_type(type);
    if (!t) return "Tipo de driver indisponível nesta compilação: " + type;
    t->scanForDevices();
    const auto names = t->getDeviceNames(false);
    if (names.isEmpty()) return "Nenhum dispositivo de saída em " + type + ".";
    juce::String wanted = name.empty() ? names[std::max(0, t->getDefaultDeviceIndex(false))] : juce::String(name);
    if (!names.contains(wanted)) {
        std::string lista;
        for (const auto& s : names) lista += "\n  - " + s.toStdString();
        return "Dispositivo '" + name + "' não encontrado em " + type + ". Disponíveis:" + lista;
    }
    std::unique_ptr<juce::AudioIODevice> d(t->createDevice(wanted, juce::String()));
    if (!d) return "Não foi possível criar o dispositivo '" + wanted.toStdString() + "' (" + type + ").";

    bool rate_ok = false;
    std::string rates;
    for (double r : d->getAvailableSampleRates()) {
        if (std::fabs(r - sample_rate) < 1.0) rate_ok = true;
        rates += " " + std::to_string(static_cast<int>(r));
    }
    if (!rate_ok)
        return "O dispositivo '" + wanted.toStdString() + "' não aceita " + std::to_string(sample_rate) +
               " Hz. Taxas aceitas:" + rates + ". Ajuste a taxa do app ou a do Windows (Som > Propriedades > Avançado).";

    juce::BigInteger in_ch, out_ch;
    out_ch.setRange(0, 2, true);
    const int bs = buffer_size > 0 ? buffer_size : d->getDefaultBufferSize();
    const juce::String err = d->open(in_ch, out_ch, sample_rate, bs);
    if (err.isNotEmpty()) return err.toStdString();
    if (std::fabs(d->getCurrentSampleRate() - sample_rate) > 1.0) {
        d->close();
        return "O dispositivo abriu a " + std::to_string(static_cast<int>(d->getCurrentSampleRate())) + " Hz, e o app usa " +
               std::to_string(sample_rate) + " Hz.";
    }
    impl_->cb.core = &impl_->core;
    impl_->cb.errors = 0;
    impl_->type = std::move(t);
    impl_->dev = std::move(d);
    impl_->rate = sample_rate;
    impl_->dev->start(&impl_->cb);
    if (!impl_->dev->isPlaying()) {
        const std::string why = impl_->dev->getLastError().toStdString();
        impl_->dev->close();
        impl_->dev.reset();
        impl_->type.reset();
        return "O dispositivo não iniciou. " + why;
    }
    impl_->desc = type + " · " + wanted.toStdString() + " · " + std::to_string(sample_rate) + " Hz · bloco " +
                  std::to_string(impl_->dev->getCurrentBufferSizeSamples());
    return "";
}

void AudioDevice::close() {
    if (impl_->dev) {
        impl_->dev->stop();
        impl_->dev->close();
        impl_->dev.reset();
    }
    impl_->type.reset();
}

bool AudioDevice::running() const { return impl_->dev && impl_->dev->isPlaying(); }

double AudioDevice::latency_ms() const {
    if (!impl_->dev || impl_->rate <= 0.0) return 0.0;
    return 1000.0 * impl_->dev->getOutputLatencyInSamples() / impl_->rate;
}

int AudioDevice::buffer_size() const { return impl_->dev ? impl_->dev->getCurrentBufferSizeSamples() : 0; }
double AudioDevice::sample_rate() const { return impl_->dev ? impl_->dev->getCurrentSampleRate() : 0.0; }
int AudioDevice::xruns() const { return impl_->dev ? impl_->dev->getXRunCount() : 0; }
std::string AudioDevice::description() const { return impl_->desc; }

std::string AudioDevice::last_status() const {
    std::string s = impl_->cb.errors.load() ? ("erros do driver: " + std::to_string(impl_->cb.errors.load())) : std::string();
    if (impl_->dev) {
        const auto e = impl_->dev->getLastError();
        if (e.isNotEmpty()) s += (s.empty() ? "" : " · ") + e.toStdString();
    }
    return s;
}

}  // namespace lt
