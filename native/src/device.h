// Saída de áudio nativa (JUCE). Sem JUCE (testes no Linux) existe uma versão de mentira que só diz "indisponível".
#pragma once
#include <string>
#include <vector>

#include "core.h"

namespace lt {

struct DeviceInfo {
    std::string type;        // "DirectSound", "Windows Audio", "Windows Audio (Exclusive Mode)", "ASIO"...
    std::string name;
    int out_channels = 0;
    double default_rate = 0.0;
};

class AudioDevice {
public:
    explicit AudioDevice(Core& core);
    ~AudioDevice();
    static bool available();
    static std::vector<std::string> types();
    static std::vector<DeviceInfo> list();
    // devolve "" se abriu, ou o texto do erro. `name` vazio = dispositivo padrão do tipo.
    std::string open(const std::string& type, const std::string& name, int sample_rate, int buffer_size);
    void close();
    bool running() const;
    double latency_ms() const;
    int buffer_size() const;
    double sample_rate() const;
    int xruns() const;
    std::string description() const;
    std::string last_status() const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace lt
