// Ligação Python (pybind11): módulo `_native` do LagoTracks.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <unordered_map>

#include "core.h"
#include "device.h"
#include "ext_ring.h"

namespace py = pybind11;
using namespace lt;

namespace {

const std::unordered_map<std::string, Param>& param_names() {
    static const std::unordered_map<std::string, Param> m = {
        {"master", P_MASTER}, {"bpm", P_BPM}, {"grid_offset", P_GRID_OFFSET}, {"beats_per_bar", P_BPB},
        {"quantize", P_QUANTIZE}, {"click_mode", P_CLICK_MODE}, {"click_volume", P_CLICK_VOLUME},
        {"click_route", P_CLICK_ROUTE}, {"pad_mode", P_PAD_MODE}, {"pad_volume", P_PAD_VOLUME},
        {"pad_route", P_PAD_ROUTE}, {"pad_under", P_PAD_UNDER}, {"pad_fade_out", P_PAD_FADE_OUT},
        {"limiter_on", P_LIMITER_ON}, {"ministry_on", P_MINISTRY_ON}, {"ambient_silenced", P_AMBIENT_SILENCED}};
    return m;
}

Param param_of(const std::string& name) {
    auto it = param_names().find(name);
    if (it == param_names().end()) throw std::invalid_argument("parâmetro desconhecido: " + name);
    return it->second;
}

// Segura o array do numpy; o objeto Python só é solto com a GIL.
std::shared_ptr<Samples> make_samples(const py::object& obj) {
    if (obj.is_none()) return nullptr;
    using Arr = py::array_t<int16_t, py::array::c_style | py::array::forcecast>;
    Arr arr = Arr::ensure(obj);
    if (!arr) throw std::invalid_argument("o áudio precisa ser um array int16");
    auto s = std::make_shared<Samples>();
    if (arr.ndim() == 1) {
        s->channels = 1;
    } else if (arr.ndim() == 2 && (arr.shape(1) == 1 || arr.shape(1) == 2)) {
        s->channels = static_cast<int>(arr.shape(1));
    } else {
        throw std::invalid_argument("formato de áudio não suportado (use (n,) ou (n,2))");
    }
    s->frames = arr.shape(0);
    s->p = arr.data();
    s->owner = std::shared_ptr<void>(new py::object(arr), [](void* p) {
        py::gil_scoped_acquire g;
        delete static_cast<py::object*>(p);
    });
    return s;
}

// Provedor que chama o Python dentro da thread de quem renderiza (para testes offline: resultado idêntico ao motor Python).
class PyProvider : public ExtProvider {
public:
    PyProvider(py::object pad_render, py::object bed_active, py::object bed_render)
        : pad_(std::move(pad_render)), active_(std::move(bed_active)), bed_(std::move(bed_render)) {}
    ~PyProvider() override {
        py::gil_scoped_acquire g;
        pad_ = py::object();
        active_ = py::object();
        bed_ = py::object();
    }
    void pad_render(bool instant, int n, float* l, float* r) override {
        py::gil_scoped_acquire g;
        py::tuple t = pad_(instant, n);
        copy(t[0], l, n);
        copy(t[1], r, n);
    }
    bool bed_active() override {
        py::gil_scoped_acquire g;
        return py::cast<bool>(active_());
    }
    void bed_render(int n, bool want, float vol, bool mono, float* ml, float* mr, float* mc) override {
        py::gil_scoped_acquire g;
        py::tuple t = bed_(n, want, vol, mono);
        copy(t[0], ml, n);
        copy(t[1], mr, n);
        copy(t[2], mc, n);
    }

private:
    static void copy(const py::handle& h, float* dst, int n) {
        auto a = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(h);
        if (!a || a.size() < n) throw std::runtime_error("o provedor devolveu bloco curto");
        std::memcpy(dst, a.data(), sizeof(float) * n);
    }
    py::object pad_, active_, bed_;
};

// Ponte para o JUCE: o dispositivo roda sempre na thread do próprio JUCE; aqui só ficam as chamadas da interface.
}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Motor de áudio nativo do LagoTracks";
    m.def("build_info", []() {
        std::string s = "lagotracks-native 1.0";
        s += AudioDevice::available() ? " +JUCE" : " (sem JUCE)";
        return s;
    });

    py::class_<TrackState, std::shared_ptr<TrackState>>(m, "NTrack")
        .def_property("volume", [](TrackState& t) { return t.volume.load(); }, [](TrackState& t, float v) { t.volume.store(v); })
        .def_property("level", [](TrackState& t) { return t.level.load(); }, [](TrackState& t, float v) { t.level.store(v); })
        .def_property("mute", [](TrackState& t) { return t.mute.load(); }, [](TrackState& t, bool v) { t.mute.store(v); })
        .def_property("solo", [](TrackState& t) { return t.solo.load(); }, [](TrackState& t, bool v) { t.solo.store(v); })
        .def_property("melodic", [](TrackState& t) { return t.melodic.load(); }, [](TrackState& t, bool v) { t.melodic.store(v); })
        .def_property("route", [](TrackState& t) { return t.route.load(); }, [](TrackState& t, int v) { t.route.store(v); })
        .def_property("role", [](TrackState& t) { return t.role.load(); }, [](TrackState& t, int v) { t.role.store(v); })
        .def_property("peak", [](TrackState& t) { return t.peak.load(); }, [](TrackState& t, float v) { t.peak.store(v); })
        .def("request_reset_gain", [](TrackState& t) { t.reset_gain.store(true); });

    py::class_<ExtProvider, std::shared_ptr<ExtProvider>>(m, "ExtProvider");

    py::class_<PyProvider, ExtProvider, std::shared_ptr<PyProvider>>(m, "PyProvider")
        .def(py::init<py::object, py::object, py::object>(), py::arg("pad_render"), py::arg("bed_active"), py::arg("bed_render"));

    py::class_<RingProvider, ExtProvider, std::shared_ptr<RingProvider>>(m, "RingProvider")
        .def(py::init<>())
        .def("pad_needed", &RingProvider::pad_needed)
        .def("pad_space", &RingProvider::pad_space)
        .def("pad_fill", &RingProvider::pad_fill)
        .def("pad_instant", &RingProvider::pad_instant)
        .def("pad_write", [](RingProvider& p, py::array_t<float, py::array::c_style | py::array::forcecast> l,
                              py::array_t<float, py::array::c_style | py::array::forcecast> r) {
            const int n = static_cast<int>(std::min(l.size(), r.size()));
            if (n > p.pad_space()) throw std::runtime_error("fila do pad cheia");
            p.pad_write(l.data(), r.data(), n);
        })
        .def("bed_needed", &RingProvider::bed_needed)
        .def("bed_space", &RingProvider::bed_space)
        .def("bed_fill", &RingProvider::bed_fill)
        .def("bed_want", &RingProvider::bed_want)
        .def("bed_click_volume", &RingProvider::bed_click_volume)
        .def("bed_mono", &RingProvider::bed_mono)
        .def("bed_write", [](RingProvider& p, py::array_t<float, py::array::c_style | py::array::forcecast> ml,
                              py::array_t<float, py::array::c_style | py::array::forcecast> mr,
                              py::array_t<float, py::array::c_style | py::array::forcecast> mc, bool active) {
            const int n = static_cast<int>(std::min({ml.size(), mr.size(), mc.size()}));
            if (n > p.bed_space()) throw std::runtime_error("fila da cama cheia");
            std::vector<float> flag(n, active ? 1.0f : 0.0f);
            p.bed_write(ml.data(), mr.data(), mc.data(), flag.data(), n);
        })
        .def_property_readonly("pad_underruns", &RingProvider::pad_underruns)
        .def_property_readonly("bed_underruns", &RingProvider::bed_underruns);

    py::class_<Core>(m, "Core")
        .def(py::init<int>(), py::arg("sample_rate"))
        .def("set", [](Core& c, const std::string& name, double v) { c.set_param(param_of(name), v); })
        .def("get", [](Core& c, const std::string& name) { return c.get_param(param_of(name)); })
        .def("set_click_waves", [](Core& c, py::array_t<float, py::array::c_style | py::array::forcecast> a,
                                    py::array_t<float, py::array::c_style | py::array::forcecast> n) {
            c.set_click_waves(a.data(), static_cast<int>(a.size()), n.data(), static_cast<int>(n.size()));
        })
        .def("set_provider", [](Core& c, std::shared_ptr<ExtProvider> p) { c.set_provider(std::move(p)); })
        .def("make_track", &Core::make_track)
        .def("set_data", [](Core& c, const std::shared_ptr<TrackState>& t, const py::object& arr) {
            c.track_set_data(t, make_samples(arr));
        })
        .def("set_shifted", [](Core& c, const std::shared_ptr<TrackState>& t, const py::object& arr) {
            c.track_set_shifted(t, make_samples(arr));
        })
        .def("set_tracks", &Core::set_tracks, py::arg("tracks"), py::arg("reset") = true)
        .def("add_track", &Core::add_track)
        .def("remove_track", &Core::remove_track)
        .def("clear_tracks", &Core::clear_tracks)
        .def("collect", &Core::collect)
        .def("play", &Core::play, py::arg("delay_frames") = 0)
        .def("cancel_start", &Core::cancel_start)
        .def("pause", &Core::pause)
        .def("stop", &Core::stop, py::arg("silence") = false)
        .def("seek", &Core::seek)
        .def("jump", &Core::jump)
        .def("cancel_jump", &Core::cancel_jump)
        .def("set_loop", &Core::set_loop)
        .def("clear_loop", &Core::clear_loop)
        .def_property_readonly("pending_jump", [](Core& c) -> py::object {
            const int64_t j = c.pending_jump();
            return j < 0 ? py::none().cast<py::object>() : py::int_(j).cast<py::object>();
        })
        .def_property_readonly("loop_region", [](Core& c) -> py::object {
            int64_t s = 0, e = 0;
            if (!c.loop_region(s, e)) return py::none();
            return py::make_tuple(s, e);
        })
        .def_property_readonly("position", &Core::position)
        .def_property_readonly("playing", &Core::playing)
        .def_property_readonly("armed", &Core::armed)
        .def_property_readonly("beat_in_bar", &Core::beat_in_bar)
        .def_property_readonly("beat_counter", &Core::beat_counter)
        .def_property_readonly("ended_count", &Core::ended_count)
        .def_property_readonly("peak_l", &Core::peak_l)
        .def_property_readonly("peak_r", &Core::peak_r)
        .def_property_readonly("duck", &Core::duck)
        .def_property_readonly("clock", &Core::clock)
        .def_property_readonly("total_frames", &Core::total_frames)
        .def_property_readonly("pad_gain", &Core::pad_gain)
        .def_property_readonly("last_render_us", &Core::last_render_us)
        .def_property_readonly("max_render_us", &Core::max_render_us)
        .def_property_readonly("slow_blocks", &Core::slow_blocks)
        .def_property_readonly("render_calls", &Core::render_calls)
        .def("reset_diagnostics", &Core::reset_diagnostics)
        .def("render", [](Core& c, py::array_t<float, py::array::c_style> out) {
            if (out.ndim() != 2 || out.shape(1) != 2) throw std::invalid_argument("out precisa ter shape (n, 2)");
            float* p = out.mutable_data();
            const int n = static_cast<int>(out.shape(0));
            py::gil_scoped_release rel;
            c.render(p, n);
        });

    py::class_<DeviceInfo>(m, "DeviceInfo")
        .def_readonly("type", &DeviceInfo::type)
        .def_readonly("name", &DeviceInfo::name)
        .def_readonly("out_channels", &DeviceInfo::out_channels)
        .def_readonly("default_rate", &DeviceInfo::default_rate);

    py::class_<AudioDevice>(m, "AudioDevice")
        .def(py::init<Core&>(), py::keep_alive<1, 2>())
        .def_static("available", &AudioDevice::available)
        .def_static("types", &AudioDevice::types)
        .def_static("list", &AudioDevice::list)
        .def("open", &AudioDevice::open, py::arg("type"), py::arg("name"), py::arg("sample_rate"), py::arg("buffer_size"),
             py::call_guard<py::gil_scoped_release>())
        .def("close", &AudioDevice::close, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("running", &AudioDevice::running)
        .def_property_readonly("latency_ms", &AudioDevice::latency_ms)
        .def_property_readonly("buffer_size", &AudioDevice::buffer_size)
        .def_property_readonly("sample_rate", &AudioDevice::sample_rate)
        .def_property_readonly("xruns", &AudioDevice::xruns)
        .def_property_readonly("description", &AudioDevice::description)
        .def_property_readonly("last_status", &AudioDevice::last_status);
}
