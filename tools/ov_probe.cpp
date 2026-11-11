// ============================================================================
//  ov_probe — can this build reach OpenVINO's devices from OUR code?
//
//  Why the C API and not the C++ one: this project is built with mingw g++,
//  while the OpenVINO runtime here is an MSVC build. Passing std::string,
//  std::vector and exceptions across that ABI boundary is undefined, whereas
//  openvino/c/openvino.h is a C surface with a fixed layout and an opaque
//  handle, which is exactly what it exists for.
//
//  The question it answers is narrow and decisive: does the installed SDK
//  (headers + import lib + runtime + device plugins) link and enumerate the
//  CPU, the Xe iGPU and the NPU. Everything else about adopting it follows
//  from that answer; nothing follows if this does not run.
//
//  Build (paths are this box's; the SDK lives under oneAPI rather than a
//  standalone install, and there is no CMake target for it in the repo yet):
//
//      g++ -O2 -std=c++17 tools/ov_probe.cpp ^
//          -I"C:\Program Files (x86)\Intel\oneAPI\runtime\include" ^
//          "C:\Program Files (x86)\Intel\oneAPI\runtime\lib\intel64\Release\openvino_c.lib" ^
//          -o ov_probe.exe
//
//  and run with
//      C:\Program Files (x86)\Intel\oneAPI\runtime\bin\intel64\Release
//  on PATH, or the loader reports an empty error and every call returns
//  OV_STATUS_GENERAL_ERROR -- the same silent-miss shape as the ROCm oracle.
// ============================================================================
#include <openvino/c/openvino.h>

#include <cstdio>

int main(void) {
    ov_version_t ver;
    if (ov_get_openvino_version(&ver) == 0) {
        std::printf("OpenVINO build %s\n", ver.buildNumber ? ver.buildNumber : "?");
        if (ver.description) std::printf("  %s\n", ver.description);
        ov_version_free(&ver);
    } else {
        std::printf("ov_get_openvino_version FAILED (runtime not loaded?)\n");
        return 1;
    }

    ov_core_t *core = nullptr;
    ov_status_e st = ov_core_create(&core);
    if (st != 0 || core == nullptr) {
        std::printf("ov_core_create failed, status %d\n", static_cast<int>(st));
        return 1;
    }

    ov_available_devices_t devs;
    st = ov_core_get_available_devices(core, &devs);
    std::printf("devices (status %d):\n", static_cast<int>(st));
    for (size_t i = 0; i < devs.size; i++)
        std::printf("  [%zu] %s\n", i, devs.devices[i]);
    ov_available_devices_free(&devs);

    // The device name string is not enough for a decision: the GPU is behind
    // the same memory controller as the CPU here (unified memory), so print
    // what each device calls itself and whether the plugin answers at all.
    const char *want[] = {"CPU", "GPU", "NPU"};
    for (const char *d : want) {
        char *name = nullptr;
        const ov_status_e s2 = ov_core_get_property(
            core, d, "SUPPORTED_PROPERTIES", &name);
        std::printf("  %-4s property query status %d%s\n", d, static_cast<int>(s2),
                    s2 == 0 ? " (plugin present)" : " (no plugin)");
        if (name) ov_free(name);
    }

    ov_core_free(core);
    return 0;
}
