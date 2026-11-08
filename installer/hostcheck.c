// Host check of the installer's first step, compiled with the bundled clang at install time: does the CPU run
// -march=x86-64-v3 code (or only x86-64-v2), or is it an aarch64 CPU (the build is armv8.2-a there, tools/build_game.py),
// and does a Vulkan 1.3 device exist. Prints one JSON object on stdout.
// The Vulkan loader is opened at run time, so a machine without one gets "loader": false instead of a failed start.
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#define HOST_X86 1
#else
#define HOST_X86 0
#endif
#include <stdio.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <string.h>
#include <vulkan/vulkan.h>
#ifdef _WIN32
#include <windows.h>
#define open_lib() LoadLibraryA("vulkan-1.dll")
#define lib_sym(l, n) ((void *)GetProcAddress((l), (n)))
#else
#include <dlfcn.h>
#define open_lib() dlopen("libvulkan.so.1", RTLD_NOW)
#define lib_sym(l, n) dlsym((l), (n))
#endif

// The JSON goes to `out`. Arm's Mali driver prints its version ("arm_release_ver: ...") on stdout when it loads,
// which would land inside the JSON: off Windows, stdout is pointed at stderr while the Vulkan calls run and the JSON
// goes to a copy of the original stdout.
static FILE *out;
#define out_c(ch) putc((ch), out)

static void json_string(const char *s) {
    out_c('"');
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') fprintf(out, "\\%c", *s);
        else if ((unsigned char)*s < 0x20) fprintf(out, "\\u%04x", *s);
        else out_c(*s);
    }
    out_c('"');
}

// The x86-64-v2 level from CPUID: SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CMPXCHG16B and LAHF/SAHF. Every CPU with
// SSE4.2 has the others, but the level is what the v2 build is compiled for, so check them all.
static int x86_64_v2(void) {
#if !HOST_X86
    return 0;
#else
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return 0;
    const unsigned sse3 = 1u << 0, ssse3 = 1u << 9, cx16 = 1u << 13, sse41 = 1u << 19, sse42 = 1u << 20,
                   popcnt = 1u << 23;
    if ((c & (sse3 | ssse3 | cx16 | sse41 | sse42 | popcnt)) != (sse3 | ssse3 | cx16 | sse41 | sse42 | popcnt))
        return 0;
    if (!__get_cpuid(0x80000001, &a, &b, &c, &d)) return 0;
    return (c & (1u << 0)) != 0;   // LAHF/SAHF
#endif
}

// The x86-64-v3 level from CPUID: AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT and MOVBE, with the OS saving the YMM state.
// Not __builtin_cpu_supports("x86-64-v3"): clang reads that from the C runtime's CPU model, and libgcc before GCC 12
// (the Linux release's sysroot has GCC 10) has no level bits, so it answers false on every CPU.
static int x86_64_v3(void) {
#if !HOST_X86
    return 0;
#else
    if (!x86_64_v2()) return 0;
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return 0;
    const unsigned fma = 1u << 12, movbe = 1u << 22, osxsave = 1u << 27, avx = 1u << 28, f16c = 1u << 29;
    if ((c & (fma | movbe | osxsave | avx | f16c)) != (fma | movbe | osxsave | avx | f16c)) return 0;
    unsigned lo, hi;
    __asm__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    if ((lo & 6) != 6) return 0;   // XMM and YMM state enabled by the OS
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return 0;
    const unsigned bmi1 = 1u << 3, avx2 = 1u << 5, bmi2 = 1u << 8;
    if ((b & (bmi1 | avx2 | bmi2)) != (bmi1 | avx2 | bmi2)) return 0;
    if (!__get_cpuid(0x80000001, &a, &b, &c, &d)) return 0;
    return (c & (1u << 5)) != 0;   // LZCNT (ABM)
#endif
}

int main(void) {
    out = stdout;
#ifndef _WIN32
    int keep = dup(1);
    FILE *f = keep >= 0 ? fdopen(keep, "w") : NULL;
    if (f) {
        fflush(stdout);
        dup2(2, 1);
        out = f;
    }
#endif
#ifdef __aarch64__
    const char *arm64 = "true";
#else
    const char *arm64 = "false";
#endif
    fprintf(out, "{\"x86_64_v2\": %s, \"x86_64_v3\": %s, \"aarch64\": %s, ", x86_64_v2() ? "true" : "false",
           x86_64_v3() ? "true" : "false", arm64);
    void *lib = open_lib();
    PFN_vkGetInstanceProcAddr gipa = lib ? (PFN_vkGetInstanceProcAddr)lib_sym(lib, "vkGetInstanceProcAddr") : NULL;
    if (!gipa) {
        fprintf(out, "\"loader\": false, \"devices\": []}\n");
        return 0;
    }
    fprintf(out, "\"loader\": true, ");
    PFN_vkCreateInstance create = (PFN_vkCreateInstance)gipa(NULL, "vkCreateInstance");
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "totk-install",
                             .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst = VK_NULL_HANDLE;
    VkResult r = create ? create(&ci, NULL, &inst) : VK_ERROR_INITIALIZATION_FAILED;
    if (r != VK_SUCCESS) {
        fprintf(out, "\"instance\": %d, \"devices\": []}\n", (int)r);
        return 0;
    }
    PFN_vkEnumeratePhysicalDevices enumerate = (PFN_vkEnumeratePhysicalDevices)gipa(inst, "vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceProperties props = (PFN_vkGetPhysicalDeviceProperties)gipa(inst, "vkGetPhysicalDeviceProperties");
    PFN_vkDestroyInstance destroy = (PFN_vkDestroyInstance)gipa(inst, "vkDestroyInstance");
    VkPhysicalDevice devs[16];
    uint32_t n = 16;
    if (enumerate(inst, &n, devs) < 0) n = 0;
    fprintf(out, "\"instance\": 0, \"devices\": [");
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        props(devs[i], &p);
        fprintf(out, "%s{\"name\": ", i ? ", " : "");
        json_string(p.deviceName);
        fprintf(out, ", \"api\": [%u, %u, %u], \"type\": %d}", VK_API_VERSION_MAJOR(p.apiVersion),
               VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion), (int)p.deviceType);
    }
    fprintf(out, "]}\n");
    destroy(inst, NULL);
    return 0;
}
