// TotK's load-dependent detail (game 1.2.1 only): the game lowers model and texture detail when its frame rate drops,
// through a quality level handed to the graphics settings setup (main+0xc7eabc, the site of patches-v121.txt's
// `steady` set) by a thin wrapper (main+0xcc3ff4) that the game reaches by an indirect call. On the CM5 the same place
// drew ~3,000 objects a frame after starting there and ~4,000 after walking there, which this is meant to explain.
//
// TOTK_LOD_LOG=1: logs every change of the level passed to the wrapper, with the caller and the frame.
// TOTK_LOD_LEVEL=<n>: passes n instead of the game's level (implies the log).
// Both are read from the environment at start (before the settings file), so they go in the launcher script.
#include <stdint.h>
#include <stdlib.h>

#include "cpu.h"
#include "dispatch.h"
#include "host.h"

#define LOD_SET_LEVEL 0x7000cc3ff4ull   // main+0xcc3ff4 in 1.2.1: w2 &= 0xffffffff, then main+0xc7eabc

static GuestFn lod_orig;
static int lod_force = -1;

static void lod_set_level(Cpu *c) {
    static uint32_t last = UINT32_MAX;
    static unsigned logged;
    uint32_t level = (uint32_t)c->x[2];
    if (level != last && logged < 500) {
        logged++;
        rt_log("lod: quality level %u from lr %#llx (object %#llx) at frame %llu%s", level,
               (unsigned long long)c->x[30], (unsigned long long)c->x[0], (unsigned long long)perf_frame_no(),
               lod_force >= 0 ? "; passed as TOTK_LOD_LEVEL" : "");
        last = level;
    }
    if (lod_force >= 0) c->x[2] = (uint32_t)lod_force;
    lod_orig(c);
}

__attribute__((constructor)) static void lod_register(void) {
    const char *lv = getenv("TOTK_LOD_LEVEL"), *lg = getenv("TOTK_LOD_LOG");
    if (lv && *lv) lod_force = atoi(lv);
    if (lod_force < 0 && !(lg && *lg && atoi(lg))) return;
    rt_dispatch_override(LOD_SET_LEVEL, lod_set_level, &lod_orig);
}
