// Uniform blocks as of the submit call. NVN starts the GPU when nvnQueueSubmitCommands returns and the GPU reads the
// constants soon after; here the replay records for several milliseconds, and the GPU reads a pool's bytes when it
// runs the part that uses them (the VRAM mirror's sync copies run then, vk_memory.c "VRAM mirror"). Meanwhile the game,
// having handed the frame over, rewrites per-object uniform blocks on other threads with no fence: late draws read
// another object's matrix (phantom rocks, a flat far field; TotK 1.2.1's producer f_70008c6460, 14-24 ms after the
// submit). So at the start of a submission's recording every uniform range it binds is copied into an arena of its own,
// and the uniform bindings of the replay use the copy.
// - Only ranges of imported pools (memory the CPU writes) are copied, aligned out to 256 bytes (the largest
//   minUniformBufferOffsetAlignment) and at most 64 KiB from the bound address, overlapping ones merged.
// - A range the GPU wrote in a submission that has not completed is left out: its bytes reach guest memory only with
//   that submission's write-back, the mirror holds them until then.
// - A GPU write within the submission (UpdateUniformBuffer, a storage buffer a shader writes, a buffer copy or fill,
//   an image-to-buffer copy: vk_usnap_drop) drops the copies it touches from then on, so later draws read the result.
// - The arena goes back to a free list once the submission's fence is done (completion thread).
// The design follows MacNativeGames' TOTK_UBO_SNAPSHOT (games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m usnap_take,
// usnap_drop; MIT, Copyright (c) 2026 jhaizhou); the code is ours.
// The copy is taken where vk_exec_record starts: at the submit call when the replay runs on the submitting thread, later
// with TOTK_REPLAY_ASYNC. TOTK_UBO_SNAPSHOT=0 binds live memory as before; =ab alternates (TOTK_AB).
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "vk.h"
#include "platform/platform.h"

#define ALIGN 256u
#define MAX_RANGE (64u << 10)
#define ARENA_MIN (4u << 20)
#define CALL_DEPTH 16
// vk_usnap_update (TOTK_UB_INPLACE): room after the copies for new versions, and the largest region versioned
#define UPD_SLACK (4u << 20)
static uint32_t upd_max = 16u << 10;   // TOTK_UB_INPLACE_MAX_KB (default 16): the largest region versioned

typedef struct Arena {
    struct Arena *next;
    VkBuffer buf;
    VmaAllocation alloc;
    uint8_t *map;
    uint8_t *shadow;            // TOTK_UB_INPLACE: the same bytes in cached host memory, for reading old versions
    uint64_t size;
} Arena;

typedef struct { uint64_t lo, hi; } Rng;
typedef struct {
    uint64_t lo, hi, off;
    int dropped, found;          // found: bound since its last version
    int split;                   // a part of a region split at an update: its copy as the submission began is
    uint64_t blo, bhi, boff;     // [blo, bhi) at boff (the base a block with a window twin binds)
} Reg;
typedef struct { uint64_t lo, hi, fence; int ub; } Wrote;   // ub: an UpdateUniformBuffer copy
typedef struct { uint64_t lo, hi, fence; uint8_t *data; } PendUb;   // an update copy on the GPU timeline, its bytes

static struct {
    int mode;                   // rt_knob: 0 off, 1 on, 2 ab
    int on;                     // this submission has a snapshot
    uint64_t fence, gen;
    Arena *arena;
    Rng *col;                   // collected ranges
    int ncol, ccol;
    Reg *reg;                   // merged, sorted by lo, disjoint
    int nreg, creg;
    Rng *excl;                  // bound uniform ranges left out of the copy (refused, GPU write in flight), sorted
    uint8_t *excl_ub;           // per excl: left out because of an earlier submission's update copy in flight
    int nexcl, cexcl;
    Wrote *wr;                  // GPU writes of submissions not known to be complete
    int nwr, cwr;
    PendUb *pub;                // TOTK_UB_INPLACE: update copies of submissions not known to be complete, in order
    int npub, cpub;
    // since the last summary
    uint64_t subs, regions, bytes, copy_ns, hits, drops, inflight, refused, off_subs;
    uint64_t used;              // bytes of the arena in use (copies, then versions)
    int random;                 // TOTK_UB_INPLACE: arenas keep a cached shadow and room for new versions
    uint64_t upd_done, upd_bytes, upd_big, upd_none, upd_full, upd_direct, upd_dropped, upd_big_bytes, upd_live, upd_live_ub;
    uint64_t ex_ub;             // ranges left out as in flight because of an UpdateUniformBuffer copy of an earlier submission
    uint64_t in_ub, late;       // ranges copied with update copies in flight applied; guest writes redone after one
    uint64_t splits, joins, join_bytes, join_fail;   // big regions split at an update; bindings across a split
    uint64_t bases, peeks;      // bindings of a split block's base; window twin reads
} U;

static Arena *free_arenas;
static PlatMutex free_lock = PLAT_MUTEX_INIT;

static void *grow(void *p, int *cap, size_t elem) {
    int n = *cap ? *cap * 2 : 1024;
    void *q = realloc(p, (size_t)n * elem);
    if (!q) rt_fatal("vk_usnap: out of memory");
    *cap = n;
    return q;
}

static Arena *arena_get(uint64_t need) {
    plat_mutex_lock(&free_lock);
    Arena **pp = &free_arenas, *a = NULL;
    for (; *pp; pp = &(*pp)->next)
        if ((*pp)->size >= need) {
            a = *pp, *pp = a->next;
            break;
        }
    plat_mutex_unlock(&free_lock);
    if (a) return a;
    a = calloc(1, sizeof *a);
    if (!a) rt_fatal("vk_usnap: out of memory");
    a->size = need < ARENA_MIN ? ARENA_MIN : (need + (1u << 20) - 1) & ~(uint64_t)((1u << 20) - 1);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = a->size,
                             // transfer source: DLAA copies the camera block out (vk_dlaa.c mv_pass)
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
    // Device-local through the BAR when the device has it (the GPU reads constants on every draw), else host memory.
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                           VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    VkResult res = vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &a->buf, &a->alloc, &info);
    if (res != VK_SUCCESS) {
        rt_log("vk_usnap: arena of %llu bytes failed (%d): this submission binds live memory",
               (unsigned long long)a->size, (int)res);
        free(a);
        return NULL;
    }
    a->map = info.pMappedData;
    // TOTK_UB_INPLACE: a new version copies the previous one; reading it back from write-combined memory is slow, and
    // host-cached arena memory needs a flush per version (a driver call on the Mali): the old bytes come from here.
    if (U.random && !(a->shadow = malloc(a->size))) rt_fatal("vk_usnap: out of memory");
    rt_log("vk_usnap: new arena of %.1f MiB", a->size / 1048576.0);
    return a;
}

static void arena_done(void *arg) {   // completion thread: the GPU has read it
    Arena *a = arg;
    plat_mutex_lock(&free_lock);
    a->next = free_arenas, free_arenas = a;
    plat_mutex_unlock(&free_lock);
}

static void collect(const op_hdr *h, int depth) {
    for (;;) {
        if (h->size < sizeof *h || (h->size & 7) || h->op >= OP_COUNT_)
            rt_fatal("vk_usnap: corrupt command record at %p: op %u size %u", (const void *)h, h->op, h->size);
        const void *a = nvn_op_payload(h);
        if (h->op == OP_END) return;
        if (h->op == OP_JUMP) {
            h = (const op_hdr *)(uintptr_t)((const p_jump *)a)->target;
            continue;
        }
        if (h->op == OP_CALL && depth < CALL_DEPTH) collect(nvn_op_first(((const p_jump *)a)->target), depth + 1);
        if (h->op == OP_BIND_UBO) {
            const p_bind_buf *b = a;
            if (b->addr && b->size) {
                if (U.ncol == U.ccol) U.col = grow(U.col, &U.ccol, sizeof *U.col);
                uint64_t n = b->size < MAX_RANGE ? b->size : MAX_RANGE;
                U.col[U.ncol++] = (Rng){b->addr & ~(uint64_t)(ALIGN - 1), b->addr + n};
            }
        }
        // TOTK_UB_INPLACE: an update's range too, so that it lies wholly in one region (an update reaching past the
        // ranges bound around it fell back to the copy on the GPU timeline: ~140 a submission outdoors).
        if (h->op == OP_UPDATE_UB && U.random) {
            const p_update_ub *u = a;
            if (u->dst && u->size && u->size <= MAX_RANGE) {
                if (U.ncol == U.ccol) U.col = grow(U.col, &U.ccol, sizeof *U.col);
                U.col[U.ncol++] = (Rng){u->dst & ~(uint64_t)(ALIGN - 1), u->dst + u->size};
            }
        }
        h = nvn_op_next(h);
    }
}

static int by_lo(const void *a, const void *b) {
    uint64_t x = ((const Rng *)a)->lo, y = ((const Rng *)b)->lo;
    return x < y ? -1 : x > y;
}
static int wr_by_lo(const void *a, const void *b) {
    uint64_t x = ((const Wrote *)a)->lo, y = ((const Wrote *)b)->lo;
    return x < y ? -1 : x > y;
}

// Sorts the collected ranges by lo. A town frame binds about 10,000 uniform ranges and qsort took half a millisecond
// of the copy's 0.96 ms on the game thread, so they go through an LSD radix sort on (lo - lowest) / ALIGN, three
// passes of 11 bits; qsort is left for a span that does not fit in 33 bits (2 TiB).
#define RADIX_BITS 11
static void sort_col(void) {
    static Rng *tmp;
    static int ctmp;
    int n = U.ncol;
    if (n < 2) return;
    uint64_t lo = UINT64_MAX, hi = 0;
    for (int i = 0; i < n; i++) {
        if (U.col[i].lo < lo) lo = U.col[i].lo;
        if (U.col[i].lo > hi) hi = U.col[i].lo;
    }
    if ((hi - lo) / ALIGN >> (3 * RADIX_BITS)) {
        qsort(U.col, (size_t)n, sizeof *U.col, by_lo);
        return;
    }
    if (ctmp < n) {
        free(tmp);
        ctmp = U.ccol;
        if (!(tmp = malloc((size_t)ctmp * sizeof *tmp))) rt_fatal("vk_usnap: out of memory");
    }
    Rng *src = U.col, *dst = tmp;
    for (int pass = 0; pass < 3; pass++) {
        int shift = pass * RADIX_BITS, count[1 << RADIX_BITS] = {0};
        for (int i = 0; i < n; i++) count[(src[i].lo - lo) / ALIGN >> shift & ((1 << RADIX_BITS) - 1)]++;
        for (int k = 0, sum = 0; k < 1 << RADIX_BITS; k++) {
            int c = count[k];
            count[k] = sum, sum += c;
        }
        for (int i = 0; i < n; i++) dst[count[(src[i].lo - lo) / ALIGN >> shift & ((1 << RADIX_BITS) - 1)]++] = src[i];
        Rng *t = src;
        src = dst, dst = t;
    }
    if (src != U.col) memcpy(U.col, src, (size_t)n * sizeof *src);   // three passes end in tmp
}

// Keeps the GPU writes of submissions that are not done, merged into disjoint ranges sorted by lo (the terrain rewrites
// one block before every draw: hundreds of equal entries), each with the newest fence among its parts.
static void wr_prune(void) {
    uint64_t done = vk_sync_fence_done();
    int np = 0;
    for (int i = 0; i < U.npub; i++)
        if (U.pub[i].fence > done) U.pub[np++] = U.pub[i];
        else free(U.pub[i].data);
    U.npub = np;
    int n = 0;
    for (int i = 0; i < U.nwr; i++)
        if (U.wr[i].fence > done) U.wr[n++] = U.wr[i];
    qsort(U.wr, (size_t)n, sizeof *U.wr, wr_by_lo);
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (m && U.wr[i].lo <= U.wr[m - 1].hi) {
            if (U.wr[i].hi > U.wr[m - 1].hi) U.wr[m - 1].hi = U.wr[i].hi;
            if (U.wr[i].fence > U.wr[m - 1].fence) U.wr[m - 1].fence = U.wr[i].fence;
            U.wr[m - 1].ub &= U.wr[i].ub;
        } else {
            U.wr[m++] = U.wr[i];
        }
    }
    U.nwr = m;
}
static int wr_hits(uint64_t lo, uint64_t hi) {   // wr_prune ran: disjoint, sorted. 2: only update copies
    int a = 0, b = U.nwr;
    while (a < b) {
        int k = (a + b) / 2;
        if (U.wr[k].hi <= lo) a = k + 1;
        else b = k;
    }
    if (!(a < U.nwr && U.wr[a].lo < hi)) return 0;
    for (int k = a; k < U.nwr && U.wr[k].lo < hi; k++)
        if (!U.wr[k].ub) return 1;
    return 2;
}

static void summary(void) {
    if (U.subs + U.off_subs < (U.random ? 600u : 3600u)) return;
    double s = U.subs ? (double)U.subs : 1.0;
    rt_log("vk_usnap: %llu submissions with a snapshot, %llu without; per submission %.0f ranges, %.1f KiB copied in "
           "%.3f ms, %.0f uniform bindings from the copy, %.1f dropped by GPU writes, %.1f left out (GPU writes in "
           "flight), %.1f refused (not an imported pool)",
           (unsigned long long)U.subs, (unsigned long long)U.off_subs, U.regions / s, U.bytes / s / 1024.0,
           U.copy_ns / s / 1e6, U.hits / s, U.drops / s, U.inflight / s, U.refused / s);
    if (U.random)
        rt_log("vk_usnap: UpdateUniformBuffer in the snapshot (TOTK_UB_INPLACE), per submission: %.1f versions "
               "(%.1f KiB copied), %.1f of a range no uniform binding of the submission touches (written to memory "
               "only), %.1f in a region over %u KiB (avg %.1f KiB), %.1f in a region dropped by a GPU write, %.1f bound "
               "live (left out of the copy; %.1f of them by an earlier submission's update copy in flight), %.1f "
               "partly in a region, %.1f with the arena full; ranges left out by an update copy in flight %.1f, copied "
               "with those copies applied %.1f, guest writes redone after one %.1f; big regions split %.1f, bindings "
               "across a split %.1f (%.1f KiB copied), not joined %.1f; split blocks bound as their base %.1f, window "
               "reads %.1f",
               U.upd_done / s, U.upd_bytes / s / 1024.0, U.upd_direct / s, U.upd_big / s, upd_max >> 10,
               U.upd_big ? U.upd_big_bytes / (double)U.upd_big / 1024.0 : 0.0, U.upd_dropped / s, U.upd_live / s,
               U.upd_live_ub / s, U.upd_none / s, U.upd_full / s, U.ex_ub / s, U.in_ub / s, U.late / s, U.splits / s,
               U.joins / s, U.join_bytes / s / 1024.0, U.join_fail / s, U.bases / s, U.peeks / s);
    U.subs = U.off_subs = U.regions = U.bytes = U.copy_ns = U.hits = U.drops = U.inflight = U.refused = 0;
    U.upd_done = U.upd_bytes = U.upd_big = U.upd_none = U.upd_full = U.upd_direct = U.upd_dropped = U.upd_big_bytes = 0;
    U.upd_live = U.upd_live_ub = U.ex_ub = U.in_ub = U.late = U.splits = U.joins = U.join_bytes = U.join_fail = 0;
    U.bases = U.peeks = 0;
}

static int first_above(uint64_t a);
static int join(int i, uint64_t addr, uint64_t n, uint64_t align, VkDescriptorBufferInfo *out);

static void ubc_summary(void);
void vk_usnap_begin(const uint64_t *handles, int n, uint64_t fence) {
    ubc_summary();
    if (!U.mode) {
        U.mode = rt_knob("UBO_SNAPSHOT", "1") + 1, U.random = vk_caps_tiler_opt("TOTK_UB_INPLACE");
        const char *m = cfg_get("TOTK_UB_INPLACE_MAX_KB");
        if (m && atoi(m) > 0) upd_max = (uint32_t)atoi(m) << 10;
    }
    U.on = 0, U.fence = fence, U.gen++;
    summary();
    if (!RT_KNOB_ON(U.mode - 1)) {
        U.off_subs++;
        return;
    }
    uint64_t t0 = kern_now_ns();
    U.ncol = 0;
    for (int i = 0; i < n; i++)
        if (handles[i]) collect(nvn_op_first(handles[i]), 0);
    sort_col();
    wr_prune();
    // merge, keeping only what lies wholly in one imported pool and no GPU write in flight touches
    U.nreg = U.nexcl = 0;
    uint64_t total = 0;
    for (int i = 0; i < U.ncol;) {
        uint64_t lo = U.col[i].lo, hi = U.col[i].hi;
        for (i++; i < U.ncol && U.col[i].lo <= hi; i++)
            if (U.col[i].hi > hi) hi = U.col[i].hi;
        VkMemSpan sp;
        int refused = !vk_mem_resolve(lo, hi - lo, &sp) || sp.alloc || !sp.pool || sp.avail < hi - lo ||
                      (sp.cls != VK_MCLS_HOST_CACHED && sp.cls != VK_MCLS_HOST_UNC);
        int wh = refused ? 0 : wr_hits(lo, hi);
        // Only update copies of earlier submissions in flight: the copy below applies their bytes (TOTK_UB_INPLACE).
        // Left out, every update into the range took a copy on the GPU timeline again, so the next submission left
        // it out too: on the Mali handheld outdoors ~150-180 updates a submission, each ending its pass.
        if (wh == 2 && U.random) wh = 0, U.in_ub++;
        if (refused || wh) {
            if (refused) U.refused++;
            else U.inflight++;
            if (U.nexcl == U.cexcl) {
                U.excl = grow(U.excl, &U.cexcl, sizeof *U.excl);
                if (!(U.excl_ub = realloc(U.excl_ub, (size_t)U.cexcl))) rt_fatal("vk_usnap: out of memory");
            }
            U.excl_ub[U.nexcl] = (uint8_t)(wh == 2);
            if (wh == 2) U.ex_ub++;
            U.excl[U.nexcl++] = (Rng){lo, hi};   // bound live: vk_usnap_update must not write it early
            continue;
        }
        if (U.nreg == U.creg) U.reg = grow(U.reg, &U.creg, sizeof *U.reg);
        U.reg[U.nreg++] = (Reg){lo, hi, total, 0, 0, 0, 0, 0, 0};
        total += (hi - lo + ALIGN - 1) & ~(uint64_t)(ALIGN - 1);
    }
    if (!U.nreg) return;
    if (!(U.arena = arena_get(total + (U.random ? UPD_SLACK : 0)))) return;
    U.used = total;
    for (int i = 0; i < U.nreg; i++)
        memcpy(U.arena->map + U.reg[i].off, (const void *)(uintptr_t)U.reg[i].lo, U.reg[i].hi - U.reg[i].lo);
    if (U.arena->shadow)
        for (int i = 0; i < U.nreg; i++)
            memcpy(U.arena->shadow + U.reg[i].off, (const void *)(uintptr_t)U.reg[i].lo, U.reg[i].hi - U.reg[i].lo);
    // The update copies still in flight, in submission order, over the bytes guest memory does not hold yet
    for (int k = 0; k < U.npub; k++) {
        const PendUb *p = &U.pub[k];
        for (int i = first_above(p->lo); i < U.nreg && U.reg[i].lo < p->hi; i++) {
            uint64_t a = p->lo > U.reg[i].lo ? p->lo : U.reg[i].lo, b = p->hi < U.reg[i].hi ? p->hi : U.reg[i].hi;
            memcpy(U.arena->map + U.reg[i].off + (a - U.reg[i].lo), p->data + (a - p->lo), b - a);
            if (U.arena->shadow) memcpy(U.arena->shadow + U.reg[i].off + (a - U.reg[i].lo), p->data + (a - p->lo), b - a);
        }
    }
    vmaFlushAllocation(vk_ctx()->vma, U.arena->alloc, 0, total);   // a no-op on coherent memory
    U.on = 1;
    U.subs++, U.regions += (uint64_t)U.nreg, U.bytes += total, U.copy_ns += kern_now_ns() - t0;
}

void vk_usnap_end(uint64_t fence) {
    if (U.arena) vk_completion_add(fence, arena_done, U.arena), U.arena = NULL;
    U.on = 0, U.gen++;
}

// First region whose end is above a.
static int first_above(uint64_t a) {
    int lo = 0, hi = U.nreg;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (U.reg[m].hi <= a) lo = m + 1;
        else hi = m;
    }
    return lo;
}

// TOTK_UB_SPLIT=1 (off): splitting big regions at their updates (below). Off since 0049's test: TotK's 64 KiB block is
// read by its shaders at computed offsets, so its bindings stay whole, every one of them needed a joined 64 KiB copy
// (~60 a submission, 3.9 MiB), the arena ran out, and bindings that could not be joined read live memory, which holds
// the newest update: the terrain flickered.
// With the window twins (TOTK_UB_WINDOW, gfx_shader.c ubo_window) it is on: the block's readers at computed offsets
// bind the base (vk_usnap_find base_ok), its window comes from a small per-draw buffer (vk_usnap_peek), and no join
// of 64 KiB is needed.
static int split_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("TOTK_UB_SPLIT");
        on = e && *e ? strcmp(e, "0") != 0 : vk_caps_tiler_opt("TOTK_UB_WINDOW");
    }
    return on;
}

// A binding across regions split from one (vk_usnap_update): a copy of its range joined from the regions' current
// versions. 0 when the regions there are not contiguous, one is dropped, or the arena is full.
static int join(int i, uint64_t addr, uint64_t n, uint64_t align, VkDescriptorBufferInfo *out) {
    uint64_t need = (n + ALIGN - 1) & ~(uint64_t)(ALIGN - 1), at = U.used, pos = addr;
    if (at % align || at + need > U.arena->size) { U.join_fail++; return 0; }
    for (int j = i; pos < addr + n; j++) {
        if (j >= U.nreg || U.reg[j].dropped || U.reg[j].lo > pos) { U.join_fail++; return 0; }
        uint64_t b = U.reg[j].hi < addr + n ? U.reg[j].hi : addr + n;
        memcpy(U.arena->shadow + at + (pos - addr), U.arena->shadow + U.reg[j].off + (pos - U.reg[j].lo), b - pos);
        pos = b;
    }
    for (int j = i; j < U.nreg && U.reg[j].lo < addr + n; j++) U.reg[j].found = 1;
    memcpy(U.arena->map + at, U.arena->shadow + at, n);
    U.used += need;
    U.joins++, U.join_bytes += n, U.hits++;
    *out = (VkDescriptorBufferInfo){U.arena->buf, at, n};
    return 1;
}

// TOTK_UB_INPLACE: the first updates into and bindings of regions over the version limit, logged once (how the game
// uses them decides how such a region can get versions).
static int big_logged;
static void big_note(const char *what, const Reg *r, uint64_t addr, uint64_t n) {
    if (big_logged >= 120) return;
    big_logged++;
    rt_log("vk_usnap: big region %#llx+%#llx (%s): %s at +%#llx, %#llx bytes", (unsigned long long)r->lo,
           (unsigned long long)(r->hi - r->lo), r->dropped ? "dropped" : "live copy", what,
           (unsigned long long)(addr - r->lo), (unsigned long long)n);
}

// The base of the split block whose head is region i may stand for [addr, addr+n): bound at the block's start, the
// head no larger than the twin (4 KiB), and the rest one region still at the copy as the submission began (no update
// reached it). Otherwise the draws read stale bytes there (TOTK_UB_CHECK found ~75 % of the terrain's draws so).
#define TWIN_BYTES 4096
static int base_ok_at(int i, uint64_t addr, uint64_t n, uint64_t align) {
    const Reg *h = &U.reg[i];
    if (!h->split || h->dropped || addr != h->blo || h->lo != h->blo || h->hi - h->lo > TWIN_BYTES) return 0;
    if (addr + n > h->bhi || addr + n <= h->hi || h->boff % align) return 0;
    if (i + 1 >= U.nreg) return 0;
    const Reg *t = &U.reg[i + 1];
    return !t->dropped && t->lo == h->hi && t->hi == h->bhi && t->off == h->boff + (t->lo - h->blo);
}

static const char *find_kind;   // TOTK_UB_CHECK: how vk_usnap_find answered last
static const Reg *find_reg;
int vk_usnap_find(uint64_t addr, uint64_t n, uint64_t align, VkDescriptorBufferInfo *out, int base_ok) {
    if (!U.on || !n) return 0;
    find_kind = "version", find_reg = NULL;
    int i = first_above(addr);
    if (i >= U.nreg) return 0;
    Reg *r = &U.reg[i];
    if (U.random && r->hi - r->lo > upd_max && addr >= r->lo && addr < r->hi) big_note("bind", r, addr, n);
    // A block with a window twin, bound at its start across a split: its copy as the submission began. The twin reads
    // the head (gfx_shader.c ubo_window: the first 4 KiB) from its current version; the rest must still be the copy.
    if (base_ok && base_ok_at(i, addr, n, align)) {
        *out = (VkDescriptorBufferInfo){U.arena->buf, r->boff + (addr - r->blo), n};
        U.hits++, U.bases++;
        find_kind = "base", find_reg = r;
        return 1;
    }
    if (!r->dropped && addr >= r->lo && addr + n > r->hi && U.arena && U.arena->shadow) {
        find_kind = "join", find_reg = r;
        if (join(i, addr, n, align, out)) return 1;
        // no room for the join: the base when it is still right (0049's flicker came from live memory)
        if (base_ok_at(i, addr, n, align)) {
            *out = (VkDescriptorBufferInfo){U.arena->buf, r->boff + (addr - r->blo), n};
            U.hits++, U.bases++;
            find_kind = "base (join failed)";
            return 1;
        }
        return 0;
    }
    if (r->dropped || addr < r->lo || addr + n > r->hi) return 0;
    uint64_t off = r->off + (addr - r->lo);
    if (off % align) return 0;
    r->found = 1;
    *out = (VkDescriptorBufferInfo){U.arena->buf, off, n};
    U.hits++;
    find_reg = r;
    return 1;
}

// TOTK_UB_CHECK=1 (diagnostic): the bytes a uniform binding from the snapshot gives the GPU against guest memory at
// replay time, which holds every update before the draw (op_update_ub writes them as it versions them). A base skips
// its window (the twin reads it). Differences are logged (the first 400): how the binding was made, the range, the
// first and last differing byte and the region. Updates still on the GPU timeline (earlier submissions' copies) and
// the game writing ahead also differ, so a few lines are expected; a stream of them at the same block is the bug.
static int ubc_on = -1;
enum { UBC_VERSION, UBC_JOIN, UBC_BASE, UBC_BASE_NOJOIN, UBC_TWIN, UBC_KINDS };
static const char *const ubc_name[UBC_KINDS] = {"version", "join", "base", "base (join failed)", "twin"};
static uint64_t ubc_checked[UBC_KINDS], ubc_bad[UBC_KINDS], ubc_subs;
static int ubc_logged[UBC_KINDS];
int vk_usnap_check_on(void) {
    if (ubc_on < 0) {
        const char *e = cfg_get("TOTK_UB_CHECK");
        ubc_on = e && *e && strcmp(e, "0");
        if (ubc_on) rt_log("vk_usnap: TOTK_UB_CHECK: uniform bindings from the snapshot are compared with guest memory");
    }
    return ubc_on;
}
// a: the bytes the GPU gets, g: guest memory now; [wlo, whi) not compared
static void ubc_compare(int kind, uint64_t addr, const uint8_t *a, uint64_t n, uint64_t wlo, uint64_t whi, int stage,
                        int slot, const Reg *r) {
    const uint8_t *g = (const uint8_t *)(uintptr_t)addr;
    uint64_t first = UINT64_MAX, last = 0, diff = 0;
    for (uint64_t k = 0; k < n; k++)
        if (a[k] != g[k] && !(k >= wlo && k < whi)) {
            if (first == UINT64_MAX) first = k;
            last = k, diff++;
        }
    ubc_checked[kind]++;
    if (!diff) return;
    ubc_bad[kind]++;
    // small differences of a version are mostly the game writing ahead: only the first few; others up to 150 each
    if (ubc_logged[kind] >= (kind == UBC_VERSION ? 20 : 150)) return;
    ubc_logged[kind]++;
    rt_log("ubcheck: %s stage %d slot %d %#llx+%#llx: %llu bytes differ, +%#llx..+%#llx; region %#llx+%#llx%s%s (base "
           "%#llx+%#llx)", ubc_name[kind], stage, slot, (unsigned long long)addr, (unsigned long long)n,
           (unsigned long long)diff, (unsigned long long)first, (unsigned long long)last,
           r ? (unsigned long long)r->lo : 0ull, r ? (unsigned long long)(r->hi - r->lo) : 0ull,
           r && r->split ? " split" : "", r && r->dropped ? " dropped" : "", r ? (unsigned long long)r->blo : 0ull,
           r ? (unsigned long long)(r->bhi - r->blo) : 0ull);
}
void vk_usnap_check(uint64_t addr, const VkDescriptorBufferInfo *bi, int stage, int slot) {
    if (!U.arena || !U.arena->shadow || bi->buffer != U.arena->buf) return;
    const Reg *r = find_reg;
    int kind = find_kind[0] == 'v' ? UBC_VERSION : find_kind[0] == 'j' ? UBC_JOIN
               : find_kind[4] == ' ' ? UBC_BASE_NOJOIN : UBC_BASE;
    uint64_t wlo = 0, whi = 0;
    if (r && r->split && (kind == UBC_BASE || kind == UBC_BASE_NOJOIN) && addr == r->blo) wlo = 0x10, whi = 0x50;
    ubc_compare(kind, addr, U.arena->shadow + bi->offset, bi->range, wlo, whi, stage, slot, r);
}
void vk_usnap_check_twin(uint64_t addr, const uint8_t *src, uint64_t n, int stage, int slot) {
    ubc_compare(UBC_TWIN, addr, src, n, 0, 0, stage, slot, NULL);
}
// once per submission (vk_usnap_begin): a line every 600 submissions with the counts per kind
static void ubc_summary(void) {
    if (ubc_on <= 0 || ++ubc_subs % 600) return;
    rt_log("ubcheck: differing / checked bindings: version %llu / %llu, join %llu / %llu, base %llu / %llu, base (join "
           "failed) %llu / %llu, twin %llu / %llu",
           (unsigned long long)ubc_bad[0], (unsigned long long)ubc_checked[0], (unsigned long long)ubc_bad[1],
           (unsigned long long)ubc_checked[1], (unsigned long long)ubc_bad[2], (unsigned long long)ubc_checked[2],
           (unsigned long long)ubc_bad[3], (unsigned long long)ubc_checked[3], (unsigned long long)ubc_bad[4],
           (unsigned long long)ubc_checked[4]);
    memset(ubc_bad, 0, sizeof ubc_bad), memset(ubc_checked, 0, sizeof ubc_checked);
}

void vk_usnap_drop_ub(uint64_t addr, uint64_t n, int ub) {
    if (!U.mode || !n) return;
    if (U.nwr == U.cwr) U.wr = grow(U.wr, &U.cwr, sizeof *U.wr);
    U.wr[U.nwr++] = (Wrote){addr, addr + n, U.fence, ub != 0};
    if (!U.on) return;
    for (int i = first_above(addr); i < U.nreg && U.reg[i].lo < addr + n; i++)
        if (!U.reg[i].dropped) U.reg[i].dropped = 1, U.drops++, U.gen++;
}

void vk_usnap_drop(uint64_t addr, uint64_t n) { vk_usnap_drop_ub(addr, n, 0); }

// TOTK_UB_INPLACE (vk_exec_draw.c op_update_ub): nvnCommandBufferUpdateUniformBuffer of [dst, dst+n) as a new version
// of the snapshot region holding it: the region's bytes as the earlier draws read them, with the update applied, at a
// new place in the arena; the draws bound from here on read the new version, the earlier ones keep theirs. 1 when done;
// 2 when no uniform binding of the submission touches [dst, dst+n): nothing of it reads the range through the
// snapshot, and the caller only writes guest memory (the update is for a later submission or frame: about half of the
// ~440 updates a frame outdoors); 0 (partly in a region, a region over TOTK_UB_INPLACE_MAX_KB, dropped by a GPU write, the arena full) leaves the
// update to the caller's copy on the GPU timeline.
int vk_usnap_update(uint64_t dst, uint64_t n, const void *data) {
    if (!U.on || !U.arena || !U.arena->shadow || !n) return 0;
    int i = first_above(dst);
    Reg *r = i < U.nreg ? &U.reg[i] : NULL;
    if (!r || r->lo >= dst + n) {
        int a = 0, b = U.nexcl;   // the first left-out range ending above dst
        while (a < b) {
            int m = (a + b) / 2;
            if (U.excl[m].hi <= dst) a = m + 1;
            else b = m;
        }
        if (a < U.nexcl && U.excl[a].lo < dst + n) {   // bound live in this submission
            U.upd_live++;
            if (U.excl_ub[a]) U.upd_live_ub++;
            return 0;
        }
        for (int k = 0; k < U.nwr; k++)   // a GPU write still to come in this submission would land after this one
            if (U.wr[k].fence >= U.fence && U.wr[k].lo < dst + n && U.wr[k].hi > dst) { U.upd_none++; return 0; }
        U.upd_direct++;
        return 2;
    }
    if (U.random && r->hi - r->lo > upd_max && dst >= r->lo && dst < r->hi) big_note("update", r, dst, n);
    if (r->dropped) { U.upd_dropped++; return 0; }
    if (dst < r->lo || dst + n > r->hi) { U.upd_none++; return 0; }
    uint64_t rn = r->hi - r->lo, need = (rn + ALIGN - 1) & ~(uint64_t)(ALIGN - 1);
    if (rn > upd_max && dst + n - r->lo <= upd_max && split_on()) {
        // A big region updated near its start (TotK: 64 bytes at +0x10 of a 64 KiB block before every draw): split
        // there, so the versions copy the head; bindings reaching past it join the parts (join), and with
        // TOTK_UB_REACH most draws bind only the start they read.
        uint64_t head = dst + n - r->lo < 4096 ? 4096 : (dst + n - r->lo + ALIGN - 1) & ~(uint64_t)(ALIGN - 1);
        if (U.nreg == U.creg) U.reg = grow(U.reg, &U.creg, sizeof *U.reg), r = &U.reg[i];
        memmove(&U.reg[i + 2], &U.reg[i + 1], (size_t)(U.nreg - i - 1) * sizeof *U.reg);
        U.nreg++;
        if (!r->split) r->split = 1, r->blo = r->lo, r->bhi = r->hi, r->boff = r->off;
        U.reg[i + 1] = (Reg){r->lo + head, r->hi, r->off + head, r->dropped, r->found, 1, r->blo, r->bhi, r->boff};
        r->hi = r->lo + head;
        rn = head, need = head;
        U.splits++;
    }
    if (rn > upd_max) { U.upd_big++, U.upd_big_bytes += rn; return 0; }
    if (U.used + need > U.arena->size) { U.upd_full++; return 0; }
    uint8_t *v = U.arena->shadow + U.used;
    memcpy(v, U.arena->shadow + r->off, rn);
    memcpy(v + (dst - r->lo), data, n);
    memcpy(U.arena->map + U.used, v, rn);   // a sequential write into the GPU's copy
    r->off = U.used;
    U.used += need;
    // Bound descriptors of the region resolve again (vk_exec_draw.c bind_sig compares a draw with the one before it);
    // only when a draw bound it since its last version: a draw that binds it later differs from the one before anyway.
    // A bump per version made every next draw rebuild its descriptor sets (~230 a frame outdoors).
    if (r->found) U.gen++, r->found = 0;
    U.upd_done++, U.upd_bytes += rn;
    return 1;
}

// An update this submission wrote into guest memory over bytes an update copy of an earlier submission is still to
// write on the GPU timeline: that copy would land after it with older bytes. The write is done again once the newest
// such copy completed (completion thread).
typedef struct { uint64_t dst, n; uint8_t data[]; } Late;
static void late_write(void *arg) {
    Late *l = arg;
    mem_thaw(l->dst, l->n);
    memcpy((void *)(uintptr_t)l->dst, l->data, l->n);
    vk_guest_written(l->dst, l->n);
    free(l);
}
void vk_usnap_written(uint64_t dst, uint64_t n, const void *data) {
    uint64_t fence = 0;
    for (int k = 0; k < U.npub; k++) {
        PendUb *p = &U.pub[k];
        if (p->lo >= dst + n || p->hi <= dst) continue;
        if (p->fence > fence) fence = p->fence;
        // The bytes guest memory ends up with once the copy and the redone write ran: the later snapshots apply them
        uint64_t a = p->lo > dst ? p->lo : dst, b = p->hi < dst + n ? p->hi : dst + n;
        memcpy(p->data + (a - p->lo), (const uint8_t *)data + (a - dst), b - a);
    }
    if (!fence || fence <= vk_sync_fence_done()) return;
    Late *l = malloc(sizeof *l + n);
    if (!l) rt_fatal("vk_usnap: out of memory");
    l->dst = dst, l->n = n;
    memcpy(l->data, data, n);
    vk_completion_add(fence, late_write, l);
    U.late++;
}

// An update copy on the GPU timeline (op_update_ub's fallback): its bytes, for later submissions' snapshots.
void vk_usnap_ub_copy(uint64_t dst, uint64_t n, const void *data) {
    if (!U.random || !n || n > MAX_RANGE) return;
    if (U.npub == U.cpub) U.pub = grow(U.pub, &U.cpub, sizeof *U.pub);
    uint8_t *d = malloc(n);
    if (!d) rt_fatal("vk_usnap: out of memory");
    memcpy(d, data, n);
    U.pub[U.npub++] = (PendUb){dst, dst + n, U.fence, d};
}

// The bytes [addr, addr+n) as the replay is at now (the window twin's source, vk_exec_draw.c window_info): the current
// version of the region holding them, which a later version must re-resolve (found). NULL when no region holds them.
const uint8_t *vk_usnap_peek(uint64_t addr, uint64_t n) {
    if (!U.on || !U.arena || !U.arena->shadow || !n) return NULL;
    int i = first_above(addr);
    if (i >= U.nreg) return NULL;
    Reg *r = &U.reg[i];
    if (r->dropped || addr < r->lo || addr + n > r->hi) return NULL;
    r->found = 1;
    U.peeks++;
    return U.arena->shadow + r->off + (addr - r->lo);
}

uint64_t vk_usnap_gen(void) { return U.gen; }

// The arenas back on the free list (vk_exec_draw_shutdown: the device is idle, every completion ran).
void vk_usnap_shutdown(void) {
    plat_mutex_lock(&free_lock);
    for (Arena *a = free_arenas, *next; a; a = next)
        next = a->next, vmaDestroyBuffer(vk_ctx()->vma, a->buf, a->alloc), free(a->shadow), free(a);
    free_arenas = NULL;
    plat_mutex_unlock(&free_lock);
}
