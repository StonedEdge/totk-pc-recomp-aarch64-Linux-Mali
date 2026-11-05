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
typedef struct { uint64_t lo, hi, off; int dropped, found; } Reg;   // found: bound since its last version
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
               "with those copies applied %.1f, guest writes redone after one %.1f",
               U.upd_done / s, U.upd_bytes / s / 1024.0, U.upd_direct / s, U.upd_big / s, upd_max >> 10,
               U.upd_big ? U.upd_big_bytes / (double)U.upd_big / 1024.0 : 0.0, U.upd_dropped / s, U.upd_live / s,
               U.upd_live_ub / s, U.upd_none / s, U.upd_full / s, U.ex_ub / s, U.in_ub / s, U.late / s);
    U.subs = U.off_subs = U.regions = U.bytes = U.copy_ns = U.hits = U.drops = U.inflight = U.refused = 0;
    U.upd_done = U.upd_bytes = U.upd_big = U.upd_none = U.upd_full = U.upd_direct = U.upd_dropped = U.upd_big_bytes = 0;
    U.upd_live = U.upd_live_ub = U.ex_ub = U.in_ub = U.late = 0;
}

static int first_above(uint64_t a);

void vk_usnap_begin(const uint64_t *handles, int n, uint64_t fence) {
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
        U.reg[U.nreg++] = (Reg){lo, hi, total, 0, 0};
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

int vk_usnap_find(uint64_t addr, uint64_t n, uint64_t align, VkDescriptorBufferInfo *out) {
    if (!U.on || !n) return 0;
    int i = first_above(addr);
    if (i >= U.nreg) return 0;
    Reg *r = &U.reg[i];
    if (r->dropped || addr < r->lo || addr + n > r->hi) return 0;
    uint64_t off = r->off + (addr - r->lo);
    if (off % align) return 0;
    r->found = 1;
    *out = (VkDescriptorBufferInfo){U.arena->buf, off, n};
    U.hits++;
    return 1;
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
    if (r->dropped) { U.upd_dropped++; return 0; }
    if (dst < r->lo || dst + n > r->hi) { U.upd_none++; return 0; }
    uint64_t rn = r->hi - r->lo, need = (rn + ALIGN - 1) & ~(uint64_t)(ALIGN - 1);
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

uint64_t vk_usnap_gen(void) { return U.gen; }

// The arenas back on the free list (vk_exec_draw_shutdown: the device is idle, every completion ran).
void vk_usnap_shutdown(void) {
    plat_mutex_lock(&free_lock);
    for (Arena *a = free_arenas, *next; a; a = next)
        next = a->next, vmaDestroyBuffer(vk_ctx()->vma, a->buf, a->alloc), free(a->shadow), free(a);
    free_arenas = NULL;
    plat_mutex_unlock(&free_lock);
}
