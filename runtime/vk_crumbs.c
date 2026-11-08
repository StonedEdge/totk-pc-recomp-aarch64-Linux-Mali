// GPU breadcrumbs for a lost device (vk.h vk_crumb). VK_AMD_buffer_marker writes a 32-bit code into a buffer from
// the command stream: at TOP_OF_PIPE once the command processor reaches the marker, at BOTTOM_OF_PIPE once every
// command before it has finished. Each crumb writes both, into two words of one host-coherent buffer, so after a
// device loss word 0 names the newest point the GPU reached and word 1 the newest point it had finished: the hang lies
// between them. A code is the crumb kind in the top 4 bits and its number in the low 28. The markers cost two small
// writes per crumb, and crumbs mark submissions, upload batches, ASTC chunks and presents, not draws.
//
// Pass marks (vk.h vk_pass_mark_begin) use words 2 and 3 of the same buffer: the replay marks the start and the end of
// every render pass and dispatch, and a ring of records says what each mark was. The design follows Savage Wind
// Recomp's GPU checkpoints (refs/SavageWindRecomp runtime/nvn/vulkan/exec.c ck_mark and vk_checkpoints_report: a mark
// per pass and dispatch, numbers in a ring of 4096, text only at the device loss); the code is ours.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "vk.h"

static PFN_vkCmdWriteBufferMarkerAMD write_marker;
static VkBuffer buf;
static VmaAllocation alloc;
static volatile uint32_t *words;   // 0, 1: crumbs; 2, 3: pass marks

// Pass marks (below)
int vk_pass_marks_on;
static int pm_amd;                          // marks into words 2 and 3
// TOTK_PASS_CRUMBS=fill, for GPUs without VK_AMD_buffer_marker (Arm's Mali): vkCmdFillBuffer writes the marks into
// words 2 and 3. A transfer only runs where a barrier puts it, so an end mark waits for everything before it (a
// barrier per pass and dispatch: slow, for finding the work a lost device was running, not for play). The start
// mark follows the pass's own barrier.
static int pm_fill;
// Without any GPU-side marks (no VK_AMD_buffer_marker, no NVIDIA checkpoints, no fill): the marks are kept on the CPU
// only, at no GPU cost, and a lost device lists those of the oldest submission the timeline had not reached
// (vk_sync_fence_done): the work the GPU was stuck in is among them. TOTK_PASS_CRUMBS=0 turns it off.
static int pm_cpu;
static PFN_vkCmdSetCheckpointNV set_checkpoint;
static void pass_marks_init(VkCtx *c);

static void buffer_init(VkCtx *c) {
    write_marker = (PFN_vkCmdWriteBufferMarkerAMD)vkGetDeviceProcAddr(c->device, "vkCmdWriteBufferMarkerAMD");
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 16,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                  .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
    VmaAllocationInfo info;
    if (!write_marker || vmaCreateBuffer(c->vma, &bi, &ai, &buf, &alloc, &info) != VK_SUCCESS) {
        rt_log("vk: GPU breadcrumbs unavailable");
        write_marker = NULL;
        return;
    }
    words = info.pMappedData;
    words[0] = words[1] = words[2] = words[3] = 0;
    rt_log("vk: GPU breadcrumbs on (VK_AMD_buffer_marker)");
}

static void fill_buffer_init(VkCtx *c) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 16,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                  .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(c->vma, &bi, &ai, &buf, &alloc, &info) != VK_SUCCESS) {
        rt_log("vk: pass marks through vkCmdFillBuffer unavailable (no host-visible buffer)");
        return;
    }
    words = info.pMappedData;
    words[0] = words[1] = words[2] = words[3] = 0;
    pm_fill = 1;
}

void vk_crumbs_init(VkCtx *c) {
    if (c->has_buffer_marker) buffer_init(c);
    else {
        const char *e = cfg_get("TOTK_PASS_CRUMBS");
        if (e && !strcmp(e, "fill")) fill_buffer_init(c);
    }
    pass_marks_init(c);
}

void vk_crumbs_release(void) {
    VkCtx *c = vk_ctx();
    if (buf && c) vmaDestroyBuffer(c->vma, buf, alloc);
    buf = VK_NULL_HANDLE, alloc = NULL, words = NULL, write_marker = NULL;
    vk_pass_marks_on = pm_amd = pm_fill = pm_cpu = 0, set_checkpoint = NULL;
}

void vk_crumb(VkCommandBuffer cb, int kind, uint64_t n) {
    if (!write_marker || !cb) return;
    uint32_t code = (uint32_t)kind << 28 | (uint32_t)(n & 0x0fffffff);
    write_marker(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, buf, 0, code);
    write_marker(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, buf, 4, code);
}

static void describe(uint32_t code, char *o, size_t size) {
    static const char *const kinds[] = {"nothing", "the start of a part of submission", "the end of submission",
                                        "the start of upload batch", "the end of upload batch",
                                        "ASTC decode chunk", "the start of present",
                                        "the end of present"};
    unsigned k = code >> 28;
    if (!code) snprintf(o, size, "nothing yet");
    else if (k < sizeof kinds / sizeof *kinds) snprintf(o, size, "%s %u", kinds[k], code & 0x0fffffff);
    else snprintf(o, size, "unknown code %#x", code);
}

void vk_crumbs_report(void) {
    if (!words) return;
    char reached[96], finished[96];
    describe(words[0], reached, sizeof reached);
    describe(words[1], finished, sizeof finished);
    rt_log("vk: GPU breadcrumbs: the GPU last reached %s; everything before %s had finished", reached, finished);
}

// ---- pass marks

// Cost per pass or dispatch: about twenty stores into the ring and two commands, one at the start and one at the end,
// each a marker write or a checkpoint. Per draw nothing, except that the branch of vk_exec_draw.c prepare_draw that
// records changed pipeline state also notes the shaders of a pass's first draw. A full barrier already precedes every
// pass and dispatch (vk_exec.c vk_exec_pass_begin, vk_exec_draw.c op_dispatch), so the end-of-pipe write adds no wait.
static VkPassMark ring[VK_PASS_MARKS];
static uint32_t seq;                        // marks made; only the replay thread writes it

// TOTK_PASS_CRUMBS: unset, buffer marks where VK_AMD_buffer_marker exists; 1, also NVIDIA checkpoints; 0, neither.
int vk_pass_marks_want_nv(void) {
    const char *e = cfg_get("TOTK_PASS_CRUMBS");
    return e && *e && strcmp(e, "0") != 0;
}

static void pass_marks_init(VkCtx *c) {
    const char *e = cfg_get("TOTK_PASS_CRUMBS");
    int off = e && !strcmp(e, "0");
    set_checkpoint = c->has_nv_checkpoints
                         ? (PFN_vkCmdSetCheckpointNV)vkGetDeviceProcAddr(c->device, "vkCmdSetCheckpointNV") : NULL;
    pm_amd = !off && write_marker && words;
    pm_cpu = !off && !pm_amd && !pm_fill && !set_checkpoint;
    vk_pass_marks_on = pm_amd || pm_fill || set_checkpoint || pm_cpu;
    if (pm_cpu)
        rt_log("vk: pass marks on the CPU only (no GPU marks on this driver): a lost device lists the passes and "
               "dispatches of the oldest unfinished submission (TOTK_PASS_CRUMBS=0 turns them off)");
    else if (pm_fill)
        rt_log("vk: pass marks on (vkCmdFillBuffer, TOTK_PASS_CRUMBS=fill: a barrier per pass and dispatch, slow)");
    else if (vk_pass_marks_on)
        rt_log("vk: pass marks on (%s%s%s; TOTK_PASS_CRUMBS=0 turns them off%s)", pm_amd ? "VK_AMD_buffer_marker" : "",
               pm_amd && set_checkpoint ? " and " : "", set_checkpoint ? "VK_NV_device_diagnostic_checkpoints" : "",
               set_checkpoint ? "" : ", =1 adds NVIDIA checkpoints where the driver has them");
    else
        rt_log("vk: pass marks off (%s)", off ? "TOTK_PASS_CRUMBS=0"
                                              : "no VK_AMD_buffer_marker; TOTK_PASS_CRUMBS=1 asks for NVIDIA checkpoints");
}

void vk_pass_mark_cmd(VkCommandBuffer cb, uint32_t id, int end) {
    if (pm_cpu) return;   // the ring record is the mark
    if (pm_fill) {
        if (end) {   // after everything recorded before it
            VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                   .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                   .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                   .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                   .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
            VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1,
                                   .pMemoryBarriers = &mb};
            vkCmdPipelineBarrier2(cb, &di);
        }
        vkCmdFillBuffer(cb, buf, end ? 12 : 8, 4, id);
        return;
    }
    if (pm_amd)
        write_marker(cb, end ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, buf,
                     end ? 12 : 8, id);
    if (set_checkpoint) set_checkpoint(cb, (const void *)(uintptr_t)((uint64_t)id << 1 | (end != 0)));
}

VkPassMark *vk_pass_mark_begin(VkCommandBuffer cb, int kind, uint64_t fence) {
    if (!vk_pass_marks_on || !cb) return NULL;
    uint32_t id = ++seq;
    if (!id) id = ++seq;   // 0 means no mark
    VkPassMark *m = &ring[id & (VK_PASS_MARKS - 1)];
    m->id = id, m->kind = (uint8_t)kind, m->fence = fence, m->draws = UINT32_MAX;
    rec_pass_mark(cb, id, 0);
    return m;
}

void vk_pass_mark_end(VkCommandBuffer cb, const VkPassMark *m) {
    if (m && cb) rec_pass_mark(cb, m->id, 1);
}

int vk_pass_marks_read(uint32_t *started, uint32_t *finished) {
    if (!pm_amd && !pm_fill) return 0;
    *started = words[2], *finished = words[3];
    return 1;
}

const VkPassMark *vk_pass_mark_record(uint32_t id) {
    const VkPassMark *m = &ring[id & (VK_PASS_MARKS - 1)];
    return id && m->id == id ? m : NULL;
}

// The first 8 bytes of the shader's input id (gfx_shader.c), the start of its stored input's name <cache>/in/<id>.in.
static unsigned long long shader_id(const GfxShader *s) {
    if (!s) return 0;
    const uint8_t *iid = gfx_shader_input_id(s);
    unsigned long long v = 0;
    for (int i = 0; i < 8; i++) v = v << 8 | iid[i];
    return v;
}

static void describe_mark(uint32_t id, char *o, size_t size) {
    const VkPassMark *m = vk_pass_mark_record(id);
    if (!id) {
        snprintf(o, size, "nothing yet");
        return;
    }
    if (!m) {
        snprintf(o, size, "no record (a later mark took its slot)");
        return;
    }
    if (m->kind == VK_MARK_DISPATCH) {
        snprintf(o, size, "dispatch %u of submission %llu: %ux%ux%u groups%s, cs %016llx", m->step,
                 (unsigned long long)m->fence, m->w, m->h, m->d, m->w ? "" : " (indirect)", shader_id(m->first[0]));
        return;
    }
    size_t n = (size_t)snprintf(o, size, "pass %u of submission %llu: %ux%u, %u layer%s, colour formats", m->step,
                                (unsigned long long)m->fence, m->w, m->h, m->d, m->d == 1 ? "" : "s");
    for (int i = 0; i < m->ncolor && i < 8 && n < size; i++)
        n += (size_t)snprintf(o + n, size - n, "%s%d", i ? "," : " ", (int)m->color[i]);
    if (!m->ncolor && n < size) n += (size_t)snprintf(o + n, size - n, " none");
    if (n < size) n += (size_t)snprintf(o + n, size - n, ", depth format %d", (int)m->depth);
    if (n >= size) return;
    if (m->draws == UINT32_MAX)
        snprintf(o + n, size - n, ", still recording; first draw vs %016llx fs %016llx", shader_id(m->first[0]),
                 shader_id(m->first[1]));
    else if (!m->draws)
        snprintf(o + n, size - n, ", no draws");
    else
        snprintf(o + n, size - n, ", %u draws; first vs %016llx fs %016llx, last vs %016llx fs %016llx", m->draws,
                 shader_id(m->first[0]), shader_id(m->first[1]), shader_id(m->last[0]), shader_id(m->last[1]));
}

void vk_pass_marks_format(const char *source, uint32_t started, uint32_t finished, char *o, size_t size) {
    char a[640], b[640];
    describe_mark(started, a, sizeof a);
    describe_mark(finished, b, sizeof b);
    snprintf(o, size,
             "vk: pass marks (%s, %u made): the GPU last started mark %u and last finished mark %u\n"
             "vk:   last started %u: %s\n"
             "vk:   last finished %u: %s",
             source, (unsigned)seq, (unsigned)started, (unsigned)finished, (unsigned)started, a, (unsigned)finished, b);
}

static void log_lines(char *t) {
    for (char *line = t, *nl; line && *line; line = nl ? nl + 1 : NULL) {
        nl = strchr(line, '\n');
        if (nl) *nl = 0;
        rt_log("%s", line);
    }
}

// vkGetQueueCheckpointData2NV, else vkGetQueueCheckpointDataNV, on the graphics queue (the replay's): one entry per
// stage with the last checkpoint that stage executed. A checkpoint is the mark's id times 2, plus 1 at its end, so the
// top-of-pipe entry names the last start or end the GPU reached, and the bottom-of-pipe entry the last one before which
// everything had finished: at a start, that is the end of the mark before it.
static void nv_report(void) {
    VkCtx *c = vk_ctx();
    if (!c || !set_checkpoint) return;
    PFN_vkGetQueueCheckpointData2NV get2 =
        (PFN_vkGetQueueCheckpointData2NV)vkGetDeviceProcAddr(c->device, "vkGetQueueCheckpointData2NV");
    PFN_vkGetQueueCheckpointDataNV get1 =
        (PFN_vkGetQueueCheckpointDataNV)vkGetDeviceProcAddr(c->device, "vkGetQueueCheckpointDataNV");
    enum { MAXN = 32 };
    uint64_t stage[MAXN], marker[MAXN];
    uint32_t n = 0;
    if (get2) {
        VkCheckpointData2NV d[MAXN];
        get2(c->gfx_queue, &n, NULL);
        if (n > MAXN) n = MAXN;
        for (uint32_t i = 0; i < n; i++) d[i] = (VkCheckpointData2NV){.sType = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_2_NV};
        get2(c->gfx_queue, &n, d);
        for (uint32_t i = 0; i < n; i++) stage[i] = d[i].stage, marker[i] = (uint64_t)(uintptr_t)d[i].pCheckpointMarker;
    } else if (get1) {
        VkCheckpointDataNV d[MAXN];
        get1(c->gfx_queue, &n, NULL);
        if (n > MAXN) n = MAXN;
        for (uint32_t i = 0; i < n; i++) d[i] = (VkCheckpointDataNV){.sType = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV};
        get1(c->gfx_queue, &n, d);
        for (uint32_t i = 0; i < n; i++) stage[i] = d[i].stage, marker[i] = (uint64_t)(uintptr_t)d[i].pCheckpointMarker;
    } else {
        rt_log("vk: pass marks: no vkGetQueueCheckpointDataNV entry point");
        return;
    }
    uint32_t started = 0, finished = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t id = (uint32_t)(marker[i] >> 1);
        int end = (int)(marker[i] & 1);
        rt_log("vk: checkpoint of stage %#llx: the %s of mark %u", (unsigned long long)stage[i], end ? "end" : "start", id);
        if (stage[i] == VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT) started = id;
        if (stage[i] == VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT) finished = end ? id : id ? id - 1 : 0;
    }
    if (!n) rt_log("vk: pass marks: the driver reported no checkpoints");
    char t[2048];
    vk_pass_marks_format("VK_NV_device_diagnostic_checkpoints", started, finished, t, sizeof t);
    log_lines(t);
}

// Marks first..last (fence 0: of any submission), runs of alike marks (same kind, size, formats, draw count and
// shaders) one line each: a stuck submission can hold a thousand small passes of a few kinds, and the odd one out is
// what to look at.
static void list_marks(uint32_t first, uint32_t last, uint64_t want) {
    char t[640];
    uint32_t lines = 0, run_from = 0, run_n = 0;
    const VkPassMark *prev = NULL;
    for (uint32_t id = first; id && id <= last + 1; id++) {
        const VkPassMark *m = id <= last ? vk_pass_mark_record(id) : NULL;
        if (m && want && m->fence != want) continue;
        int alike = m && prev && m->kind == prev->kind && m->w == prev->w && m->h == prev->h && m->d == prev->d &&
                    m->ncolor == prev->ncolor && m->depth == prev->depth && m->draws == prev->draws &&
                    m->first[0] == prev->first[0] && m->first[1] == prev->first[1] &&
                    !memcmp(m->color, prev->color, sizeof m->color);
        if (alike) {
            run_n++;
            continue;
        }
        if (prev) {
            describe_mark(run_from, t, sizeof t);
            if (run_n > 1)
                rt_log("vk:   marks %u-%u (%u alike): %s", (unsigned)run_from, (unsigned)(run_from + run_n - 1),
                       (unsigned)run_n, t);
            else
                rt_log("vk:   mark %u: %s", (unsigned)run_from, t);
            if (++lines >= 200) {
                rt_log("vk:   ... (200 lines shown)");
                return;
            }
        }
        if (!m) break;
        prev = m, run_from = id, run_n = 1;
    }
}

static volatile uint32_t part_done;   // TOTK_PART_WAIT: every mark up to this one finished

uint32_t vk_pass_marks_seq(void) { return __atomic_load_n(&seq, __ATOMIC_ACQUIRE); }
void vk_pass_marks_part_done(uint32_t s) { __atomic_store_n(&part_done, s, __ATOMIC_RELEASE); }

// The marks of the oldest submission the GPU had not finished, in the order they were recorded; with TOTK_PART_WAIT,
// only those after the last part that finished.
static void cpu_report(void) {
    uint64_t done = vk_sync_fence_done(), want = done + 1;
    uint32_t first = seq > VK_PASS_MARKS - 1 ? seq - (VK_PASS_MARKS - 1) : 1, n = 0, later = 0;
    uint32_t pd = __atomic_load_n(&part_done, __ATOMIC_ACQUIRE);
    if (pd) {
        rt_log("vk: pass marks (TOTK_PART_WAIT): every mark up to %u finished; the GPU stopped in marks %u-%u",
               (unsigned)pd, (unsigned)pd + 1, (unsigned)seq);
        if (pd + 1 > first) first = pd + 1;
    }
    for (uint32_t id = first; id && id <= seq; id++) {
        const VkPassMark *m = vk_pass_mark_record(id);
        if (!m) continue;
        if (m->fence == want) n++;
        else if (m->fence > want) later++;
    }
    rt_log("vk: pass marks (CPU only): submissions done up to %llu; submission %llu, the oldest unfinished, has %u "
           "passes and dispatches (%u more in later submissions):", (unsigned long long)done, (unsigned long long)want,
           n, later);
    list_marks(first, seq, want);
}

// TOTK_PART_WAIT (vk_sync.c): a part did not finish in time. Its marks, after the last part that did.
void vk_pass_marks_part_hung(uint32_t from, uint32_t to, uint64_t fence) {
    rt_log("vk: pass marks (TOTK_PART_WAIT): a part of submission %llu did not finish; marks up to %u had finished, "
           "the part holds marks %u-%u:", (unsigned long long)fence, (unsigned)from, (unsigned)from + 1, (unsigned)to);
    list_marks(from + 1, to, 0);
}

void vk_pass_marks_report(void) {
    if (!vk_pass_marks_on) return;
    if (pm_cpu) {
        cpu_report();
        return;
    }
    uint32_t started, finished;
    char t[2048];
    if (vk_pass_marks_read(&started, &finished)) {
        vk_pass_marks_format(pm_fill ? "vkCmdFillBuffer" : "VK_AMD_buffer_marker", started, finished, t, sizeof t);
        log_lines(t);
    }
    nv_report();
}
