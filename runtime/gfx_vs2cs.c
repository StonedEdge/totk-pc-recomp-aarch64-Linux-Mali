// A vertex shader that writes storage buffers, made a compute shader (gfx_vs_to_cs). For drivers without
// vertexPipelineStoresAndAtomics (the Mali-G610's): they accept such a vertex shader and drop its writes. TotK draws
// points with vertex shaders that sample a texture at a screen position and write the result into a storage buffer
// for the CPU (Ultra Hand finds its target that way; with the writes dropped it found nothing). Such a shader reads
// only gl_VertexIndex as input; its outputs (gl_PerVertex, varyings) are unused for the write. The compute version:
// entry point GLCompute with LocalSize 1 1 1, gl_VertexIndex a Private int set from gl_GlobalInvocationID.x at the
// start of main (one workgroup per vertex; the caller adds the draw's first vertex through the dispatch base), every
// Output a Private variable, the interface's builtin and location decorations dropped. Anything else in the interface
// (vertex attributes, other builtins) and the shader is left as it is (0).
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "gfx.h"

enum { OP_EXT_INST_IMPORT = 11, OP_MEMORY_MODEL = 14, OP_ENTRY_POINT = 15, OP_EXECUTION_MODE = 16, OP_CAPABILITY = 17,
       OP_TYPE_INT = 21, OP_TYPE_VECTOR = 23, OP_TYPE_POINTER = 32, OP_FUNCTION = 54, OP_LABEL = 248,
       OP_VARIABLE = 59, OP_LOAD = 61, OP_STORE = 62, OP_DECORATE = 71, OP_MEMBER_DECORATE = 72,
       OP_COMPOSITE_EXTRACT = 81, OP_BITCAST = 124, OP_NAME = 5, OP_MEMBER_NAME = 6 };
enum { SC_INPUT = 1, SC_OUTPUT = 3, SC_PRIVATE = 6, DEC_BUILTIN = 11, DEC_LOCATION = 30, DEC_BLOCK = 2,
       DEC_INVARIANT = 18, DEC_COMPONENT = 31, DEC_FLAT = 14, DEC_NOPERSPECTIVE = 13,
       BI_VERTEX_INDEX = 42, BI_GLOBAL_INVOCATION_ID = 28, EM_LOCAL_SIZE = 17, EXEC_VERTEX = 0, EXEC_GLCOMPUTE = 5 };

static int put_word(uint32_t **o, uint64_t *m, uint64_t *cap, uint32_t v) {
    if (*m == *cap) {
        uint64_t c = *cap ? *cap * 2 : 1024;
        uint32_t *n = realloc(*o, c * 4);
        if (!n) return 0;
        *o = n, *cap = c;
    }
    (*o)[(*m)++] = v;
    return 1;
}

int gfx_vs_to_cs(const uint32_t *w, uint64_t n, uint32_t **out, uint64_t *out_words) {
    *out = NULL, *out_words = 0;
    if (n < 5 || w[0] != 0x07230203) return 0;
    uint32_t bound = w[3];
    if (bound >= (1u << 22)) return 0;
    // pass 1: the entry point, the interface, the types needed
    uint32_t entry = 0, vindex = 0, tuint = 0, tint = 0, tv3uint = 0;
    uint8_t *ptr_sc = calloc(bound, 1), *is_out_var = calloc(bound, 1);
    uint32_t *ptr_pointee = calloc(bound, 4), *builtin = calloc(bound, 4);
    uint32_t *o = NULL;
    uint64_t m = 0, cap = 0;
    int ok = 0;
    if (!ptr_sc || !is_out_var || !ptr_pointee || !builtin) goto done;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) goto done;
        const uint32_t *a = w + i + 1;
        if (op == OP_ENTRY_POINT) {
            if (entry || a[0] != EXEC_VERTEX) goto done;   // one vertex entry point only
            entry = a[1];
        } else if (op == OP_DECORATE && len >= 4 && a[0] < bound && a[1] == DEC_BUILTIN) {
            builtin[a[0]] = a[2] + 1;
        } else if (op == OP_TYPE_INT && len >= 4 && a[1] == 32) {
            if (a[2]) tint = a[0];
            else tuint = a[0];
        } else if (op == OP_TYPE_VECTOR && len >= 4 && a[1] == tuint && tuint && a[2] == 3) {
            tv3uint = a[0];
        } else if (op == OP_TYPE_POINTER && len >= 4 && a[0] < bound) {
            ptr_sc[a[0]] = (uint8_t)a[1], ptr_pointee[a[0]] = a[2];
        } else if (op == OP_VARIABLE && len >= 4 && a[1] < bound) {
            if (a[2] == SC_INPUT) {
                if (builtin[a[1]] != BI_VERTEX_INDEX + 1 || vindex) goto done;   // only gl_VertexIndex as input
                vindex = a[1];
            } else if (a[2] == SC_OUTPUT) {
                is_out_var[a[1]] = 1;
            }
        }
        i += len;
    }
    if (!entry || !vindex || !tint) goto done;
    // new ids
    uint32_t id_uint = tuint ? tuint : bound++, id_v3 = tv3uint ? tv3uint : bound++, id_pin_v3 = bound++, id_gid = bound++;
    uint32_t id_ppriv_int = bound++, id_ld = bound++, id_x = bound++, id_xi = bound++;
    // pass 2: copy with the changes
    int types_done = 0, in_main = 0, main_started = 0;
    for (uint64_t i = 0; i < 5; i++)
        if (!put_word(&o, &m, &cap, w[i])) goto done;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        const uint32_t *a = w + i + 1;
        uint64_t next = i + len;
        // the code that sets gl_VertexIndex: after the variables at the start of main's first block
        if (in_main && main_started == 1 && op != OP_VARIABLE) {
            uint32_t seq[] = {4u << 16 | OP_LOAD, id_v3, id_ld, id_gid,
                              5u << 16 | OP_COMPOSITE_EXTRACT, id_uint, id_x, id_ld, 0,
                              4u << 16 | OP_BITCAST, tint, id_xi, id_x,
                              3u << 16 | OP_STORE, vindex, id_xi};
            for (size_t k = 0; k < sizeof seq / sizeof *seq; k++)
                if (!put_word(&o, &m, &cap, seq[k])) goto done;
            main_started = 2;
        }
        if (op == OP_ENTRY_POINT) {
            // GLCompute, same id and name, interface: gl_GlobalInvocationID only
            uint32_t name_words = len - 3;   // a[2..] holds the name then the interface
            const uint32_t *nm = a + 2;
            uint32_t nw = 0;
            while (nw < name_words) {
                uint32_t v = nm[nw++];
                if (!(v & 0xff000000u) || !(v & 0x00ff0000u) || !(v & 0x0000ff00u) || !(v & 0xffu)) break;
            }
            if (!put_word(&o, &m, &cap, (3u + nw + 1u) << 16 | OP_ENTRY_POINT) || !put_word(&o, &m, &cap, EXEC_GLCOMPUTE) ||
                !put_word(&o, &m, &cap, entry))
                goto done;
            for (uint32_t k = 0; k < nw; k++)
                if (!put_word(&o, &m, &cap, nm[k])) goto done;
            if (!put_word(&o, &m, &cap, id_gid)) goto done;
            uint32_t em[] = {6u << 16 | OP_EXECUTION_MODE, entry, EM_LOCAL_SIZE, 1, 1, 1};
            for (size_t k = 0; k < 6; k++)
                if (!put_word(&o, &m, &cap, em[k])) goto done;
            i = next;
            continue;
        }
        if (op == OP_EXECUTION_MODE) { i = next; continue; }   // a vertex shader's modes do not apply
        if (op == OP_DECORATE && len >= 3 && a[0] < bound &&
            (a[1] == DEC_BUILTIN || a[1] == DEC_LOCATION || a[1] == DEC_COMPONENT || a[1] == DEC_INVARIANT ||
             a[1] == DEC_FLAT || a[1] == DEC_NOPERSPECTIVE) &&
            (a[0] == vindex || is_out_var[a[0]])) {
            if (a[0] == vindex && a[1] == DEC_BUILTIN) {   // the new input in its place
                uint32_t d[] = {4u << 16 | OP_DECORATE, id_gid, DEC_BUILTIN, BI_GLOBAL_INVOCATION_ID};
                for (size_t k = 0; k < 4; k++)
                    if (!put_word(&o, &m, &cap, d[k])) goto done;
            }
            i = next;
            continue;
        }
        if (op == OP_MEMBER_DECORATE && len >= 4 && (a[2] == DEC_BUILTIN || a[2] == DEC_INVARIANT)) { i = next; continue; }
        if (op == OP_DECORATE && len >= 3 && a[1] == DEC_BLOCK && a[0] < bound) {
            // the Block of gl_PerVertex (an output struct): dropped; a pointer to it is an Output pointer
            int is_out_struct = 0;
            for (uint32_t t = 0; t < bound && !is_out_struct; t++)
                is_out_struct = ptr_sc[t] == SC_OUTPUT && ptr_pointee[t] == a[0];
            if (is_out_struct) { i = next; continue; }
        }
        if (op == OP_FUNCTION && !types_done) {   // the new types and the input variable, before the first function
            types_done = 1;
            if (!tuint) {
                uint32_t t[] = {4u << 16 | OP_TYPE_INT, id_uint, 32, 0};
                for (size_t k = 0; k < 4; k++)
                    if (!put_word(&o, &m, &cap, t[k])) goto done;
            }
            if (!tv3uint) {
                uint32_t t[] = {4u << 16 | OP_TYPE_VECTOR, id_v3, id_uint, 3};
                for (size_t k = 0; k < 4; k++)
                    if (!put_word(&o, &m, &cap, t[k])) goto done;
            }
            uint32_t t[] = {4u << 16 | OP_TYPE_POINTER, id_pin_v3, SC_INPUT, id_v3,
                            4u << 16 | OP_VARIABLE, id_pin_v3, id_gid, SC_INPUT};
            for (size_t k = 0; k < 8; k++)
                if (!put_word(&o, &m, &cap, t[k])) goto done;
        }
        if (op == OP_FUNCTION && len >= 3 && a[1] == entry) in_main = 1, main_started = 0;
        if (op == OP_LABEL && in_main && main_started == 0) main_started = 1;
        if (op == OP_TYPE_POINTER && len >= 4 && a[1] == SC_OUTPUT) {   // outputs become private
            uint32_t t[] = {w[i], a[0], SC_PRIVATE, a[2]};
            for (size_t k = 0; k < 4; k++)
                if (!put_word(&o, &m, &cap, t[k])) goto done;
            i = next;
            continue;
        }
        if (op == OP_TYPE_POINTER && len >= 4 && a[1] == SC_INPUT && a[2] == tint) {
            // the pointer type of gl_VertexIndex: a private one next to it for the variable
            if (!put_word(&o, &m, &cap, w[i]) || !put_word(&o, &m, &cap, a[0]) || !put_word(&o, &m, &cap, a[1]) ||
                !put_word(&o, &m, &cap, a[2]))
                goto done;
            uint32_t t[] = {4u << 16 | OP_TYPE_POINTER, id_ppriv_int, SC_PRIVATE, tint};
            for (size_t k = 0; k < 4; k++)
                if (!put_word(&o, &m, &cap, t[k])) goto done;
            i = next;
            continue;
        }
        if (op == OP_VARIABLE && len >= 4 && a[1] == vindex) {
            uint32_t t[] = {4u << 16 | OP_VARIABLE, id_ppriv_int, vindex, SC_PRIVATE};
            for (size_t k = 0; k < 4; k++)
                if (!put_word(&o, &m, &cap, t[k])) goto done;
            i = next;
            continue;
        }
        if (op == OP_VARIABLE && len >= 4 && a[1] < bound && is_out_var[a[1]]) {
            uint32_t t[] = {w[i], a[0], a[1], SC_PRIVATE};
            for (size_t k = 0; k < 4; k++)
                if (!put_word(&o, &m, &cap, t[k])) goto done;
            for (uint32_t k = 4; k < len; k++)   // an initializer stays
                if (!put_word(&o, &m, &cap, w[i + k])) goto done;
            i = next;
            continue;
        }
        for (uint32_t k = 0; k < len; k++)
            if (!put_word(&o, &m, &cap, w[i + k])) goto done;
        i = next;
    }
    if (!types_done || main_started != 2) goto done;
    o[3] = bound;
    ok = 1;
done:
    free(ptr_sc), free(is_out_var), free(ptr_pointee), free(builtin);
    if (!ok) { free(o); return 0; }
    *out = o, *out_words = m;
    return 1;
}
