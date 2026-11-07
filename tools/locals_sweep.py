"""Static cost of the hottest lifted functions at several --reg-locals counts, compiled for this machine.

Usage: locals_sweep.py --exefs DIR --order ORDER.txt [--top 300] [--locals 6,10,14,18] [--cc clang] [--work DIR]

recomp/regs.py keeps a function's LOCALS most used guest registers in C locals; the rest are loads and stores of
the Cpu (c->x[n]) at every use, which the compiler cannot keep in host registers across guest stores. LOCALS = 6 was
chosen on x86-64 (16 general registers). AArch64 has 31, so the best count there can be higher. This emits the top
functions of an order file (tools/hot_order.py: hottest first) with each count, compiles them as the build does
(-O1, -fno-strict-aliasing) and counts per count: instructions, loads, stores, and the loads and stores relative to
the register that holds the Cpu pointer. Fewer is better; the hottest functions count most when --weight is given
(1/rank). Static counts, not executed ones: the game run decides, this picks the candidates.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'recomp'))
sys.path.insert(0, str(ROOT / 'tools'))

import gen  # noqa: E402
from layout import layout  # noqa: E402
from module import Module  # noqa: E402

LOAD = re.compile(r'^(ldr|ldp|ldur|ldrb|ldrh|ldrsb|ldrsh|ldrsw|ldurb|ldurh|ldursw|ldar|ldaxr|ldxr|ld1)')
STORE = re.compile(r'^(str|stp|stur|strb|strh|sturb|sturh|stlr|stxr|stlxr|st1)')


def count(objdump, obj):
    """Per function: instructions, loads, stores, Cpu-relative loads and stores."""
    out = subprocess.run([objdump, '-d', '--no-show-raw-insn', str(obj)], capture_output=True, text=True,
                         check=True).stdout
    res, cur, cpu = {}, None, None
    for line in out.splitlines():
        m = re.match(r'^[0-9a-f]+ <(f_[0-9a-f]+|fb_[0-9a-f]+)>:', line)
        if m:
            cur = m.group(1)
            res[cur] = [0, 0, 0, 0]
            cpu = None
            continue
        m = re.match(r'^\s+[0-9a-f]+:\s+(\S+)\s*(.*)$', line)
        if not m or not cur:
            continue
        op, args = m.group(1), m.group(2).split('//')[0]
        r = res[cur]
        r[0] += 1
        mm = re.match(r'(x\d+), x0$', args.strip())
        if op == 'mov' and mm and cpu is None:
            cpu = mm.group(1)
        on_cpu = bool(re.search(r'\[(x0|' + (cpu or 'x0') + r')[,\]]', args))
        if LOAD.match(op):
            r[1] += 1
            r[3] += on_cpu
        elif STORE.match(op):
            r[2] += 1
            r[3] += on_cpu
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--exefs', type=Path, required=True)
    ap.add_argument('--order', type=Path, required=True)
    ap.add_argument('--top', type=int, default=300)
    ap.add_argument('--locals', default='6,10,14,18')
    ap.add_argument('--cc', default='clang')
    ap.add_argument('--objdump', default='objdump')
    ap.add_argument('--cflags', default='-march=armv8.2-a')
    ap.add_argument('--work', type=Path, default=ROOT / 'build' / 'locals-sweep')
    ap.add_argument('--weight', action='store_true', help='weight function k (hottest = 1) by 1/k')
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)

    want = []
    for line in a.order.read_text().splitlines():
        if line.startswith('f_'):
            want.append(int(line[2:], 16))
        if len(want) >= a.top:
            break
    mods = []
    for name, base, _ in layout(a.exefs):
        mod = Module(name, a.exefs / name, base, 'c')
        starts, _ = gen.discover(mod)
        abs_starts = mod.lo + starts * 4
        ends = np.append(abs_starts[1:], mod.hi)
        mods.append((name, mod, abs_starts, ends))
    picks = {}   # module name -> [(entry, end)]
    rank = {}
    for k, addr in enumerate(want):
        for name, mod, abs_starts, ends in mods:
            if mod.lo <= addr < mod.hi:
                i = int(np.searchsorted(abs_starts, addr))
                if i < len(abs_starts) and abs_starts[i] == addr:
                    picks.setdefault(name, []).append((addr, int(ends[i])))
                    rank[f'f_{addr:x}'] = k + 1
                break
    print(f'{sum(len(v) for v in picks.values())} of the {len(want)} functions found in the modules')

    flags = ['-O1', '-fno-strict-aliasing', '-Wno-return-type', '-Wno-unused-label', '-fvisibility=hidden',
             '-ffunction-sections', '-I', str(ROOT / 'runtime')] + a.cflags.split()
    print(f'{"locals":>6} {"instructions":>12} {"loads":>8} {"stores":>8} {"Cpu ld/st":>10} {"vs first":>9}')
    first = None
    for n in [int(x) for x in a.locals.split(',')]:
        tot = np.zeros(4)
        for name, fns in picks.items():
            mod = next(m for nm, m, _, _ in mods if nm == name)
            em = gen.Emitter(mod, set(next(s for nm, _, s, _ in mods if nm == name).tolist()), reg_locals=n)
            src = a.work / f'{name}_l{n}.c'
            src.write_text(gen.chunk_text(name, 0, em, fns))
            obj = src.with_suffix('.o')
            subprocess.run([a.cc] + flags + ['-c', str(src), '-o', str(obj)], check=True)
            for fn, r in count(a.objdump, obj).items():
                w = 1.0 / rank.get(fn.replace('fb_', 'f_'), len(want)) if a.weight else 1.0
                tot += w * np.array(r)
        if first is None:
            first = tot
        print(f'{n:6d} {tot[0]:12.0f} {tot[1]:8.0f} {tot[2]:8.0f} {tot[3]:10.0f} {100 * (tot[0] / first[0] - 1):+8.1f}%')
    return 0


if __name__ == '__main__':
    sys.exit(main())
