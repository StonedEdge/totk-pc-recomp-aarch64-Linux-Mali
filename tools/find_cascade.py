"""Find the sun shadow cascade distances in a TotK main NSO and print them as `cascade` lines of a patch table.

Usage: find_cascade.py --main <exefs>/main [--patches recomp/patches-v121.txt --append]

The depth shadow parameter object registers depth_shadow_near_0_1..3 = 20, 40, 80 m and depth_shadow_far_0 = 160 m
with the floats as immediates (1.4.3: main+0x6fabb8 mov w8, #20.0, then 40.0, 80.0 and 160.0 0x84-0x88 bytes apart;
recomp/patches-v143.txt, set cascade). This looks for the same four `movz wN, #imm, lsl #16` in that order within
MAX_GAP bytes of each other, and for an adrp/add near them that points at a "depth_shadow" string, so other code that
happens to use 20/40/80/160 does not count. Each match becomes four sites that load the value from the patch page
(+64..+76, written by runtime/resolution.c res_shadow_range) instead. The generator checks every original word, so a
wrong site stops the lift instead of patching the wrong code.

--append adds `set cascade TOTK_PATCH_CASCADE 0` and the sites to the patch file when exactly one match has the
string reference (or there is only one match) and the file has no cascade set yet. Standard library and recomp/nso.py (lz4) only.
"""
import argparse
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'recomp'))

from nso import Nso  # noqa: E402

VALUES = [(0x41A0, '20.0'), (0x4220, '40.0'), (0x42A0, '80.0'), (0x4320, '160.0')]
MAX_GAP = 0x200
PAGE_OFF = 64
TAG = b'depth_shadow'


def movz_hi(w):
    """(imm16, rd) of `movz wN, #imm16, lsl #16`, else None."""
    if (w & 0xFFE00000) == 0x52A00000:
        return (w >> 5) & 0xFFFF, w & 31
    return None


def adrp_add_targets(img, lo, hi):
    """Addresses computed by adrp (+ a following add of the same register) in [lo, hi)."""
    out = []
    for pc in range(lo & ~3, hi, 4):
        if pc < 0 or pc + 4 > len(img):
            continue
        w = struct.unpack_from('<I', img, pc)[0]
        if (w & 0x9F000000) != 0x90000000:
            continue
        imm = ((w >> 5) & 0x7FFFF) << 2 | (w >> 29) & 3
        if imm & (1 << 20):
            imm -= 1 << 21
        rd, page = w & 31, (pc & ~0xFFF) + (imm << 12)
        for nxt in range(pc + 4, min(pc + 0x20, len(img) - 3), 4):
            v = struct.unpack_from('<I', img, nxt)[0]
            if (v & 0xFF800000) == 0x91000000 and (v >> 5) & 31 == rd:   # add xD, xN, #imm{, lsl #12}
                add = (v >> 10) & 0xFFF
                out.append((pc, page + (add << 12 if v >> 22 & 1 else add)))
                break
        else:
            out.append((pc, page))
    return out


def find(nso):
    img = nso.image
    text_lo, text_n = nso.text
    words = struct.unpack_from(f'<{text_n // 4}I', img, text_lo)
    firsts = [i for i, w in enumerate(words) if movz_hi(w) and movz_hi(w)[0] == VALUES[0][0]]
    found = []
    for i in firsts:
        chain, at = [i], i
        for imm, _ in VALUES[1:]:
            nxt = next((j for j in range(at + 1, min(at + 1 + MAX_GAP // 4, len(words)))
                        if movz_hi(words[j]) and movz_hi(words[j])[0] == imm), None)
            if nxt is None:
                break
            chain.append(nxt)
            at = nxt
        if len(chain) != 4:
            continue
        offs = [text_lo + 4 * j for j in chain]
        refs = []
        for pc, t in adrp_add_targets(img, offs[0] - 0x100, offs[-1] + 0x100):
            if 0 <= t < len(img) - len(TAG) and img[t:t + len(TAG)] == TAG:
                end = img.index(b'\0', t)
                refs.append((pc, img[t:end].decode('ascii', 'replace')))
        found.append((offs, [words[j] for j in chain], refs))
    return found


def site_lines(offs, words):
    lines = []
    for k, (off, w) in enumerate(zip(offs, words)):
        rd = w & 31
        disp = PAGE_OFF + 4 * k
        ldr = 0xB9400000 | (disp // 4) << 10 | rd << 5 | rd
        lines.append(f'site cascade main {off:x} {w:08x} @page:x{rd} {ldr:08x}   '
                     f'# mov w{rd}, #{VALUES[k][1]} -> ldr w{rd}, [page+{disp}]')
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--main', type=Path, required=True, help='the ExeFS main (NSO)')
    ap.add_argument('--patches', type=Path, help='patch file to add the set to (with --append)')
    ap.add_argument('--append', action='store_true')
    a = ap.parse_args()
    nso = Nso(str(a.main))
    print(f'main build id {nso.build_id}, .text {nso.text[1]:#x} bytes')
    found = find(nso)
    good = [f for f in found if f[2]]
    for offs, words, refs in found:
        print(f'match at main+{offs[0]:x}: gaps ' + ', '.join(f'{b - a_:#x}' for a_, b in zip(offs, offs[1:])) +
              f'; strings nearby: {", ".join(f"{s} (adrp at +{pc:x})" for pc, s in refs) or "none"}')
        for line in site_lines(offs, words):
            print('  ' + line)
    if not found:
        print('no match: the four distances are not movz immediates in this build')
        return 2
    if not good and len(found) == 1:
        print('the only match has no depth_shadow string nearby (the names may be loaded another way): using it; '
              'check the shadows with shadow_range 50 against 100')
        good = found
    if len(good) != 1:
        print(f'{len(good)} matches with a depth_shadow string nearby: not adding anything (needs a look)')
        return 3
    if a.append:
        if not a.patches:
            ap.error('--append needs --patches')
        text = a.patches.read_text()
        if any(l.split('#', 1)[0].split()[:2] == ['set', 'cascade'] for l in text.splitlines()):
            print(f'{a.patches} already has a cascade set: left as it is')
            return 0
        offs, words, _ = good[0]
        add = ['', '# Sun shadow cascade distances (runtime/resolution.c res_shadow_range switches the set on). The depth '
               'shadow', '# parameter object registers depth_shadow_near_0_1..3 = 20, 40, 80 m and depth_shadow_far_0 = '
               '160 m as immediates;', '# with the set on they come from page +64..+76. Sites found by '
               f'tools/find_cascade.py (main build id {nso.build_id[:16]}).',
               'set cascade  TOTK_PATCH_CASCADE  0', *site_lines(offs, words)]
        a.patches.write_text(text.rstrip('\n') + '\n' + '\n'.join(add) + '\n')
        print(f'added the cascade set ({len(offs)} sites) to {a.patches}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
