"""List the game's parameters whose names suggest distances, detail levels or budgets, from the RomFS of the player's
game, so switchable patches that make the game do less (shorter object, detail or grass distances) can be chosen.

Usage: find_params.py --romfs <dir with index.json and romfs.bin> [--main <exefs>/main] [-o out.txt] [--all-keys]
       [--limit N]

Reads the RomFS as tools/extract.py leaves it (index.json + romfs.bin), the zstd dictionaries from Pack/ZsDic.pack.zs,
every BYML file (.byml, .bgyml, zstd-compressed or not) on its own or inside a SARC pack (.pack, .sarc), and walks it.
Keys matching KEY (distance, LOD, range, cull, density, budget, ...) are collected by where they occur: the file's path
with the first part of its name replaced by * (so the thousands of actor packs fold into one row per component kind)
and the key path inside the file. Per row: how many files have it, the distinct values (the most common first, with
counts), and three example files. Paths under System/ and GameParameter-like folders are also listed key by key in
full. Prints an inventory of the RomFS folders first. The output holds names and numbers only, no file contents, and
goes into build/ (ignored): it is for choosing candidates, the patches then name the files they change.

--main also lists the parameter names in the main program's strings that match NAME (LOD, distance, cull, density,
...): the engine registers many parameters in code with their defaults as immediates (the shadow cascade distances,
tools/find_cascade.py), and their names show which parameter objects exist.

Decompression: compression.zstd (Python 3.14, .venv-py314) or the zstandard package.
"""
import argparse
import json
import re
import struct
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_z = None
try:
    import compression.zstd as _z
    _ZK = 'std'
except ImportError:
    try:
        import zstandard as _z
        _ZK = 'zstandard'
    except ImportError:
        _ZK = None


def _need():
    if not _ZK:
        sys.exit('find_params: needs compression.zstd (Python 3.14) or the zstandard package')


def zdict(b):
    _need()
    return _z.ZstdDict(b) if _ZK == 'std' else _z.ZstdCompressionDict(b)


def zdict_id(d):
    return d.dict_id if _ZK == 'std' else d.dict_id()


def zframe_dict(b):
    _need()
    return _z.get_frame_info(b).dictionary_id if _ZK == 'std' else _z.get_frame_parameters(b).dict_id


def zdec(b, d=None):
    _need()
    if _ZK == 'std':
        return _z.decompress(b, zstd_dict=d) if d else _z.decompress(b)
    dc = _z.ZstdDecompressor(dict_data=d) if d else _z.ZstdDecompressor()
    return dc.decompressobj().decompress(b)


KEY = re.compile(r'lod|dist|range|radius|far|near|cull|visib|draw|densit|grass|foliage|budget|quality|perf|dynamic|'
                 r'activ|spawn|limit|maxnum|nummax|max_?num|max_?count|load|stream|detail|level|scale|interval|'
                 r'priority|cost|reduc', re.I)
NAME = re.compile(rb'(?i)lod|dist|far_|_far|near_|_near|cull|densit|grass|foliage|budget|quality|reduc|draw_?range|'
                  rb'visib|activat|load_?range|unload')
FULL_DIRS = re.compile(r'^(System|GameParameter|Global|Environment|Sky|Performance)', re.I)   # not Banc: millions of lines
BYML_EXT = ('.byml', '.bgyml')
PACK_EXT = ('.pack', '.sarc')


def sarc_members(data):
    """name -> bytes of every member of a SARC archive (little or big endian)."""
    if data[:4] != b'SARC':
        return {}
    e = '<' if data[6:8] == b'\xff\xfe' else '>'
    hlen = struct.unpack_from(e + 'H', data, 4)[0]
    data_off = struct.unpack_from(e + 'I', data, 12)[0]
    sfat = hlen
    assert data[sfat:sfat + 4] == b'SFAT'
    slen, count = struct.unpack_from(e + 'HH', data, sfat + 4)
    nodes = sfat + slen
    sfnt = nodes + 16 * count
    assert data[sfnt:sfnt + 4] == b'SFNT'
    names = sfnt + struct.unpack_from(e + 'H', data, sfnt + 4)[0]
    out = {}
    for i in range(count):
        _, attr, lo, hi = struct.unpack_from(e + 'IIII', data, nodes + 16 * i)
        if attr >> 24:
            n = names + (attr & 0xFFFF) * 4
            name = data[n:data.index(b'\0', n)].decode('utf-8', 'replace')
        else:
            name = f'#{i}'
        out[name] = data[data_off + lo:data_off + hi]
    return out


class Byml:
    """A BYML document (versions 2-7, either endianness), walked into (key path, value) pairs."""

    def __init__(self, data):
        if data[:2] == b'YB':
            self.e = '<'
        elif data[:2] == b'BY':
            self.e = '>'
        else:
            raise ValueError('not BYML')
        self.d = data
        self.version, keys, strs, root = struct.unpack_from(self.e + 'HIII', data, 2)
        self.keys = self.table(keys)
        self.strs = self.table(strs)
        self.root = root

    def u32(self, o):
        return struct.unpack_from(self.e + 'I', self.d, o)[0]

    def u24(self, o):
        b = self.d[o:o + 3]
        return int.from_bytes(b, 'little' if self.e == '<' else 'big')

    def table(self, off):
        if not off or self.d[off] != 0xC2:
            return []
        n = self.u24(off + 1)
        return [self.d[off + self.u32(off + 4 + 4 * i):self.d.index(b'\0', off + self.u32(off + 4 + 4 * i))]
                .decode('utf-8', 'replace') for i in range(n)]

    def value(self, t, v):
        if t == 0xA0:
            return self.strs[v] if v < len(self.strs) else f'<str {v}>'
        if t == 0xD0:
            return bool(v)
        if t == 0xD1:
            return struct.unpack(self.e + 'i', struct.pack(self.e + 'I', v))[0]
        if t == 0xD2:
            return round(struct.unpack(self.e + 'f', struct.pack(self.e + 'I', v))[0], 6)
        if t == 0xD3:
            return v
        if t in (0xD4, 0xD5, 0xD6):
            fmt = {0xD4: 'q', 0xD5: 'Q', 0xD6: 'd'}[t]
            x = struct.unpack_from(self.e + fmt, self.d, v)[0]
            return round(x, 9) if t == 0xD6 else x
        if t == 0xFF:
            return None
        return f'<type {t:#x}>'

    def walk(self, out, node=None, path='', depth=0):
        """Appends (path, value) for every leaf under node (default: the root)."""
        if node is None:
            if not self.root:
                return
            node = self.root
        if depth > 40:
            return
        t = self.d[node]
        n = self.u24(node + 1)
        if t == 0xC1:   # dictionary: n entries (u24 key index, u8 type, u32 value)
            for i in range(n):
                o = node + 4 + 8 * i
                k = self.u24(o)
                ct = self.d[o + 3]
                v = self.u32(o + 4)
                self.child(out, ct, v, f'{path}.{self.keys[k] if k < len(self.keys) else k}', depth)
        elif t == 0xC0:   # array: n types, aligned, then n values
            vals = node + 4 + ((n + 3) & ~3)
            for i in range(n):
                self.child(out, self.d[node + 4 + i], self.u32(vals + 4 * i), f'{path}[]', depth)
        elif t in (0x20, 0x21):   # hash maps (v7): n (hash, value) entries, then n types
            w = 8 if t == 0x20 else 12
            types = node + 4 + w * n
            for i in range(n):
                o = node + 4 + w * i
                self.child(out, self.d[types + i], self.u32(o + w - 4), f'{path}{{#}}', depth)

    def child(self, out, t, v, path, depth):
        if t in (0xC0, 0xC1, 0x20, 0x21):
            if 0 < v < len(self.d):
                self.walk(out, v, path, depth + 1)
        else:
            out.append((path, self.value(t, v)))


def fold(path):
    """The path with the first dot-part of its file name replaced by * (actor and instance names folded)."""
    head, _, name = path.rpartition('/')
    parts = name.split('.')
    if len(parts) > 1:
        parts[0] = '*'
    return (head + '/' if head else '') + '.'.join(parts)


def fold_dir(path):
    """Pack/Actor/Npc_Zelda.pack.zs -> Pack/Actor/*.pack.zs."""
    return fold(path)


def fmt(v):
    if isinstance(v, float):
        return f'{v:g}'
    s = str(v)
    return s if len(s) <= 60 else s[:57] + '...'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--romfs', type=Path, required=True)
    ap.add_argument('-o', '--out', type=Path)
    ap.add_argument('--main', type=Path, help='the ExeFS main (NSO): also list matching parameter names')
    ap.add_argument('--all-keys', action='store_true', help='every key, not only those matching the name pattern')
    ap.add_argument('--limit', type=int, default=0, help='stop after this many RomFS files (testing)')
    a = ap.parse_args()
    out = open(a.out, 'w') if a.out else sys.stdout

    def p(*x):
        print(*x, file=out, flush=True)

    files = {f['path']: f for f in json.loads((a.romfs / 'index.json').read_text(encoding='utf-8'))['files']}
    img = (a.romfs / 'romfs.bin').open('rb')

    def read(path):
        f = files[path]
        img.seek(f['offset'])
        return img.read(f['size'])

    dicts = {}
    if 'Pack/ZsDic.pack.zs' in files:
        for name, b in sarc_members(zdec(read('Pack/ZsDic.pack.zs'))).items():
            d = zdict(b)
            dicts[zdict_id(d)] = d

    def load(path, raw):
        if path.endswith('.zs'):
            did = zframe_dict(raw)
            return zdec(raw, dicts.get(did)) if did else zdec(raw)
        return raw

    # inventory: files and bytes per top two folder levels
    inv = Counter()
    for path, f in files.items():
        inv['/'.join(path.split('/')[:2]) if path.count('/') > 1 else path.split('/')[0]] += 1
    p(f'RomFS: {len(files)} files, {len(dicts)} zstd dictionaries')
    p('== inventory (folders, files)')
    for k, n in sorted(inv.items()):
        if n > 1:
            p(f'  {k:60s} {n}')

    rows = defaultdict(lambda: [0, Counter(), []])   # (folded file, key path) -> [files, values, examples]
    full = []                                          # (file, key path, value) of FULL_DIRS files
    seen_files = 0
    t0 = time.time()
    names = [x for x in files if x.endswith(tuple(e + s for e in BYML_EXT + PACK_EXT for s in ('', '.zs')))]
    for i, path in enumerate(sorted(names)):
        if a.limit and i >= a.limit:
            break
        if i and i % 2000 == 0:
            print(f'  {i} of {len(names)} files, {time.time() - t0:.0f} s', file=sys.stderr, flush=True)
        try:
            data = load(path, read(path))
        except Exception as ex:   # noqa: BLE001 - a file the scan cannot read is skipped, and counted
            rows[('<unreadable>', type(ex).__name__)][0] += 1
            continue
        docs = []
        if data[:4] == b'SARC':
            for m, b in sarc_members(data).items():
                if m.endswith(BYML_EXT) or b[:2] in (b'YB', b'BY'):
                    docs.append((fold_dir(path) + ':' + fold(m), path + ':' + m, b))
        elif data[:2] in (b'YB', b'BY'):
            docs.append((fold(path), path, data))
        for folded, real, b in docs:
            try:
                leaves = []
                Byml(b).walk(leaves)
            except Exception:   # noqa: BLE001
                rows[('<bad byml>', folded)][0] += 1
                continue
            seen_files += 1
            want_full = FULL_DIRS.match(real)
            for kp, v in leaves:
                if want_full:
                    full.append((real, kp, v))
                if a.all_keys or KEY.search(kp):
                    r = rows[(folded, kp)]
                    r[0] += 1
                    r[1][fmt(v)] += 1
                    if len(r[2]) < 3 and real not in r[2]:
                        r[2].append(real)
    p(f'\n{seen_files} BYML documents read in {time.time() - t0:.0f} s')
    p('\n== keys matching the pattern, by where they occur (times found; values, most common first; examples)')
    last = None
    for (folded, kp), (n, vals, ex) in sorted(rows.items()):
        if folded != last:
            p(f'\n{folded}')
            last = folded
        top = ', '.join(f'{v} x{c}' if c > 1 else v for v, c in vals.most_common(8))
        more = f' (+{len(vals) - 8} other values)' if len(vals) > 8 else ''
        p(f'  {kp}  [{n}]  {top}{more}')
        if n < 50 and ex:
            p(f'      e.g. {", ".join(ex)}')
    p('\n== every key of the files under System/, GameParameter and similar folders')
    last = None
    for real, kp, v in full[:40000]:
        if real != last:
            p(f'\n{real}')
            last = real
        p(f'  {kp} = {fmt(v)}')
    if len(full) > 40000:
        p(f'... {len(full) - 40000} more lines')
    if a.main:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'recomp'))
        from nso import Nso
        img_ = Nso(str(a.main)).image
        found = []
        for m in re.finditer(rb'[\x20-\x7e]{5,120}\0', img_):
            t = m.group()[:-1]
            if NAME.search(t) and not re.search(rb'[ %\\]{3,}', t):
                found.append((m.start(), t.decode()))
        p(f'\n== {len(found)} strings in main matching the parameter name pattern (offset in main, string)')
        for off, t in found[:6000]:
            p(f'  {off:#09x}  {t}')
        if len(found) > 6000:
            p(f'... {len(found) - 6000} more')
    return 0


if __name__ == '__main__':
    sys.exit(main())
