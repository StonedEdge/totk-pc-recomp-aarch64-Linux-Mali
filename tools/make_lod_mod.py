"""Make a RomFS mod that shortens the actors' LOD distances, so the game updates fewer actors at full rate.

Usage: make_lod_mod.py --romfs <dir with index.json and romfs.bin> --out <mods>/<name> [--near 0.5] [--middle 1]
       [--push 0.5] [--push-min 100]

Each actor pack (Pack/Actor/*.pack.zs) that holds a Component/LODParam/*.engine__component__LODParam.bgyml gets these
values scaled (tools/find_params.py lists them; on 1.2.1 2305 packs have one):
  AreaNearDistance            the near area, 10 m for most actors (--near)
  AreaMiddle0DistanceScale    the middle area as a multiple of the near one, 5 for most (--middle)
  ForbidPushJobDistance       the distance past which the actor's update jobs are no longer queued, 500 or 800 m for
                              most (--push; values of 0 or below --push-min stay)
The values are changed in place: the BYML and the SARC keep their sizes, so the resource size table needs no change.
Each changed pack is compressed again with its zstd dictionary and written to <out>/romfs/Pack/Actor/. The game loads
it as a mod at the next start (runtime/mods.c; on or off in the Mods tab or with TOTK_MODS_DISABLED=<name>).
"""
import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import find_params as fp  # noqa: E402

KEYS = {'AreaNearDistance': 'near', 'AreaMiddle0DistanceScale': 'middle', 'ForbidPushJobDistance': 'push'}


def zcomp(data, d, level):
    fp._need()
    if fp._ZK == 'std':
        return fp._z.compress(data, level=level, zstd_dict=d)
    return fp._z.ZstdCompressor(level=level, dict_data=d).compress(data)


def sarc_spans(data):
    """(name, start, end) of every SARC member, little endian."""
    e = '<'
    hlen = struct.unpack_from(e + 'H', data, 4)[0]
    data_off = struct.unpack_from(e + 'I', data, 12)[0]
    slen, count = struct.unpack_from(e + 'HH', data, hlen + 4)
    nodes = hlen + slen
    sfnt = nodes + 16 * count
    names = sfnt + struct.unpack_from(e + 'H', data, sfnt + 4)[0]
    for i in range(count):
        _, attr, lo, hi = struct.unpack_from(e + 'IIII', data, nodes + 16 * i)
        n = names + (attr & 0xFFFF) * 4
        yield data[n:data.index(b'\0', n)].decode('utf-8', 'replace'), data_off + lo, data_off + hi


def patch_lod(buf, lo, hi, scale, push_min, stats):
    """Scales the keys of the LODParam BYML at buf[lo:] in place (root dictionary, float or int values)."""
    doc = fp.Byml(bytes(buf[lo:hi]))
    if doc.e != '<' or not doc.root or doc.d[doc.root] != 0xC1:
        return 0
    n = doc.u24(doc.root + 1)
    changed = 0
    for i in range(n):
        o = doc.root + 4 + 8 * i
        k = doc.u24(o)
        name = doc.keys[k] if k < len(doc.keys) else ''
        if name not in KEYS:
            continue
        f = scale[KEYS[name]]
        if f == 1:
            continue
        t = doc.d[o + 3]
        at = lo + o + 4
        if t == 0xD2:
            v = struct.unpack_from('<f', buf, at)[0]
            if name == 'ForbidPushJobDistance' and (v <= 0 or v < push_min):
                continue
            nv = max(v * f, push_min) if name == 'ForbidPushJobDistance' else v * f
            struct.pack_into('<f', buf, at, nv)
        elif t == 0xD1:
            v = struct.unpack_from('<i', buf, at)[0]
            if name == 'ForbidPushJobDistance' and (v <= 0 or v < push_min):
                continue
            nv = int(round(max(v * f, push_min) if name == 'ForbidPushJobDistance' else v * f))
            struct.pack_into('<i', buf, at, nv)
        else:
            continue
        stats[name] = stats.get(name, 0) + 1
        changed += 1
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--romfs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='the mod folder, e.g. <install>/mods/LessActorWork')
    ap.add_argument('--near', type=float, default=0.5)
    ap.add_argument('--middle', type=float, default=1.0)
    ap.add_argument('--push', type=float, default=0.5)
    ap.add_argument('--push-min', type=float, default=100)
    ap.add_argument('--level', type=int, default=9, help='zstd level of the written packs')
    a = ap.parse_args()
    scale = {'near': a.near, 'middle': a.middle, 'push': a.push}
    files = {f['path']: f for f in json.loads((a.romfs / 'index.json').read_text(encoding='utf-8'))['files']}
    img = (a.romfs / 'romfs.bin').open('rb')

    def read(path):
        f = files[path]
        img.seek(f['offset'])
        return img.read(f['size'])

    dicts = {}
    for _, b in fp.sarc_members(fp.zdec(read('Pack/ZsDic.pack.zs'))).items():
        d = fp.zdict(b)
        dicts[fp.zdict_id(d)] = d
    packs = sorted(p for p in files if p.startswith('Pack/Actor/') and p.endswith('.pack.zs'))
    dest = a.out / 'romfs'
    stats, written = {}, 0
    for i, path in enumerate(packs):
        if i and i % 2000 == 0:
            print(f'  {i} of {len(packs)} packs, {written} changed', flush=True)
        raw = read(path)
        did = fp.zframe_dict(raw)
        buf = bytearray(fp.zdec(raw, dicts.get(did)) if did else fp.zdec(raw))
        if buf[:4] != b'SARC' or buf[6:8] != b'\xff\xfe':
            continue
        changed = 0
        for name, lo, hi in list(sarc_spans(bytes(buf))):
            if name.startswith('Component/LODParam/') and name.endswith('.bgyml'):
                changed += patch_lod(buf, lo, hi, scale, a.push_min, stats)
        if not changed:
            continue
        out = dest / path
        out.parent.mkdir(parents=True, exist_ok=True)
        data = zcomp(bytes(buf), dicts.get(did), a.level) if did else zcomp(bytes(buf), None, a.level)
        if fp.zframe_dict(data) != did or len(fp.zdec(data, dicts.get(did))) != len(buf):
            sys.exit(f'make_lod_mod: {path} did not compress back with its dictionary')
        out.write_bytes(data)
        written += 1
    print(f'{written} of {len(packs)} actor packs changed into {dest}; values changed: '
          + ', '.join(f'{k} {v}' for k, v in sorted(stats.items())))
    return 0 if written else 1


if __name__ == '__main__':
    sys.exit(main())
