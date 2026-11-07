"""Make a symbol ordering file (hottest first) for TOTK_ORDER_FILE (tools/build_game.py) from perf reports.

Usage: hot_order.py <report.txt>... -o order.txt

Each report is `perf report --no-children --sort sym -g none --percent-limit 0 --stdio` of a profile of the game,
whose lines read "  0.71%  [.] f_7000981248" (a shared-object column, as --sort dso,sym prints, is allowed;
other objects than totk are skipped). The shares of a symbol are summed over the reports, so profiles of
different scenes (standing, moving, towns) all count. Only user-space symbols of the program are kept; names perf
could not resolve (0x...) are skipped. lld ignores names it does not find, so runtime symbols do no harm.
"""
import argparse
import re
import sys
from collections import defaultdict

LINE = re.compile(r'^\s*([0-9.]+)%\s+(?:(\S+)\s+)?\[\.\]\s+(\S+)')   # with or without a shared-object column


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('reports', nargs='+')
    ap.add_argument('-o', '--out', required=True)
    a = ap.parse_args()
    weight = defaultdict(float)
    for path in a.reports:
        with open(path, errors='replace') as f:
            for line in f:
                m = LINE.match(line)
                if m and not m.group(3).startswith('0x') and m.group(2) in (None, 'totk'):
                    weight[m.group(3)] += float(m.group(1))
    names = sorted(weight, key=lambda n: -weight[n])
    lifted = [n for n in names if n.startswith('f_')]
    with open(a.out, 'w') as f:
        f.write(''.join(n + '\n' for n in names))
    total = sum(weight[n] for n in lifted)
    top = sum(weight[n] for n in lifted[:2000])
    print(f'{len(names)} symbols ({len(lifted)} lifted functions) from {len(a.reports)} reports; '
          f'the first 2000 lifted functions hold {100 * top / total if total else 0:.0f} % of the lifted samples')
    return 0 if lifted else 1


if __name__ == '__main__':
    sys.exit(main())
