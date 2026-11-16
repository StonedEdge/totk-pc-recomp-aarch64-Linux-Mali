#!/bin/bash
# Profile-guided build of TotK Recomp on an aarch64 Linux handheld (docs/pgo.md, docs/mali.md), run from the repository.
#
#   bash tools/linux/totk-pgo.sh setup   <install>   llvm-profdata and the clang profile runtime of the installed clang 20
#   bash tools/linux/totk-pgo.sh gen     <install>   instrumented build in build/pgo-gen, put into the install
#   (play the instrumented game for a while: shrines, combat, the sky, the surface; close it from its menu)
#   bash tools/linux/totk-pgo.sh merge   <install>   build/pgo/pgo-v121.profdata from the counts
#   bash tools/linux/totk-pgo.sh final   <install>   the game compiled with that profile, put into the install
#   bash tools/linux/totk-pgo.sh restore <install>   the build from before `gen` back into the install
#
# setup downloads Ubuntu's llvm-20 and libclang-rt-20-dev packages of exactly the clang that is installed, unpacks them
# into build/llvm-pgo (no apt), and copies the one profile runtime file into clang's resource folder with sudo: clang
# links it from there for -fprofile-generate and has no other place to look.
set -euo pipefail
cmd=${1:-}; dest=${2:-}
[ -n "$cmd" ] && [ -n "$dest" ] || { sed -n '2,15p' "$0"; exit 2; }
dest=$(realpath "$dest")
cd "$(dirname "$0")/../.."
root=$PWD
llvm=/usr/lib/llvm-20
py=.venv-py314/bin/python
[ -x "$py" ] || py=python3
export PATH=$llvm/bin:$PATH LIBRARY_PATH=/usr/local/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
temp=$(ls -d build/install-temp/totk-* 2>/dev/null | head -1)
ver=v121

build_args() {   # build_game.py's arguments as the installer passes them (installer/steps.py compile)
    [ -n "$temp" ] && [ -d "$temp/gen" ] || { echo "no lifted code in build/install-temp: run the installer first"; exit 1; }
    echo --gen "$temp/gen" --toolchain "$llvm" --runtime runtime --ninja "$(command -v ninja)" --version $ver \
        --exefs "$dest/exefs" --romfs "$dest/romfs" --profile "$dest/profile"
}

case $cmd in
setup)
    v=$(dpkg-query -W -f '${Version}' clang-20 2>/dev/null || true)
    [ -n "$v" ] || { echo "clang-20 is not a dpkg package here: tell Claude what clang --version prints"; exit 1; }
    v=${v#*:}   # file names carry no epoch
    echo "clang-20 $v"
    mkdir -p build/llvm-pgo/deb
    for p in llvm-20 libclang-rt-20-dev; do
        f=build/llvm-pgo/deb/${p}_${v}_arm64.deb
        [ -s "$f" ] || wget -q --show-progress -O "$f" \
            "http://ports.ubuntu.com/ubuntu-ports/pool/universe/l/llvm-toolchain-20/${p}_${v}_arm64.deb"
        dpkg-deb -x "$f" build/llvm-pgo
    done
    tool=build/llvm-pgo$llvm/bin/llvm-profdata
    "$tool" --version | head -2
    rt=$(cd build/llvm-pgo && find .$llvm/lib/clang -name 'libclang_rt.profile*.a' | head -1)
    [ -n "$rt" ] || { echo "no profile runtime in libclang-rt-20-dev"; exit 1; }
    echo "copying ${rt#.} (clang's profile runtime) into place:"
    sudo install -D -m 644 "build/llvm-pgo/$rt" "${rt#.}"
    echo "setup done"
    ;;
gen)
    [ -e "$dest/totk-plain" ] || cp "$dest/totk" "$dest/totk-plain"
    $py tools/build_game.py $(build_args) --out build/pgo-gen --pgo gen
    cp build/pgo-gen/totk "$dest/totk"
    echo "the instrumented build is in $dest (the plain one is totk-plain). Play it, then close it from its menu."
    echo "Its counts go to $root/build/pgo-gen/pgo/totk-<pid>.profraw every minute (one file a run)."
    ;;
merge)
    ls build/pgo-gen/pgo/*.profraw >/dev/null 2>&1 || { echo "no counts in build/pgo-gen/pgo: play the instrumented build first"; exit 1; }
    ls -la build/pgo-gen/pgo/
    $py tools/pgo.py merge build/pgo-gen --version $ver --toolchain build/llvm-pgo$llvm
    ls -la build/pgo/
    ;;
final)
    [ -s build/pgo/pgo-$ver.profdata ] || { echo "no build/pgo/pgo-$ver.profdata: run merge first"; exit 1; }
    $py tools/build_game.py $(build_args) --out build/pgo-use --pgo "$root/build/pgo/pgo-$ver.profdata"
    cp build/pgo-use/totk "$dest/totk"
    echo "the profile-guided build is in $dest (the plain one is totk-plain)"
    ;;
restore)
    [ -e "$dest/totk-plain" ] || { echo "no totk-plain in $dest"; exit 1; }
    cp "$dest/totk-plain" "$dest/totk"
    echo "the plain build is back in $dest"
    ;;
*) sed -n '2,15p' "$0"; exit 2 ;;
esac
