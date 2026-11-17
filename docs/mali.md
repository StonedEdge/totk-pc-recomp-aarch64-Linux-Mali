# TotK Recomp on Mali-G610 (RK3588), aarch64 Linux

The `mali` branch of this fork runs TotK Recomp on an Arm handheld: a Retro Lite CM5 (Rockchip RK3588, Mali-G610 MP4,
4 Cortex-A76 + 4 Cortex-A55). The `mali-040` branch is upstream 0.4.0 plus upstream's `dev` of 2026-11-14 with the changes below; `mali` has the same work on 0.3.4. Everything else in this repository's
documentation still applies; this page is only what differs on this hardware.

> [!NOTE]
> As upstream, no game files are included. You need your own copy of the game (1.2.1 is the version tested here),
> extracted and recompiled on your machine by the setup.

## Tested setup

| | |
|---|---|
| Board | Retro Lite CM5: RK3588, Mali-G610, 8 GB RAM (+ zram) |
| OS | Ubuntu 24.04 aarch64 (Armbian base), kernel 6.1.118, Wayland desktop |
| GPU driver | Arm's Mali Vulkan driver (the "blob", g29p1). PanVK (Mesa) is not tested |
| Game | 1.2.1 |
| Compiler | clang 20 (`/usr/lib/llvm-20`): the runtime uses `#embed` |

Kernel 6.1 has no userfaultfd write protection (upstream asks for 6.7 or later), so the runtime falls back to a
software write watch there.

## How it runs

- 720p, render scale 75 % with FSR, FXAA off: **30 fps in light views**, about **28-29 fps moving through woods and
  grassy hillsides** (4,000-5,600 draws a frame).
- While moving, the limit is the CPU: the game's own threads on the four A76 cores. The GPU is not the limit.
- With the LessActorWork mod below, the 1 % low while moving through woods went from about 11-16 fps to 18-21 fps
  and frames over 50 ms fell to about a third.
- Memory: 4.4-5.6 GB resident. Close other apps.

## Settings

Set these in the environment before starting `totk` (they win over `totk.ini`):

| Setting | What it does |
|---|---|
| `SDL_VIDEO_DRIVER=wayland` | Native Wayland window |
| `TOTK_SYNC_AHEAD=1` | The game's per-frame GPU wait waits one frame older (two frames in flight); upstream needs shadow storage for it, here the Mali's imported memory is allowed too |
| `TOTK_SPLIT_DRAWS=64`, `TOTK_SPLIT_US=1000` | Submission part size |

Render scale, upscaler and FXAA are in the in-game menu (`totk.ini`: `render_scale`, `upscaler`, `fxaa`). The Guide
button, Minus + Plus, or F1 opens the menu; L3 + R3 shows the built-in performance overlay.

These are on by default on Mali or on CPUs with cores of two speeds, and are listed so they can be switched off to
compare:

| Default | Switch | What it does |
|---|---|---|
| Mali | `TOTK_MEMORY=import`, `TOTK_SHADOW` | Guest memory imported by the GPU instead of shadow copies |
| Mali (tilers) | `TOTK_CLEAR_LOAD` | Whole-target clears become the next pass's load operation |
| Mali (tilers) | `TOTK_BARRIER_NVN` | The game's fragment-ordering barriers become fragment-only |
| Mali (tilers) | `TOTK_UB_INPLACE`, `TOTK_UB_WINDOW` | Uniform updates kept inside render passes, as versions; a window twin for the terrain's uniform block (`TOTK_UB_WINDOW=0` if the terrain ever flickers) |
| Mali | `TOTK_FEEDBACK=restart` | Brings back pass restarts for feedback loops |
| two-speed CPUs | `TOTK_BIG_CORES=0` | The game kept on the fast cores |
| two-speed CPUs (aarch64) | `TOTK_SLOW_BG=0` | Background threads moved to the slow cores |
| two-speed CPUs (aarch64) | `TOTK_REC_SPIN_US` (20) | How long the command recorder spins before it sleeps (200 elsewhere) |
| all | `TOTK_PACE_SPIN_US=0` | Frame pacing sleeps on the timer instead of yielding while it waits |
| two-speed CPUs (aarch64) | `TOTK_REC_ASYNC=1` | Upstream's: submissions are handed to the recorder without waiting for it; froze the GPU queue on the CM5 within minutes |

Vertex shaders that write storage buffers run as compute shaders where the driver lacks
`vertexPipelineStoresAndAtomics` (the Mali blob): without it Ultra Hand cannot grab anything.

## The LessActorWork mod (optional)

`tools/make_lod_mod.py` makes a RomFS mod from your own game files that shortens each actor's LOD distances, so the
game updates fewer actors at full rate. The trade-off: actors 15-50 m away update less often, and ones past 250 m
(400 m for some) stop updating.

```sh
python3 tools/make_lod_mod.py --romfs build/extracted/<version>/romfs --out <install>/mods/LessActorWork \
    --near 0.5 --middle 0.6 --push 0.5
```

It needs Python with `compression.zstd` (3.14) or the `zstandard` package. The mod loads at the next start; switch it
off in the menu's Mods tab, with `TOTK_MODS_DISABLED=LessActorWork`, or by deleting its folder.
`tools/find_params.py` lists the game's other distance and detail parameters.

## Profile-guided optimisation (PGO)

Upstream's PGO build mode (`tools/build_game.py --pgo`, `tools/pgo.py`, [pgo.md](pgo.md)) works here too: an
instrumented build is played for a while, its profile merged, and the game code compiled again with it, so hot paths
are laid out together and branches predicted from real play. On Linux the clang profile runtime comes with
`libclang-rt-20-dev` and `llvm-profdata` with `llvm-20` (`--toolchain /usr/lib/llvm-20`); the instrumented game
writes its counts every minute and when it is closed from its menu. Every setting of the fork is in [settings.md](settings.md).

## Diagnostics

- `TOTK_TRACE=perf,sched`: per-frame times, draw counts, waits and the scheduler's figures in the log.
- `TOTK_PASS_TIME`: GPU time of every pass and dispatch. `TOTK_CRIT=/<file>`: records the critical path while the
  file exists.
- `tools/hot_order.py`, `tools/locals_sweep.py`: code layout and code-generation experiments (`TOTK_ORDER_FILE`,
  `TOTK_GEN_OPT`).

## Known issues

- Subtitles can lose letters (the R8 font atlas).
- Some texture aliasing fails (4 KB alignment).
- Only Great Sky Island and the woods around it have been played much; towns and the surface are less tested.

## Thread priority

`TOTK_CRIT_PRIO=1` (off by default: no gain measured on the RK3588, 26.6 against 26.8 fps) raises the threads the frame waits on above the game's job workers, but
Linux only lets a user raise a thread's priority as far as the `nice` limit allows. Once, then log out and in again:

```
echo "$USER - nice -10" | sudo tee /etc/security/limits.d/totk.conf
```

The log then says "runs at a higher priority than the job workers" for the main, render, replay and recorder threads.

## Texture memory limit

The Mali driver reports nearly all of system memory as its device-local budget (6.3 GB of the CM5's 7.8 GB), so
`TOTK_TEX_VRAM_PRESSURE` defaults to 15 % there instead of 90 %. With 90 %, released textures piled up after a death or
a save load (VMA 483 -> 913 MiB), the system went to swap and the driver's recording slowed: ~26 fps where a fresh
start held 30. The "Texture memory limit" slider overrides it.

## Pool imports

The game keeps creating small CPU-visible memory pools as it streams the world (about 1,300 after loading, 2,800 after
20 minutes). Imported one by one, each was a memory object of its own in the driver, and the recorder's time in one
driver routine grew from 97 to 251 ms/s over such a session while the rest of its work stayed flat: frame preparation
slowed the longer one played. `TOTK_IMPORT_CHUNK_MIB` (2 by default on Mali) makes the pools share imports of 2 MiB
windows of guest memory instead (vk_memory.c, chunk_get); the log line "pools imported into chunks" counts them, and
"import window ... refused" names windows that fell back to one import per pool. `0` restores the old behaviour.

## Stutter from replay drains

With `TOTK_REPLAY_ASYNC=1` some calls wait until the replay thread has caught up with everything queued. Texture
releases queued behind the replay were counted as presents, so the one-frame bound never held and the replay fell
several frames behind; over 2.5 minutes on the CM5 the game's threads then waited 5.5 s in 479 WriteTexels drains and
3.1 s in 109 pool finalizes, 12-28 ms each: frames of 50-70 ms several times a second. Holding the bound of one frame
lowered the frame rate (the game waited for the replay every frame), so the bound is now `TOTK_REPLAY_FRAMES` and off
by default, and WriteTexels queues its image write instead of draining (`TOTK_WT_QUEUE`, on by default). The log's
"replay queue" lines count the remaining waits by cause.

