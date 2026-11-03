# Personal Mixxx build: analysis tuned against Mixed In Key

A personal fork of Mixxx (branch `personal`: `bpm-best` plus `qm-octave`, based on upstream `main` 44dc9ed7e8) with different
analysis algorithms, an energy rating, stem separation, and all results written to file tags. The
reference throughout is Mixed In Key 11, and the evaluation scripts are in `../mik-tempo-research`.

## What changed

| Area | This build | Stock Mixxx | Measured |
|---|---|---|---|
| BPM | Beat This! neural beat tracker (ONNX Runtime); Queen Mary as fallback, with a bass-band fix for 2/3 errors and a metrical-level fix for half/double tempo; beatless gaps can no longer set the tempo | Queen Mary | GiantSteps EDM exact tempo 81.9% vs 35.7%; GTZAN 82.9% vs 71.2%; no results below 40 BPM on GiantSteps (was 14) |
| Tempo changes | Tempo Map: steady sections, e.g. `0:00 -, 0:35 120.00, 2:10 140.00` ("-" = no steady beat) | none | unit tested |
| Key | OpenKeyScan 3 (KeyNet CNN, ONNX); falls back to KeyFinder, then Queen Mary | Queen Mary | exact match with MIK on 298 tracks: 60.1% vs 36% (KeyFinder 43.6%) |
| Energy | New 1–10 rating (7 audio features, 150 boosted trees fitted to MIK) | none | vs MIK on 2218 tracks (5-fold CV): 53.5% exact, 92.9% within ±1 |
| Stems | Track menu → Analyze → Separate Stems writes an NI `.stem.mp4` (drums, bass, other, vocals) and adds it to the library | playback only | vocals 14.6 dB SDR on a MUSDB18-HQ excerpt |
| Hot cues | On tracks with no hot cues, up to 8 are set at section changes on the beat grid (loudness and bass, 16 beats either side, at least 8 bars apart), labelled Start / Drop / Break / Phrase. Off with `AutoHotcues=0` under `[Library]` in `mixxx.cfg` | none | unit tested on synthetic levels; not yet scored against real cues |
| Display | LateNight "OLED" colour scheme (Preferences → Interface): true black backgrounds, white text, and neutral white icons, blue instead of orange for active buttons, and black Preferences/dialogs with dark title bars. `style_oled.qss` and the `oled/` icon set are generated from PaleMoon by `tools/oled_scheme.py`. Text uses full font hinting, so it stays sharp at fractional scaling (e.g. 150 %) | grey backgrounds, ivory text, light dialogs, soft text at 150 % | not tested visually by the author |

The library gets an Energy column (editable, sorted as a number) and a Tempo Map column (hidden by
default); the database moves to schema 41.

At most 4 ONNX model runs happen at once, and ONNX Runtime's memory pool is off: analyzing 26 tracks
peaked at 4.4 GB (15.9 GB before) and memory is released afterwards.

## Tags

| | MP3 (ID3v2) | FLAC/Ogg | MP4/M4A |
|---|---|---|---|
| BPM | `TBPM` | `BPM` | `tmpo`, `----:com.apple.iTunes:BPM` |
| Key | `TKEY` | `INITIALKEY` | `----:com.apple.iTunes:initialkey` |
| Energy | `TXXX:EnergyLevel` | `ENERGYLEVEL` | `----:com.apple.iTunes:EnergyLevel` |
| Tempo map | `TXXX:TempoMap` | `TEMPOMAP` | `----:com.apple.iTunes:TempoMap` |

The energy fields use MIK's names: energy that MIK already wrote is imported and kept. Tags are only
written when Preferences → Library → "Synchronize library track metadata to file tags" is on.

## Setup

- Build: `build_mixxx.bat --target mixxx mixxx-test` (needs `buildenv/` and ONNX Runtime 1.22).
- Models are not committed. Put them in `res/models/`:
  - `beatthis_final0.onnx`, from `mik-tempo-research/export_onnx.py`
  - `openkeyscan3.onnx`, from `mik-tempo-research/export_openkeyscan_onnx.py`
  Without them, analysis falls back to Queen Mary (BPM) and KeyFinder (key).
- Stems need a Python with `audio-separator`, `audioread`, `soundfile` and `numpy`, plus ffmpeg on PATH.
  For an AMD GPU, install PyTorch from `https://repo.amd.com/rocm/whl/gfx120X-all/` first. Point
  Mixxx at that interpreter with `Python` in the `[Stems]` section of `mixxx.cfg`. Separation runs
  slightly slower than real time on an RX 9070 XT.
- Back up `mixxxdb.sqlite` before the first run (schema 40 → 41).

## Known test failures

1311 of 1318 tests pass. The 7 failures are all in code this fork does not touch: 4 controller
script timer tests that are timing-sensitive on this machine, 2 QML tests, and
`TrackMetadataExportTest.keepWithespaceKey`. The last one fails only after
`ControllerScriptEngineLegacyTest` runs in the same process, and passes on its own.
