"""Separate a track into a Native Instruments stem file (.stem.mp4) that Mixxx's stem decks play.

    python mixxx_stems.py INPUT [-o OUTPUT.stem.mp4] [--models-dir DIR]

Needs Python with audio-separator, audioread, numpy and soundfile, plus ffmpeg on PATH. On an AMD card install
AMD's ROCm PyTorch build first (https://repo.amd.com/rocm/whl/gfx120X-all/ for RDNA4), otherwise it runs on the CPU.

Stems (chosen on MUSDB18-HQ excerpts, mik-tempo-research/stem_bench.py):
  vocals   BS-RoFormer Revive V2 by unwa (best vocal SDR of the open models tested)
  drums    BS-Roformer-SW by jarredou (6-stem model)
  bass     BS-Roformer-SW
  other    mix - vocals - drums - bass, so the four stems sum back to the original
The file holds the mix plus the four stems as AAC, the source file's tags, and the stem manifest atom
(moov/udta/stem) Mixxx reads. The output path is printed as the last line.
"""
import argparse
import json
import os
import pathlib
import re
import struct
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 44100
VOCALS_MODEL = "bs_roformer_vocals_revive_v2_unwa.ckpt"
SW_MODEL = "BS-Roformer-SW.ckpt"
STEMS = [("Drums", "#009E73"), ("Bass", "#D55E00"), ("Other", "#CC79A7"), ("Vocals", "#56B4E9")]  # NI order


def manifest():
    return json.dumps({"version": 1, "mastering_dsp": {"compressor": {"enabled": False}, "limiter": {"enabled": False}},
                       "stems": [{"name": n, "color": c} for n, c in STEMS]}).encode()


def add_stem_atom(path, payload):
    """Append moov/udta/stem to an MP4 whose moov box is the last top-level box (no chunk offsets move)."""
    data = bytearray(pathlib.Path(path).read_bytes())
    pos, moov = 0, None
    while pos < len(data):
        size, kind = struct.unpack(">I4s", data[pos:pos + 8])
        if size < 8:
            raise ValueError(f"unsupported box size {size} at {pos}")
        if kind == b"moov":
            moov = pos
        pos += size
    if moov is None or moov + struct.unpack(">I", data[moov:moov + 4])[0] != len(data):
        raise ValueError("moov must be the last box")
    stem = struct.pack(">I4s", 8 + len(payload), b"stem") + payload
    moov_size = struct.unpack(">I", data[moov:moov + 4])[0]
    child, udta = moov + 8, None
    while child < moov + moov_size:
        size, kind = struct.unpack(">I4s", data[child:child + 8])
        if kind == b"udta":
            udta = child
            break
        child += size
    if udta is not None and udta + struct.unpack(">I", data[udta:udta + 4])[0] == len(data):
        data[udta:udta + 4] = struct.pack(">I", struct.unpack(">I", data[udta:udta + 4])[0] + len(stem))
        data += stem
    else:  # new udta at the end of moov
        stem = struct.pack(">I4s", 8 + len(stem), b"udta") + stem
        data += stem
    data[moov:moov + 4] = struct.pack(">I", moov_size + len(stem))
    pathlib.Path(path).write_bytes(bytes(data))


def pin_gpu():
    """ROCm on Windows also lists the Ryzen iGPU; with it visible every kernel fails. Keep the device with the most CUs."""
    if "HIP_VISIBLE_DEVICES" in os.environ:
        return
    probe = ("import torch; n = torch.cuda.device_count() if torch.version.hip else 0; "
             "print(max(range(n), key=lambda d: torch.cuda.get_device_properties(d).multi_processor_count) if n > 1 else '')")
    best = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True).stdout.strip()
    if best:
        os.environ["HIP_VISIBLE_DEVICES"] = best


def load(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "2", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2)


def separate(model, path, out_dir, models_dir):
    """-> {stem name (lower case): stereo array} from one audio-separator model."""
    from audio_separator.separator import Separator
    sep = Separator(output_dir=str(out_dir), model_file_dir=str(models_dir), log_level=40, normalization_threshold=1.0)
    sep.load_model(model_filename=model)
    stems = {}
    for f in sep.separate(str(path)):
        f = pathlib.Path(f) if os.path.isabs(f) else pathlib.Path(out_dir) / f
        m = re.search(r"_\(([^)]+)\)_", f.name)
        if m:
            stems[m[1].lower()] = sf.read(f, dtype="float32", always_2d=True)[0]
    return stems


def export(src, dst, models_dir):
    pin_gpu()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        wav = tmp / "mix.wav"
        mix = load(src)
        sf.write(wav, mix, SR, subtype="FLOAT")
        vocals = separate(VOCALS_MODEL, wav, tmp / "voc", models_dir)["vocals"]
        sw = separate(SW_MODEL, wav, tmp / "sw", models_dir)
        n = min(len(mix), len(vocals), len(sw["drums"]), len(sw["bass"]))
        parts = {"Drums": sw["drums"][:n], "Bass": sw["bass"][:n], "Vocals": vocals[:n]}
        parts["Other"] = mix[:n] - parts["Vocals"] - parts["Drums"] - parts["Bass"]
        inputs = [tmp / "0.wav"]
        sf.write(inputs[0], mix[:n], SR, subtype="FLOAT")
        for i, (name, _) in enumerate(STEMS, 1):
            inputs.append(tmp / f"{i}.wav")
            sf.write(inputs[-1], parts[name], SR, subtype="FLOAT")
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src)]
        for p in inputs:
            cmd += ["-i", str(p)]
        cmd += ["-map_metadata", "0"]
        for i in range(1, len(inputs) + 1):
            cmd += ["-map", f"{i}:a"]
        out = tmp / "out.mp4"
        cmd += ["-c:a", "aac", "-b:a", "256k", "-f", "mp4", str(out)]
        subprocess.run(cmd, check=True)
        add_stem_atom(out, manifest())
        os.replace(out, dst)


def demo():
    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "t.mp4"
        mdat = struct.pack(">I4s", 12, b"mdat") + b"abcd"
        moov = struct.pack(">I4s", 16, b"moov") + struct.pack(">I4s", 8, b"mvhd")
        p.write_bytes(struct.pack(">I4s", 8, b"ftyp") + mdat + moov)
        add_stem_atom(p, b'{"version":1}')
        d = p.read_bytes()
        assert d.endswith(b"stem" + b'{"version":1}') and struct.unpack(">I", d[20:24])[0] == len(d) - 20  # moov grew
        assert d[40:44] == b"udta" and struct.unpack(">I", d[36:40])[0] == len(d) - 36
        add_stem_atom(p, b"x")  # appends into the existing trailing udta
        d2 = p.read_bytes()
        assert struct.unpack(">I", d2[36:40])[0] == len(d2) - 36 and struct.unpack(">I", d2[20:24])[0] == len(d2) - 20
    assert json.loads(manifest())["stems"][3]["name"] == "Vocals"
    print("mixxx_stems self-check ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        demo()
        raise SystemExit
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--models-dir", default=str(pathlib.Path.home() / ".cache" / "mixxx-stems" / "models"))
    args = ap.parse_args()
    src = pathlib.Path(args.input)
    dst = pathlib.Path(args.output) if args.output else src.with_name(src.stem + ".stem.mp4")
    export(src, dst, args.models_dir)
    print(dst)
