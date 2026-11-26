"""Build a verified, memory-mappable 64x64 RGB cache from a Folsom tar.bz2 archive."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from io import BytesIO
from itertools import islice
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

from PIL import Image


def members(source):
    seen = set()
    for entry in source:
        suffix = PurePosixPath(entry.name).suffix.lower()
        if not entry.isfile() or suffix not in {".jpg", ".jpeg"}:
            continue
        if entry.name in seen:
            raise ValueError(f"duplicate archive member: {entry.name}")
        seen.add(entry.name)
        with source.extractfile(entry) as stream:
            yield entry.name, int(entry.mtime), stream.read()


def resize(item):
    name, mtime, payload = item
    try:
        with Image.open(BytesIO(payload)) as image:
            rgb = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
            return name, mtime, rgb.tobytes(), ""
    except OSError as error:
        return name, mtime, None, str(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--decoder", default="bzip2")
    parser.add_argument("--decoder-option", action="append", default=[])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    command = [args.decoder, "-dc", *args.decoder_option, str(args.archive)]
    count = rejected = 0
    decoder = subprocess.Popen(command, stdout=subprocess.PIPE)
    try:
        with (tarfile.open(fileobj=decoder.stdout, mode="r|") as archive,
              (args.output / "rgb.bin").open("wb") as binary,
              (args.output / "frames.csv").open("w", newline="") as frame_file,
              (args.output / "unreadable.csv").open("w", newline="") as bad_file,
              ThreadPoolExecutor(max_workers=args.workers) as pool):
            frame_rows, bad_rows = csv.writer(frame_file), csv.writer(bad_file)
            frame_rows.writerow(["path", "mtime"])
            bad_rows.writerow(["path", "mtime", "error"])
            stream = members(archive)
            while batch := list(islice(stream, 64)):
                for name, mtime, rgb, error in pool.map(resize, batch):
                    if rgb is None:
                        bad_rows.writerow([name, mtime, error])
                        rejected += 1
                    else:
                        binary.write(rgb)
                        frame_rows.writerow([name, mtime])
                        count += 1
                if count and count % 10000 < 64:
                    print(f"{args.archive.name}: {count} readable, {rejected} unreadable", flush=True)
            while decoder.stdout.read(2**20):
                pass
    finally:
        decoder.stdout.close()
    if decoder.wait() != 0 or count == 0:
        raise RuntimeError("archive decoder failed or produced no readable frames")
    expected = count * 64 * 64 * 3
    observed = (args.output / "rgb.bin").stat().st_size
    if observed != expected:
        raise RuntimeError(f"RGB cache length {observed} != {expected}")
    summary = {"archive": str(args.archive), "archive_bytes": args.archive.stat().st_size,
               "readable_frames": count, "unreadable_frames": rejected,
               "shape": [count, 64, 64, 3], "dtype": "uint8", "decoder": command}
    temporary = args.output / "complete.tmp"
    temporary.write_text(json.dumps(summary, indent=2))
    os.replace(temporary, args.output / "complete.json")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

