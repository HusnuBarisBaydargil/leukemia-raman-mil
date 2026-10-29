"""Preprocess the downloaded Vanna dataset into patch bags."""

import argparse
import sys
from pathlib import Path

from src.data import ROOT, load_config, prepare_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Folder containing the dataset .tar or .feather files.")
    parser.add_argument("--config", type=Path, help="Optional JSON configuration; defaults to config.json.")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        source_path = args.data_dir or ROOT / config["paths"]["raw_data"]
        prepare_data(
            source_path,
            ROOT / config["paths"]["cache"],
            ROOT / "data/manifest.csv",
            config["preprocessing"],
        )
    except (Exception, KeyboardInterrupt) as error:
        was_interrupted = isinstance(error, KeyboardInterrupt)
        print(
            "Preprocessing interrupted." if was_interrupted else f"Preprocessing failed: {error}",
            file=sys.stderr,
        )
        for note in getattr(error, "__notes__", []):
            print(note, file=sys.stderr)
        return 130 if was_interrupted else 1
    print("Done. Run: python train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
