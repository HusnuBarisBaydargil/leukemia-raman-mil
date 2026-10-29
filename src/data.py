"""Raman-map loading and the fixed release-view patch-bag preprocessing."""

import csv
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    return json.loads(Path(path or ROOT / "config.json").read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def iter_maps(source_path, selected_names=None):
    """Read source Feather maps or tar members without extracting archives."""
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset/input not found: {source_path}")
    source_paths = sorted(source_path.rglob("*")) if source_path.is_dir() else [source_path]
    seen_files = set()
    for file_path in source_paths:
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() == ".feather":
            map_members = [(file_path.name, None)]
            archive = None
        elif file_path.suffix.lower() == ".tar":
            archive = tarfile.open(file_path, "r")
            map_members = [
                (Path(archive_entry.name).name, archive_entry)
                for archive_entry in archive.getmembers()
                if archive_entry.isfile() and archive_entry.name.lower().endswith(".feather")
            ]
        else:
            continue
        try:
            for file_name, archive_member in map_members:
                if selected_names is not None and file_name not in selected_names:
                    continue
                if file_name in seen_files:
                    raise ValueError(f"Duplicate map {file_name}; use archives or extracted maps, not both.")
                seen_files.add(file_name)
                if archive is None:
                    table = feather.read_table(file_path)
                else:
                    with archive.extractfile(archive_member) as map_stream:
                        table = feather.read_table(io.BytesIO(map_stream.read()))
                yield file_name, table
        finally:
            if archive is not None:
                archive.close()
    if not seen_files:
        raise ValueError("No matching .feather maps were found in the input.")


def make_bag(map_table, preprocessing):
    """Keep released intensities; average non-overlapping spatial patches."""
    height, width = preprocessing["map_shape"]
    spectral_columns = sorted(
        (column_name for column_name in map_table.column_names if column_name not in {"x_pos", "y_pos"}),
        key=float,
    )
    if len(spectral_columns) != preprocessing["channels"] or map_table.num_rows != height * width:
        raise ValueError(f"Expected {height}x{width} pixels and {preprocessing['channels']} channels.")
    x_coordinates = map_table["x_pos"].combine_chunks().to_numpy(zero_copy_only=False)
    y_coordinates = map_table["y_pos"].combine_chunks().to_numpy(zero_copy_only=False)
    if not np.isfinite(x_coordinates).all() or not np.isfinite(y_coordinates).all():
        raise ValueError("Nonfinite map coordinates.")
    unique_x_coordinates, x_indices = np.unique(x_coordinates, return_inverse=True)
    unique_y_coordinates, y_indices = np.unique(y_coordinates, return_inverse=True)
    pixel_indices = y_indices * width + x_indices
    if (
        len(unique_x_coordinates) != width
        or len(unique_y_coordinates) != height
        or len(np.unique(pixel_indices)) != height * width
    ):
        raise ValueError("Incomplete or duplicate spatial coordinates.")
    map_spectra = np.empty((height, width, len(spectral_columns)), dtype=np.float32)
    for channel_index, column_name in enumerate(spectral_columns):
        map_spectra[y_indices, x_indices, channel_index] = (
            map_table[column_name].combine_chunks().to_numpy(zero_copy_only=False)
        )
    if not np.isfinite(map_spectra).all():
        raise ValueError("Nonfinite spectra.")
    patch_size = preprocessing["patch_size"]
    if height % patch_size or width % patch_size:
        raise ValueError("Map dimensions must be divisible by patch_size.")
    patch_grid = map_spectra.reshape(
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
        len(spectral_columns),
    )
    patch_means = patch_grid.mean(axis=(1, 3), dtype=np.float64)
    patch_bag = patch_means.reshape(-1, len(spectral_columns)).astype(np.float32)
    return patch_bag


def prepare_data(source_path, cache_dir, manifest_path, preprocessing):
    manifest_rows = read_csv(manifest_path)
    manifest_by_name = {row["file_name"]: row for row in manifest_rows}
    if len(manifest_by_name) != len(manifest_rows):
        raise ValueError("Duplicate filenames in manifest.")
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        raise FileExistsError(
            f"Cache folder already exists: {cache_dir.resolve()}. "
            "It may contain a completed or partial cache. "
            "Existing files were left unchanged. To rebuild, rename this folder as a backup or set "
            "paths.cache in config.json to a new folder, then rerun preprocessing."
        )
    if not Path(source_path).exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")
    cache_dir.mkdir(parents=True)
    processed_files = set()
    cache_index_rows = []
    try:
        for file_name, table in iter_maps(source_path, set(manifest_by_name)):
            bag = make_bag(table, preprocessing)
            cache_file_name = Path(file_name).stem + ".npz"
            np.savez_compressed(cache_dir / cache_file_name, bag=bag)
            cache_index_rows.append({**manifest_by_name[file_name], "cache_file": cache_file_name})
            processed_files.add(file_name)
            print(f"Preprocessed {len(processed_files)}/{len(manifest_rows)}: {file_name}", flush=True)
        missing_files = set(manifest_by_name) - processed_files
        if missing_files:
            raise ValueError(f"Missing {len(missing_files)} maps, including {sorted(missing_files)[:3]}")
        write_csv(cache_dir / "index.csv", sorted(cache_index_rows, key=lambda row: row["file_name"]))
        (cache_dir / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    except (Exception, KeyboardInterrupt) as error:
        error.add_note(
            f"Preprocessing did not finish; {len(processed_files)}/{len(manifest_rows)} maps were saved. "
            f"Partial cache files were preserved in {cache_dir.resolve()}. "
            "Do not train with this partial cache. Rename the folder as a backup or set paths.cache "
            "in config.json to a new folder, then rerun preprocessing from the beginning."
        )
        raise


def load_bags(cache_dir, split_rows, preprocessing):
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Preprocessed cache not found: {cache_dir.resolve()}. Run python preprocess.py first."
        )
    missing_metadata = [
        file_name for file_name in ("index.csv", "preprocessing.json")
        if not (cache_dir / file_name).is_file()
    ]
    if missing_metadata:
        raise ValueError(
            f"Cache is incomplete at {cache_dir.resolve()}: missing {', '.join(missing_metadata)}. "
            "Rename this folder as a backup or set paths.cache in config.json to a new folder, "
            "then rerun preprocessing before training."
        )
    try:
        cache_preprocessing = json.loads((cache_dir / "preprocessing.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        error.add_note(
            f"Cache metadata is incomplete or invalid at {cache_dir.resolve()}. "
            "Rename the cache folder as a backup or set paths.cache in config.json to a new folder, "
            "then rerun preprocessing before training."
        )
        raise
    if cache_preprocessing != preprocessing:
        raise ValueError("Cache preprocessing differs from config.json; use the matching configuration.")
    cache_index = {row["file_name"]: row for row in read_csv(cache_dir / "index.csv")}
    bag_lookup = {}
    expected_shape = (
        np.prod(preprocessing["map_shape"]) // preprocessing["patch_size"] ** 2,
        preprocessing["channels"],
    )
    for row in split_rows:
        file_name = row["file_name"]
        if file_name in bag_lookup:
            continue
        if (
            file_name not in cache_index
            or cache_index[file_name]["patient_id"] != row["patient_id"]
            or cache_index[file_name]["aml_all_label"] != row["aml_all_label"]
        ):
            raise ValueError(f"Cache/split identity mismatch: {file_name}")
        with np.load(cache_dir / cache_index[file_name]["cache_file"], allow_pickle=False) as payload:
            bag = payload["bag"]
        if bag.shape != expected_shape or not np.isfinite(bag).all():
            raise ValueError(f"Invalid bag: {file_name}")
        bag_lookup[file_name] = bag
    return bag_lookup


def inner_partition(outer_train_rows, fold, seed):
    validation_groups = []
    for label_name, multiplier in [("ALL", 3), ("AML", 5)]:
        class_groups = sorted({
            row["patient_id"] for row in outer_train_rows
            if row["aml_all_label"] == label_name
        })
        if len(class_groups) < 2:
            raise ValueError("Each class needs at least two outer-train groups.")
        validation_groups.append(class_groups[(seed + fold * multiplier) % len(class_groups)])
    inner_train_rows = [row for row in outer_train_rows if row["patient_id"] not in validation_groups]
    validation_rows = [row for row in outer_train_rows if row["patient_id"] in validation_groups]
    return inner_train_rows, validation_rows


def normalize(bags, channel_mean, channel_std):
    flattened_bags = bags.reshape(-1, bags.shape[-1])
    normalized_bags = (flattened_bags - channel_mean[None, :]) / channel_std[None, :]
    return normalized_bags.reshape(bags.shape).astype(np.float32)
