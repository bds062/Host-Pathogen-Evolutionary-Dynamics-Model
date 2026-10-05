#!/usr/bin/env python3
"""Compute and save aligned ESS raster sweeps for three parameter groups.

Each germination-time frame is saved in its own directory.  Unlike the
interactive ``RasterModel.plot_heatmap`` method, the figures made here always
show the full day-of-year frame (0--365 on both axes).

Examples
--------
Run the complete sweep (the same nine frames as ESS.ipynb)::

    python plot_ess_rasters.py

Smoke-test one frame on a small grid::

    python plot_ess_rasters.py --frames 1 --grid-size 5 --processes 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import Model
from plot import RasterModel


FRAME_DEFINITIONS = (
    # Original narrow ranges retained as reference; default grids use 0--365.
    (62.5, 83, 103),
    (92.5, 111, 131),
    (122.5, 144, 164),
    (152.5, 177, 197),
    (182.5, 205, 225),
    (212.5, 233, 253),
    (242.5, 266, 286),
    (272.5, 294, 314),
    (302.5, 326, 346),
)


PARAMETER_GROUP_LABELS = {
    "our_params": "Default",
    "dianthus_params": "Dianthus pavonius",
    "silene_params": "Silene latifolia",
}


PARAMETER_GROUPS = {
    "our_params": {"Bj": 0.4 / 365, "maturity": 4 / 365},
    "dianthus_params": {
        "Bj": 3.38e-4, "Bv": 3.38e-5, "Bf": 3.38e-5,
        "births": 1.3 / 365, "gamma": 1.5e-4, "death": 0.187 / 365,
        "maturity": 4 / 365, "g": 2 / 365,
    },
    "silene_params": {
        "Bj": 0.6 / 365, "Bv": 0.4 / 365, "Bf": 0.4 / 365,
        "births": 4 / 365, "gamma": 0.001, "death": 0.36 / 365,
        "maturity": 0.85 / 365, "g": 2 / 365,
    },
}


def frame_name(index: int, germination_time: float) -> str:
    """Return a filesystem-safe, naturally sortable frame directory name."""
    day = f"{germination_time:g}".replace(".", "p")
    return f"frame_{index + 1:02d}_germination_{day}"


def completed_frame(directory: Path, germination_time: float, grid_size: int,
                    model_config: dict[str, float]) -> float | None:
    """Return a cached ESS if this frame is complete and matches the request."""
    metadata_path = directory / "metadata.json"
    if not (directory / "raster.png").is_file() or not (directory / "raster.npz").is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
        matches = (
            metadata.get("germination_time") == germination_time
            and metadata.get("flowering_min") == 0.0
            and metadata.get("flowering_max") == 365.0
            and metadata.get("grid_size") == grid_size
            and metadata.get("parameters") == model_config
        )
        return float(metadata["ess_flowering_time"]) if matches else None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_raster_figure(data: np.ndarray, output: Path, germination_time: float, group_label: str) -> None:
    """Save one raster with a common 0--365 coordinate frame."""
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(
        data,
        extent=(0, 365, 0, 365),
        origin="lower",
        aspect="equal",
        cmap="RdBu_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_xlim(0, 365)
    ax.set_ylim(0, 365)
    ax.set_xlabel("Genotype 2 Flowering Time (day of year)", color="#2166AC")
    ax.set_ylabel("Genotype 1 Flowering Time (day of year)", color="#B2182B")
    ax.set_title(f"{group_label} — ESS raster — germination time = {germination_time:g} days")
    ax.set_xticks(np.arange(0, 366, 60))
    ax.set_yticks(np.arange(0, 366, 60))
    ax.grid(True, color="black", alpha=0.15, linewidth=0.5)
    ax.axline((0, 0), slope=1, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    fig.colorbar(image, ax=ax, label="Genotype 1 abundance")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def run_frame(index: int, germination_time: float, reference_low: float,
              reference_high: float, grid_size: int, output_dir: Path,
              processes: int | None, model_config: dict[str, float],
              group_label: str) -> float:
    # Identical grid for every frame and group, so adjacent images line up.
    frame_low, frame_high = 0.0, 365.0
    values = np.linspace(frame_low, frame_high, grid_size)
    model = Model(
        maturity=4 / 365,
        t=(0, 365 * 400),
        germination_time=germination_time,
        Bj=0.4 / 365,
        infected_time=182.5,
        infected_time2=182.5,
    )
    for parameter, value in model_config.items():
        setattr(model, parameter, value)
    raster = RasterModel(model, "infected_time", "infected_time2", values, values)

    directory = output_dir / frame_name(index, germination_time)
    directory.mkdir(parents=True, exist_ok=True)
    cached_ess = completed_frame(directory, germination_time, grid_size, model_config)
    if cached_ess is not None:
        # Re-render the lightweight PNG when presentation labels change, but
        # never rerun the expensive simulation for a completed frame.
        with np.load(directory / "raster.npz") as saved:
            save_raster_figure(saved["heatmap"], directory / "raster.png",
                               germination_time, group_label)
        print(f"Skipping completed {directory} (ESS={cached_ess:.2f})")
        return cached_ess

    started = time.time()
    data = raster.raster(num_processes=processes)
    ess = float(raster.ess_flowering_time)

    np.savez_compressed(
        directory / "raster.npz",
        heatmap=data,
        flowering_times=values,
        germination_time=germination_time,
        ess_flowering_time=ess,
    )
    (directory / "metadata.json").write_text(json.dumps({
        "germination_time": germination_time,
        "flowering_min": frame_low,
        "flowering_max": frame_high,
        "reference_sweep_min": reference_low,
        "reference_sweep_max": reference_high,
        "grid_size": grid_size,
        "ess_flowering_time": ess,
        "parameters": model_config,
    }, indent=2) + "\n")
    save_raster_figure(data, directory / "raster.png", germination_time, group_label)
    print(f"Saved {directory} (ESS={ess:.2f}; {time.time() - started:.1f}s)")
    return ess


def save_gif(output_dir: Path, definitions: tuple, duration: float) -> None:
    """Stitch the frame rasters in one parameter-group directory into a GIF."""
    from PIL import Image

    frame_paths = [
        output_dir / frame_name(index, germination_time) / "raster.png"
        for index, (germination_time, _, _) in enumerate(definitions)
    ]
    frame_paths = [path for path in frame_paths if path.is_file()]
    if not frame_paths:
        print(f"No raster PNGs found for GIF: {output_dir}")
        return

    images = [Image.open(path).convert("RGB") for path in frame_paths]
    try:
        images[0].save(
            output_dir / "rasters.gif", save_all=True, append_images=images[1:],
            duration=max(1, int(duration * 1000)), loop=0, optimize=False
        )
    finally:
        for image in images:
            image.close()
    print(f"Saved GIF to {output_dir / 'rasters.gif'} ({len(frame_paths)} frames)")


def save_combined_gif(output_dir: Path, definitions: tuple,
                      group_names: list[str], duration: float) -> None:
    """Make one GIF with one shared-axis/colorbar figure per frame.

    The composite frames are rendered from the saved ``raster.npz`` files,
    rather than stitched from the individual PNGs.  This lets Matplotlib
    share the x/y axes, one colorbar, and one overall title cleanly.
    """
    from tempfile import TemporaryDirectory
    from PIL import Image

    frame_data = []
    for index, (germination_time, _, _) in enumerate(definitions):
        arrays = []
        for group_name in group_names:
            raster_path = (
                output_dir / group_name / frame_name(index, germination_time) / "raster.npz"
            )
            if not raster_path.is_file():
                print(f"Cannot make combined GIF; missing {raster_path}")
                return
            with np.load(raster_path) as saved:
                arrays.append(np.asarray(saved["heatmap"]))
        frame_data.append((germination_time, arrays))

    if not frame_data:
        print("No frames available for combined GIF")
        return

    with TemporaryDirectory(prefix="ess-composite-") as temp_dir:
        temp_dir = Path(temp_dir)
        frame_paths = []
        for index, (germination_time, arrays) in enumerate(frame_data):
            fig, axes = plt.subplots(
                1, len(group_names), figsize=(15, 5.6),
                sharex=True, sharey=True, constrained_layout=True,
            )
            axes = np.atleast_1d(axes)
            images = []
            for ax, group_name, data in zip(axes, group_names, arrays):
                image = ax.imshow(
                    data, extent=(0, 365, 0, 365), origin="lower",
                    aspect="equal", cmap="RdBu_r", vmin=0.0, vmax=1.0,
                    interpolation="nearest",
                )
                images.append(image)
                ax.set_title(PARAMETER_GROUP_LABELS[group_name], fontsize=12, pad=8)
                ax.set_xlim(0, 365)
                ax.set_ylim(0, 365)
                ax.set_xticks(np.arange(0, 366, 60))
                ax.set_yticks(np.arange(0, 366, 60))
                ax.grid(True, color="black", alpha=0.15, linewidth=0.5)
                ax.axline((0, 0), slope=1, color="black", linestyle="--",
                          linewidth=0.8, alpha=0.5)

            axes[0].set_ylabel("Genotype 1 flowering time (day of year)")
            fig.supxlabel("Genotype 2 flowering time (day of year)")
            fig.suptitle(
                f"ESS raster comparison — germination time = {germination_time:g} days",
                fontsize=15,
            )
            fig.colorbar(images[0], ax=axes.tolist(), shrink=0.86,
                         label="Genotype 1 abundance")
            frame_path = temp_dir / f"frame_{index:02d}.png"
            fig.savefig(frame_path, dpi=160, facecolor="white")
            plt.close(fig)
            frame_paths.append(frame_path)

        images = [Image.open(path).convert("RGB") for path in frame_paths]
        try:
            output_path = output_dir / "all_parameters_rasters.gif"
            images[0].save(
                output_path, save_all=True, append_images=images[1:],
                duration=max(1, int(duration * 1000)), loop=0, optimize=False,
            )
        finally:
            for image in images:
                image.close()
    print(f"Saved combined GIF to {output_dir / 'all_parameters_rasters.gif'} "
          f"({len(frame_paths)} frames)")


def save_summary(output_dir: Path, germination: np.ndarray, ess: np.ndarray, group_name: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.plot(germination, ess, marker="o", linewidth=2, color="#2563eb")
    ax.plot((0, 365), (0, 365), "--", color="0.4", linewidth=1, label="ESS = germination time")
    ax.set(xlim=(0, 365), ylim=(0, 365), xlabel="Germination time (day of year)",
           ylabel="ESS flowering time (day of year)", title=f"{PARAMETER_GROUP_LABELS.get(group_name, group_name)}: ESS movement across germination frames")
    ax.set_xticks(np.arange(0, 366, 60))
    ax.set_yticks(np.arange(0, 366, 60))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.savefig(output_dir / "ess_across_frames.png", dpi=300)
    plt.close(fig)
    np.savez(output_dir / "ess_across_frames.npz", germination_times=germination, ess_flowering_times=ess)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "figs")
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument("--processes", type=int, default=None,
                        help="Worker processes; defaults to the model's core setting.")
    parser.add_argument("--frames", type=int, default=None,
                        help="Only run the first N frames (useful for testing).")
    parser.add_argument("--groups", nargs="+", choices=tuple(PARAMETER_GROUPS),
                        default=list(PARAMETER_GROUPS),
                        help="Parameter groups to run; defaults to all three.")
    parser.add_argument("--gif-duration", type=float, default=0.8,
                        help="Seconds per GIF frame (default: 0.8).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grid_size < 2:
        raise SystemExit("--grid-size must be at least 2")
    if args.frames is not None and not 1 <= args.frames <= len(FRAME_DEFINITIONS):
        raise SystemExit(f"--frames must be between 1 and {len(FRAME_DEFINITIONS)}")

    definitions = FRAME_DEFINITIONS[:args.frames]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for group_name in args.groups:
        group_dir = args.output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        germination = []
        ess = []
        for index, (germination_time, low, high) in enumerate(definitions):
            germination.append(germination_time)
            ess.append(run_frame(index, germination_time, low, high, args.grid_size,
                                 group_dir, args.processes, PARAMETER_GROUPS[group_name],
                                 PARAMETER_GROUP_LABELS[group_name]))
        save_summary(group_dir, np.asarray(germination), np.asarray(ess), group_name)
        save_gif(group_dir, definitions, args.gif_duration)
        print(f"Saved summary to {group_dir / 'ess_across_frames.png'}")

    save_combined_gif(args.output_dir, definitions, args.groups, args.gif_duration)


if __name__ == "__main__":
    main()
