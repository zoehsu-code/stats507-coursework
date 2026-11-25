"""From Global to Local: Exploring Temporal Correspondence Between Video Color and Music."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

DATASET = "HKUSTAudio/VidMuse-V2M-Dataset"
REVISION = "cb3a422"
COUNT = 50
LAB_BINS = (8, 8, 8)
LAB_RANGE = ((0, 100), (-128, 128), (-128, 128))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def download(data, limit):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    def fetch(name):
        return hf_hub_download(DATASET, name, repo_type="dataset",
                               revision=REVISION, local_dir=data)

    manifest = Path(fetch("V2M-bench.txt"))
    ids = [line.strip() for line in manifest.read_text().splitlines() if line.strip()][:COUNT]
    if len(ids) != COUNT or len(set(ids)) != COUNT:
        raise ValueError("Expected 50 distinct IDs at the start of the benchmark manifest")
    selected = [
        {"id": key, "video": f"benchmark/video/{key}.mp4",
         "music": f"benchmark/music/{key}.wav"}
        for key in ids
    ]
    (data / "sample.json").write_text(json.dumps(
        {"dataset": DATASET, "revision": REVISION, "pairs": selected}, indent=2) + "\n")
    failures = []
    for pair in selected[:limit]:
        for field in ("video", "music"):
            try:
                fetch(pair[field])
            except (HfHubHTTPError, OSError, RuntimeError) as error:
                failures.append({"id": pair["id"], "file": pair[field], "error": str(error)})
        print(f"Checked downloads for {pair['id']}", flush=True)
    write_csv(data / "download_failures.csv", failures, ["id", "file", "error"])
    if failures:
        raise SystemExit(f"{len(failures)} file downloads failed. See download_failures.csv.")


def extract_color(reader):
    from PIL import ImageCms

    to_lab = ImageCms.buildTransform(ImageCms.createProfile("sRGB"),
                                    ImageCms.createProfile("LAB"), "RGB", "LAB")

    means, histograms = [], []
    for frame, _ in reader.extract_frames():
        # Use the same decoded frames and spatial resolution for both methods.
        frame.thumbnail((160, 160))
        encoded = np.asarray(ImageCms.applyTransform(frame, to_lab)).reshape(-1, 3)
        # Pillow stores scaled L* and two signed, eight-bit chromatic channels.
        pixels = encoded.astype(np.float32)
        pixels[:, 0] *= 100 / 255
        pixels[:, 1:] = encoded[:, 1:].view(np.int8)
        means.append(pixels.mean(axis=0))
        counts, _ = np.histogramdd(pixels, bins=LAB_BINS, range=LAB_RANGE)
        histograms.append(np.sqrt(counts.ravel() / counts.sum()))
    return {
        "lab_mean": np.linalg.norm(np.diff(means, axis=0), axis=1),
        "lab_histogram": np.linalg.norm(np.diff(histograms, axis=0), axis=1) / np.sqrt(2),
    }


def analyze(args):
    sys.path.insert(0, str(args.music_repo.resolve()))
    import torch
    from analyze_train_temporal_shift import SHIFTS, aligned_curves, shift_scores
    from music_event_curve.audio_io import load_audio
    from music_event_curve.encoder import load_musicfm, extract_musicfm_features
    from video_event_curve.event_curve import compute_cosine_dissimilarity, compute_timestamps
    from video_event_curve.video_io import load_video

    sample = json.loads((args.data / "sample.json").read_text())
    pairs = sample["pairs"][:args.limit]
    bundle = load_musicfm(args.device, str(args.cache), args.offline)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "sample.json").write_text(json.dumps({**sample, "pairs": pairs}, indent=2) + "\n")
    scores, media, failures = [], [], []
    example = {}
    for pair in pairs:
        key = pair["id"]
        try:
            reader = load_video(args.data / pair["video"])
            colors = extract_color(reader)
            waveform, audio = load_audio(args.data / pair["music"])
            video_seconds, audio_seconds = reader.metadata["duration"], audio["duration"]
            media.append({"id": key, "video_format": "MP4", "music_format": "WAV",
                          "video_seconds": video_seconds, "music_seconds": audio_seconds,
                          "music_minus_video_seconds": audio_seconds - video_seconds,
                          "fps": reader.metadata["fps"], "frames": reader.metadata["num_frames"],
                          "video_start_pts": reader.metadata["first_frame_pts_seconds"],
                          "music_start_pts": audio["audio_start_pts_seconds"]})
            features, feature_times, _ = extract_musicfm_features(waveform, encoder_bundle=bundle)
            music_curve = compute_cosine_dissimilarity(features)
            # The official separate media files are compared from their decoded starts.
            mt = compute_timestamps(feature_times)
            vt = compute_timestamps(reader.timestamps)
        except torch.OutOfMemoryError as error:
            failures.append({"id": key, "stage": "music_encoder", "error": str(error)})
            continue
        except (OSError, ValueError) as error:
            failures.append({"id": key, "stage": "extraction", "error": str(error)})
            continue
        np.savez_compressed(args.output / f"{key}_raw.npz", color_times=vt,
                            music_times=mt, music=music_curve, **colors)
        for method, color in colors.items():
            try:
                if color.size < 3 or color.std() < 1e-8 or music_curve.std() < 1e-8:
                    raise ValueError("Insufficient or near-constant raw curve")
                times, aligned_color, aligned_music = aligned_curves(
                    color, vt, music_curve, mt, audio_seconds)
                correlations, anchors = shift_scores(times, aligned_color, aligned_music)
            except ValueError as error:
                failures.append({"id": key, "stage": method, "error": str(error)})
                continue
            scores.extend({"id": key, "method": method, "shift_seconds": float(shift),
                           "pearson": float(r), "abs_pearson": float(abs(r)), "anchors": anchors}
                          for shift, r in zip(SHIFTS, correlations))
            if method not in example:
                example[method] = (key, times, aligned_color, aligned_music)
        print(f"Processed {key}", flush=True)

    write_csv(args.output / "dataset.csv", media, [
        "id", "video_format", "music_format", "video_seconds", "music_seconds",
        "music_minus_video_seconds", "fps", "frames", "video_start_pts", "music_start_pts"])
    write_csv(args.output / "scores.csv", scores, [
        "id", "method", "shift_seconds", "pearson", "abs_pearson", "anchors"])
    write_csv(args.output / "failures.csv", failures, ["id", "stage", "error"])
    summarize(args.output, pairs, media, scores, example, SHIFTS)
    if not scores:
        raise SystemExit("No valid correlations. See failures.csv.")


def summarize(output, pairs, media, scores, example, shifts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ("lab_mean", "lab_histogram")
    ids = [{row["id"] for row in scores if row["method"] == method} for method in methods]
    common = set.intersection(*ids)
    summary = []
    for cohort in ("available", "both_methods"):
        for method in methods:
            for shift in shifts:
                values = np.array([row["pearson"] for row in scores
                                   if row["method"] == method and row["shift_seconds"] == shift
                                   and (cohort == "available" or row["id"] in common)])
                if values.size:
                    summary.append({"cohort": cohort, "method": method, "shift_seconds": shift,
                                    "n": len(values), "mean_r": values.mean(),
                                    "median_r": np.median(values), "mean_abs_r": abs(values).mean()})
    write_csv(output / "summary.csv", summary,
              ["cohort", "method", "shift_seconds", "n", "mean_r", "median_r", "mean_abs_r"])
    durations = []
    for field in ("video_seconds", "music_seconds", "music_minus_video_seconds"):
        values = np.array([row[field] for row in media])
        if values.size:
            durations.append({"metric": field, "n": len(values), "mean": values.mean(),
                              "min": values.min(), "median": np.median(values), "max": values.max()})
    write_csv(output / "duration_summary.csv", durations, ["metric", "n", "mean", "min", "median", "max"])
    (output / "counts.json").write_text(json.dumps(
        {"selected": len(pairs), "decoded": len(media), "scored_both_methods": len(common)}, indent=2))

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    for ax, (method, result) in zip(axes, example.items()):
        key, times, color, music = result
        ax.plot(times, color, label="Color")
        ax.plot(times, music, label="MusicFM", alpha=.8)
        ax.set(title=f"{method} ({key})", xlabel="Time (s)", ylabel="Smoothed standardized change")
        ax.legend()
    fig.savefig(output / "event_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for method in methods:
        zero = [row["pearson"] for row in scores if row["method"] == method and row["shift_seconds"] == 0]
        ax.hist(zero, bins=np.linspace(-1, 1, 21), alpha=.55, label=method)
    ax.set(xlabel="Zero-shift Pearson r", ylabel="Pairs")
    ax.legend()
    fig.savefig(output / "pearson_distribution.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric in zip(axes, ("mean_r", "mean_abs_r")):
        for method in methods:
            rows = [row for row in summary if row["method"] == method and row["cohort"] == "available"]
            ax.plot([row["shift_seconds"] for row in rows], [row[metric] for row in rows],
                    marker="o", label=method)
        ax.axvline(0, color="0.5", linestyle="--")
        ax.set(xlabel="Music shift (s)", ylabel=metric)
        ax.legend()
    fig.savefig(output / "temporal_shift.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "analyze"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=COUNT, choices=range(1, COUNT + 1), metavar="1..50")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--music-repo", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.command == "download":
        download(args.data, args.limit)
    else:
        if args.output is None or args.music_repo is None:
            parser.error("analyze requires --output and --music-repo")
        analyze(args)


if __name__ == "__main__":
    main()
