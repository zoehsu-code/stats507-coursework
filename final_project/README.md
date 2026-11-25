# From Global to Local: Exploring Temporal Correspondence Between Video Color and Music

This pilot analyzes the first 50 paired MP4 and WAV files in VidMuse’s `V2M-bench.txt`, in manifest order. It uses CIELAB mean and 8×8×8 histogram distances for visual change and MusicFM-MSD layer 7 for musical change. Both event curves use adjacent observations and midpoint timestamps.

## Great Lakes

Run from `final_project` in a Python 3.10–3.12 environment. Set `WORK` to a writable Great Lakes scratch directory and `ACCOUNT` to your Slurm account with A40 access.

```bash
WORK=/path/to/great-lakes/scratch/vidmuse
ACCOUNT=your_slurm_account
PYTHON="$(command -v python)"
mkdir -p "$WORK"
git clone https://github.com/zoehsu-code/music_generate.git "$WORK/music_generate"
git -C "$WORK/music_generate" checkout --detach 769ee4a
"$PYTHON" -m pip install -r "$WORK/music_generate/requirements-music.txt"
"$PYTHON" -m pip install 'huggingface_hub[hf_xet]<1'
PYTHONPATH="$WORK/music_generate" "$PYTHON" - "$WORK/hf-cache" <<'PY'
import sys
from huggingface_hub import hf_hub_download
from music_event_curve.encoder import CONFIG_REVISION, WEIGHT_REVISION

for name in ("msd_stats.json", "pretrained_msd.pt"):
    hf_hub_download("minzwon/MusicFM", name, revision=WEIGHT_REVISION, cache_dir=sys.argv[1])
hf_hub_download("facebook/wav2vec2-conformer-rope-large-960h-ft", "config.json",
                revision=CONFIG_REVISION, cache_dir=sys.argv[1])
PY
```

Download one pair and submit the smoke test.

```bash
"$PYTHON" analyze_vidmuse.py download --data "$WORK/data" --limit 1
sbatch --account="$ACCOUNT" --partition=spgpu --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:30:00 --output="$WORK/smoke-%j.out" \
  --wrap="\"$PYTHON\" analyze_vidmuse.py analyze --data \"$WORK/data\" --limit 1 --output \"$WORK/smoke\" --music-repo \"$WORK/music_generate\" --cache \"$WORK/hf-cache\" --offline"
```

After checking the smoke output, download and analyze the first 50. The first pair is reused.

```bash
"$PYTHON" analyze_vidmuse.py download --data "$WORK/data"
sbatch --account="$ACCOUNT" --partition=spgpu --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=02:00:00 --output="$WORK/pilot-%j.out" \
  --wrap="\"$PYTHON\" analyze_vidmuse.py analyze --data \"$WORK/data\" --output \"$WORK/pilot\" --music-repo \"$WORK/music_generate\" --cache \"$WORK/hf-cache\" --offline"
```

## Method and outputs

Video frames are resized to fit 160 × 160 and converted to CIELAB (D50) using Pillow ImageCms. Mean colors use Euclidean distance and normalized histograms use Hellinger distance. MusicFM encodes mono 24 kHz audio at 25 Hz and uses adjacent cosine dissimilarity.

The pilot aligns the decoded starts of the separate files and records durations and start timestamps for inspection. It interpolates standardized curves to a shared 25 Hz grid and applies a 31-point Hann window. Pearson uses the same video anchors at every shift from −2 to +2 seconds. A positive shift compares color with later music.

- `dataset.csv` and `duration_summary.csv` record formats, frame rates, durations, and duration differences.
- `scores.csv` contains signed and absolute Pearson per pair and shift. `summary.csv` includes mean signed Pearson, median signed Pearson, and mean absolute Pearson. Use cohort `both_methods` for the comparison on identical pairs.
- `event_curves.png`, `pearson_distribution.png`, and `temporal_shift.png` show the aligned curves, zero-shift distributions, and shift profiles.
- `sample.json`, `counts.json`, `failures.csv`, and the raw `.npz` files preserve sample IDs, coverage, errors, and unprocessed curves. Download errors are recorded separately in `data/download_failures.csv`. Use a new output directory for each analysis run.
