# Hi-Drum

**Hi-Drum: A Framework for Drum Transcription Based on Self-Supervised Music Representations**  
ICASSP 2027 manuscript code + demo.

- Paper demo page: https://hanguang9527.github.io/Hi-Drum/
- This repository: training / evaluation / transcription code for reproduction

## What is included

| Path | Contents |
|------|----------|
| `onsets_and_frames/` | Model (encoder, HDM/adapter, LLM decoder, multi-task heads) |
| `utils/` | Dataset preparation helpers |
| `train.py` | Training entry (Sacred) |
| `evaluate.py` | Evaluation entry |
| `transcribe.py` | Audio → MIDI transcription |
| `requirements_ubuntu.txt` / `deploy_env.sh` | Environment setup |
| `website/` | Static demo site (GitHub Pages) |
| `model/` | Placeholder only — download pretrained backbones locally (see `model/README.md`) |

**Not included (by design):** training logs, run directories, checkpoints, ablation notes, or paper PDF packages.

## Environment

Ubuntu + Conda is the tested path:

```bash
chmod +x deploy_env.sh
./deploy_env.sh
conda activate drum
```

Or install from the lock file:

```bash
pip install -r requirements_ubuntu.txt
```

See also `Environment_Deployment_Guide.md`.

## Data

Prepare Groove / E-GMD (and optional others) into the processed layout expected by `onsets_and_frames/dataset.py`.  
Default paths in `train.py` are placeholders:

```text
./data/groove_processed_5_10_24khz_7class
./data/egmd_processed_5_10_24khz_7class
```

Override them on the command line. Helpers under `utils/` and `build_h5_cache.py` can assist preprocessing.

## Pretrained backbones

Download encoder/decoder weights into `model/` before training.  
For the paper default (**MERT-330M + Qwen3-0.6B**), see [`model/README.md`](./model/README.md).

## Train (paper-style)

Sacred config lives in `train.py` (`@ex.config`). Example for the paper backbone pair:

```bash
python train.py with \
  train_on=GrooveEgmd \
  dataset_path=./data/groove_processed_5_10_24khz_7class \
  egmd_dataset_path=./data/egmd_processed_5_10_24khz_7class \
  encoder_model=MERT_330m \
  decoder_model=Qwen3-0.6B \
  decoder_freeze_model=True \
  enable_branch1=True enable_branch2=True enable_branch3=True \
  head_mode=paper_hierarchical \
  iterations=90000 \
  matched_samples_per_batch=8
```

Checkpoints and Sacred/TensorBoard outputs are written under `runs/` (gitignored).

Multi-GPU:

```bash
bash train_distributed.sh
```

## Evaluate / transcribe

```bash
python evaluate.py runs/<your_run>/model-<iter>.pt Egmd test --save-path ./results/
python transcribe.py --model runs/<your_run>/model-<iter>.pt --audio path/to/audio.wav --out out.mid
```

(Exact CLI flags: see `python evaluate.py -h` / `python transcribe.py -h`.)

## Demo website

Static site sources live in `website/`. Live page: https://hanguang9527.github.io/Hi-Drum/

## Citation

If you use this code, please cite the Hi-Drum ICASSP 2027 paper (title above).

## License

Demo site license: see `website/LICENCE`.  
Project code license will follow the camera-ready release policy of the authors.
