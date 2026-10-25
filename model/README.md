# Pretrained backbones

Hi-Drum loads frozen music / language backbones from the local `model/` directory
(see `onsets_and_frames/music_encoder.py` and `onsets_and_frames/llm_decoder.py`).

Weights are **not** shipped in this repository. Download them yourself before training.

## Paper default (ICASSP)

| Role | Config name | Suggested local path | Hugging Face ID |
|------|-------------|----------------------|-----------------|
| Encoder | `MERT_330m` | `model/MERT-v1-330M` | [`m-a-p/MERT-v1-330M`](https://huggingface.co/m-a-p/MERT-v1-330M) |
| Decoder | `Qwen3-0.6B` | `model/Qwen3-0.6B` | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) |

Example:

```bash
# from repo root
huggingface-cli download m-a-p/MERT-v1-330M --local-dir model/MERT-v1-330M
huggingface-cli download Qwen/Qwen3-0.6B --local-dir model/Qwen3-0.6B
```

## Other supported backbones

Place each checkpoint under the `local_path` listed in the code configs, e.g.:

- `model/MERT-v1-95M`
- `model/MuQ-large-msd-iter`
- `model/MusicFM_MSD/...`
- `model/Qwen2.5-0.5B-Instruct`
- `model/Hunyuan-0.5B-Instruct`
- `model/MiniCPM4-0.5B`
- …
