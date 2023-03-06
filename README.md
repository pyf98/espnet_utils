# espnet_utils
Small utility scripts for ESPnet.

## Profiler for encoder

Please use the latest version (e.g., >= 0.8.1) of DeepSpeed. Prior versions have some issues.

[`profile_encoder.py`](profile_encoder.py) uses DeepSpeed's `flops_profiler` to calculate the computational cost of `ESPnetASRModel`. Please put it under the path: `egs2/TEMPLATE/asr1/pyscripts/utils` and put [`profile.sh`](profile.sh) in a specific recipe: `egs2/xxx/asr1/`.

This script only supports ASR model for now. But it should be straightforward to extend it to others.
