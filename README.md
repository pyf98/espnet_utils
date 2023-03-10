# espnet_utils
Small utility scripts for ESPnet.


## Profiler for encoder

Please use the latest version (e.g., >= 0.8.1) of DeepSpeed. Prior versions might cause some unexpected problems like https://github.com/microsoft/DeepSpeed/issues/2231

[`profile_encoder.py`](profile_encoder.py) uses DeepSpeed's `flops_profiler` to calculate the computational cost of `ESPnetASRModel`. Please put it under the path: `egs2/TEMPLATE/asr1/pyscripts/utils` and put [`profile.sh`](profile.sh) in a specific recipe: `egs2/xxx/asr1/`.

This script only supports ASR model for now. But it should be straightforward to extend it to others.


## Diagonality of encoder attention

Please put [`compute_diagonality.py`](compute_diagonality.py) under the path: `egs2/TEMPLATE/asr1/pyscripts/utils` and put [`compute_diagonality.sh`](compute_diagonality.sh) in a specific recipe: `egs2/xxx/asr1/`.

Reference: S. Zhang, E. Loweimi, P. Bell and S. Renals, "On The Usefulness of Self-Attention for Automatic Speech Recognition with Transformers," 2021 IEEE Spoken Language Technology Workshop (SLT), Shenzhen, China, 2021, pp. 89-96, doi: 10.1109/SLT48900.2021.9383521.
