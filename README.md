# espnet_utils
Small utility scripts for ESPnet.

## Profiler for encoder

Please use latest version (>= 0.8.1) of DeepSpeed to avoid issues.

`profile_encoder.py` uses DeepSpeed's `flops_profiler` to calculate the computational cost of `ESPnetASRModel`. Please put it under the path: `egs2/TEMPLATE/asr1/pyscripts/utils`.

