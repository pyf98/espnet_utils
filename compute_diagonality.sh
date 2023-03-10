#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16          # <- match to OMP_NUM_THREADS
#SBATCH --mem=60000M
#SBATCH --partition=gpuA40x4-interactive        # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbjs-delta-gpu
#SBATCH --job-name=diagonality
#SBATCH --time=1:00:00           # hh:mm:ss for the job

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh

log_tag=conformer_e12
asr_exp=exp/asr_train_asr_conformer_e12_linear2048_raw_en_bpe500_sp

asr_train_config="${asr_exp}/config.yaml"
asr_model_file="${asr_exp}/valid.acc.ave_10best.pth"
device=gpu

python pyscripts/utils/compute_diagonality.py \
    --asr_train_config "${asr_train_config}" \
    --asr_model_file "${asr_model_file}" \
    --device "${device}" \
    --log_file "diagonality.${log_tag}.log" \
    --wav_scp "dump/raw/dev/wav.scp"
