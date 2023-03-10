from collections import defaultdict
import argparse
import torch
import sys
import numpy as np
from tqdm import tqdm
from kaldiio import ReadHelper

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet2.tasks.asr import ASRTask
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none
from espnet2.torch_utils.device_funcs import to_device


def diagonality(att):
    # att (torch.Tensor): (B, NHeads, Tout, Tin)
    if isinstance(att, torch.Tensor):
        att = att.detach().cpu().numpy()
    bs, nheads, len_out, len_in = att.shape

    rel_distance = np.zeros((len_out, len_in))
    for i in range(len_out):
        for j in range(len_in):
            rel_distance[i, j] = np.abs(i - j)
    
    result = (1. - (att * rel_distance).sum(-1) / rel_distance.max(-1)).mean(-1)    # (B, NHeads)
    return result


def format_array(arr):
    string = np.array2string(
        arr,
        formatter={'float_kind': lambda x: f"{x:.6f}"},
        separator=','
    )
    return string


def get_parser():
    parser = argparse.ArgumentParser(
        description="Calculate the diagonality of attention weights."
    )
    parser.add_argument(
        "--asr_train_config",
        type=str,
        help="path to the asr train config file"
    )
    parser.add_argument(
        "--asr_model_file",
        type=str,
        help="path to the trained model file"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=['cpu', 'gpu'],
        default='cpu',
        help="device name: cpu (default), gpu"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="log file name"
    )
    parser.add_argument(
        "--wav_scp",
        type=str_or_none,
        help="Path to wav.scp"
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    log_fp = open(args.log_file, 'a')
    log_fp.write(' '.join(sys.argv) + '\n')

    if args.device == 'gpu':
        args.device = 'cuda'
    
    asr_model, asr_train_args = ASRTask.build_model_from_file(
        args.asr_train_config, args.asr_model_file, args.device
    )
    asr_model.eval()

    # forward hook
    outputs = {}
    handles = {}
    for name, modu in asr_model.named_modules():
        if "encoder" in name:
            def hook(module, input, output, name=name):
                if isinstance(module, MultiHeadedAttention):
                    # NOTE(kamo): MultiHeadedAttention doesn't return attention weight
                    # attn: (B, Head, Tout, Tin)
                    outputs[name] = module.attn.detach().cpu()

            handle = modu.register_forward_hook(hook)
            handles[name] = handle
    
    # iterate over all samples
    return_dict = defaultdict(list)
    with ReadHelper(f"scp:{args.wav_scp}") as reader:
        for key, (rate, numpy_array) in tqdm(reader):
            speech = torch.tensor(numpy_array, dtype=torch.float32).unsqueeze(0)     # (1, T)
            lengths = torch.tensor([len(numpy_array)], dtype=torch.long)
            batch = {"speech": speech, "speech_lengths": lengths}

            # a. To device
            batch = to_device(batch, device=args.device)

            # b. Forward Encoder
            enc, _ = asr_model.encode(**batch)

            # Derive the attention results
            for name, output in outputs.items():
                # name: e.g., encoder.encoders.23.self_attn
                # output: (Batch, NHead, Tout, Tin)
                diag = diagonality(output)      # (B, NHeads)
                return_dict[name].append(diag)

            outputs.clear()

    # 3. Remove all hooks
    for _, handle in handles.items():
        handle.remove()

    return_dict = dict(return_dict)

    for name, diags in return_dict.items():
        diags = np.concatenate(diags, axis=0)     # (num_samples, num_heads)
        log_fp.write(
            f"{name}: nsamples={diags.shape[0]}, mean={format_array(diags.mean(0))}, std={format_array(diags.std(0))}\n"
        )
    
    log_fp.close()
