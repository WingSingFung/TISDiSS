#!/usr/bin/env bash
export PYTHONPATH="/23TW026/mss/espnet-efficient-train:$PYTHONPATH"
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
speech_segment=16000 # use 16000 like TIGER and SepReformer

# for en in 4 5 6 7 8 9; do
#     CUDA_VISIBLE_DEVICES=4 python pyscripts/utils/profiler.py \
#     --enh_config conf/efficient_train/tflocoformer_efficient/train_enh_tflocoformer_efficient_residual_en1x6_l1x6.yaml \
#     --model_file valid.loss.best.pth \
#     --inference_config conf/efficient_infer/en/en_${en}.yaml \
#     --speech_segment ${speech_segment}
# done

# for en in 1 2 3; do
#     for re in 1 2 3 4 5 6; do
#         CUDA_VISIBLE_DEVICES=4 python pyscripts/utils/profiler.py \
#         --enh_config conf/efficient_train/tflocoformer_efficient_splitdecoder/train_enh_tflocoformer_efficient_splitdecoder_residual_en1x2_re1x3_l1x2+1x3.yaml \
#         --model_file valid.loss.best.pth \
#         --inference_config conf/efficient_infer/en_re/en_${en}_re_${re}.yaml \
#         --speech_segment ${speech_segment}

#         CUDA_VISIBLE_DEVICES=4 python pyscripts/utils/profiler.py \
#         --enh_config conf/efficient_train/tflocoformer_efficient_splitdecoder/train_enh_tflocoformer_efficient_splitdecoder_residual_compress_en1x2_re1x3_l1x2+1x3.yaml \
#         --model_file valid.loss.best.pth \
#         --inference_config conf/efficient_infer/en_re/en_${en}_re_${re}.yaml \
#         --speech_segment ${speech_segment}
#     done
# done

python pyscripts/utils/profiler.py summary