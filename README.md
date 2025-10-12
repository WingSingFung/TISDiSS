# TISDiSS: Training- and Inference-Time Scalable Framework for Discriminative Source Separation

Official implementation of TISDiSS, a scalable framework for discriminative source separation.

## Paper

📄 **arXiv**: https://arxiv.org/abs/2509.15666

🎯 **Status**: Submitted to ICASSP 2026

## Environmental setup:

pip install -r requirements.txt

## Infer

```sh
# Go to the corresponding example directory.
cd egs2/wsj0_2mix/enh1

# Data preparation and stats collection if necessary.
# NOTE: please fill the corresponding part of db.sh for data preparation.
# You need to download the WSJ0 dataset by yourself which may costs money.
./run.sh --stage 1 --stop_stage 5

# Inference. You can change the number of re like:for re in 3 6 8; do ... done for different M_re settings.
./infer_run.sh
```

## Training Code Release

The complete training code based on ESPnet-Enh will be released upon paper acceptance.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{feng2025tisdisstrainingtimeinferencetimescalable,
      title={TISDiSS: A Training-Time and Inference-Time Scalable Framework for Discriminative Source Separation}, 
      author={Yongsheng Feng and Yuetonghui Xu and Jiehui Luo and Hongjia Liu and Xiaobing Li and Feng Yu and Wei Li},
      year={2025},
      eprint={2509.15666},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2509.15666}, 
}
```
