# TISDiSS: Training- and Inference-Time Scalable Framework for Discriminative Source Separation

Official implementation of **TISDiSS**, a scalable framework for discriminative source separation that enables flexible model scaling at both training and inference time.

## 🏆 Highlights

- ⚡ **State-of-the-art Performance**: Achieves SOTA results on WSJ0-2mix, WHAMR!, and Libri2Mix datasets
- 🔧 **Flexible Scalability**: Supports dynamic model scaling at both training and inference stages
- 💡 **Parameter Efficient**: Uses only 8.0M parameters while outperforming larger models (14.2M-59.4M)
- 📈 **Scalable Architecture**: Adjustable refinement blocks (M_re) for performance-efficiency trade-offs

## 🖼️ Architecture

<div align="center">

### Overall Framework
<img src="pics/TISDiSS-framework.pdf" alt="TISDiSS Framework" width="800"/>

### Separation Block
<img src="pics/TISDiSS-sepblock.pdf" alt="Separation Block" width="600"/>

### Refinement Block
<img src="pics/TISDiSS-reblock.pdf" alt="Refinement Block" width="600"/>

</div>

> **💡 Tip**: If the PDF images don't display properly on GitHub, you can view them directly in the [pics](./pics) folder or convert them to PNG format using:
> ```bash
> # Install ImageMagick if needed
> sudo apt-get install imagemagick
> # Convert PDF to PNG
> convert -density 300 pics/TISDiSS-framework.pdf pics/TISDiSS-framework.png
> ```

## 📊 Performance Comparison

### WSJ0-2mix Benchmark

Comparisons with prior methods on WSJ0-2mix dataset (with and without dynamic mixing). Results are shown in dB.

| Methods | Param [M] | SI-SNRi | SDRi |
|:--------|:---------:|:-------:|:----:|
| **Previous SOTA** | | | |
| SepReformer-B | 14.2 | 23.8 | 23.9 |
| SepReformer-L+DM | 59.4 | 25.1 | 25.2 |
| TF-Locoformer-M | 15.0 | 23.6 | 23.8 |
| TF-Locoformer-M+DM | 15.0 | 24.6 | 24.7 |
| TF-Locoformer-L | 22.5 | 24.2 | 24.3 |
| TF-Locoformer-L+DM | 22.5 | 25.1 | 25.2 |
| **TISDiSS (Ours)** | | | |
| TISDiSS-sep1×2-re1×3 (M_re=3) | 8.0 | 23.9 | 24.0 |
| TISDiSS-sep1×2-re1×3 (M_re=5) | 8.0 | 24.3 | 24.4 |
| TISDiSS-sep1×2-re1×6 (M_re=3) | 8.0 | 24.4 | 24.5 |
| TISDiSS-sep1×2-re1×6 (M_re=6) | 8.0 | 25.1 | 25.2 |
| TISDiSS-sep1×2-re1×6 (M_re=8) | 8.0 | **25.2** | **25.3** |

> **Note**: M_re indicates the number of refinement blocks used at inference time, demonstrating the scalability of our framework.


## 📄 Paper

**arXiv**: [https://arxiv.org/abs/2509.15666](https://arxiv.org/abs/2509.15666)

**Status**: Submitted to ICASSP 2026

---

## 🚀 Quick Start

### Environment Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Inference

Run inference on your audio files:

```bash
cd egs2/wsj0_2mix/enh1

python separate.py \
    --model_path ./exp/enh_train_enh_tflocoformer_pretrained/valid.loss.ave_5best.pth \
    --audio_path /path/to/input_audio \
    --audio_output_dir /path/to/output_directory
```

---

## 🔧 Training

### 1. Data Preparation

Navigate to the example directory:

```bash
cd egs2/wsj0_2mix/enh1
```

**Note**: You need to download the WSJ0 dataset separately (commercial license required).

#### Option A: WSJ0 in WAV format
If your WSJ0 dataset is already in WAV format, create a symbolic link:

```bash
mkdir -p ./data/wsj0
ln -s /path/to/your/WSJ0 ./data/wsj0/wsj0
```

Alternatively, modify line 24 in `./local/data.sh` to point to your WSJ0 path:
```bash
wsj_full_wav=/path/to/your/WSJ0/
```

#### Option B: WSJ0 in other formats
If your dataset is not in WAV format:
1. Uncomment lines 76-81 in `./local/data.sh`
2. Fill in the `WSJ0=` path in `db.sh`

### 2. Preprocessing

Run data preparation and statistics collection:

```bash
./run.sh --stage 1 --stop_stage 5
```

### 3. Model Training

Train the TISDiSS model:

```bash
CUDA_VISIBLE_DEVICES=1 ./run.sh \
    --stage 6 \
    --stop_stage 6 \
    --enh_config conf/efficient_train/tisdiss/train_enh_tisdiss_tflocoformer_en-residual_en1x2_re1x6_l1+1x6.yaml \
    --ngpu 1
```

### 4. Inference with Different Scalability Settings

Run inference with various refinement block configurations (M_re):

```bash
./infer_run.sh
```

You can modify the script to test different M_re settings:
```bash
for re in 3 6 8; do
    # Your inference commands here
done
```

---

## 📝 Note

This repository contains a streamlined version of ESPnet-Enh, specifically designed for easier training and inference of TISDiSS. The full ESPnet framework can be complex for new users, so we provide this simplified codebase focused on our method.

For more examples, additional features, and the complete ESPnet-Enh toolkit, please refer to the [ESPnet-Enh repository](https://github.com/espnet/espnet).

---

## 📚 Citation

If you find this work useful in your research, please consider citing:

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

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the authors.
