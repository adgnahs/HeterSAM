# HeterSAM

Prompt-Free Heterogeneous Fusion for Medical Image Segmentation

## Overview

**HeterSAM** upgrades SAM for medical image segmentation **without fine-tuning SAM’s image encoder or mask head**. It introduces a **Swin-based dense prompt encoder (U-shaped)** and a **Deformable two-way cross-attention module with an SE-Gate**, replacing manual points/boxes with image-driven prompts to enable **zero-interaction** segmentation. Only the prompt and fusion paths are trained, keeping inference lightweight and deployment friendly, while achieving competitive results across diverse medical and video benchmarks.

**Highlights**

* **Dense Prompt Encoder (Swin-U):** Generates structured prompt embeddings from the image itself, removing manual prompt dependency.
* **Deformable Two-Way Cross-Attention:** Sparse offset sampling + **SE-Gate** to suppress attention diffusion and sharpen boundary alignment.
* **Frozen SAM Backbone:** Stable transfer; train only prompt/fusion branches.
* **Low-latency path:** Optional lightweight decoder for near-lossless speedups.

## Relation to the Manuscript

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17548104.svg)](https://doi.org/10.5281/zenodo.17548104)

This repository contains the **official implementation directly associated** with our manuscript submitted to *The Visual Computer*:

**Enhancing Medical Image Segmentation with Heterogeneous Representation Alignment and Deformable Attention**  
 Submission ID ：**f8cdd24e-a6aa-431f-9a13-bde323a6ab9a**

We release the code and permanent artifacts to support transparency and reproducibility.  
Please **cite the manuscript  and the Zenodo DOI below** if you use this code or its derivatives.

- GitHub: https://github.com/adgnahs/HeterSAM  
- Zenodo DOI: https://doi.org/10.5281/zenodo.17548104


## Datasets

Example datasets used in our experiments:


* [polyp](https://pan.baidu.com/s/1o9LE4H55f8MtRWEw_pP2FQ?pwd=2wjn)
* [ISIC2018](https://pan.baidu.com/s/1wXmd-mfuw3eGnpBLzFnzRg?pwd=n9m8)
* [Verse2020](https://pan.baidu.com/s/1A_8YjO8kYm8mq1mTe7V4ig?pwd=nedu)
* [Spine1K](https://pan.baidu.com/s/1gv7pn-FMwGdGf3WLRLJydg?pwd=e99v)

> You may also extend to ISIC 2018, PH2, Kvasir-SEG, ClinicDB, ColonDB, ETIS, Spine1K, VerSe2020, etc.

## SAM Checkpoints

* [SAM ViT-B (base)](https://drive.google.com/file/d/1ZwKc-7Q8ZaHfbGVKvvkz_LPBemxHyVpf/view?usp=drive_link)
* [SAM ViT-L (large)](https://drive.google.com/file/d/16AhGjaVXrlheeXte8rvS2g2ZstWye3Xx/view?usp=drive_link)
* [SAM ViT-H (huge)](https://drive.google.com/file/d/1tFYGukHxUCbCG3wPtuydO-lYakgpSYDd/view?usp=drive_link)

Place them under `checkpoints/` or pass via `--sam_ckpt`.

## Usage

### 1) Clone

```bash
git clone https://github.com/adgnahs/HeterSAM.git
cd HeterSAM
```

### 2) Environment

```bash
conda create -n hetersam python=3.10 -y
conda activate hetersam
pip install -r requirements.txt
```

### 3) Data Layout (example)

```
data/
  monu/
    images/*.png
    masks/*.png
  polyp/
    images/*.png
    masks/*.png
```

> Override paths via CLI args or configs as needed.

### 4) Training

```bash
# Example: train on monu with SAM ViT-B
python train.py \
  --dataset monu \
  --data_root ./data/monu \
  --sam_ckpt ./checkpoints/sam_vit_b.pth \
  --epochs 300 --batch_size 8 --lr 1e-4 \
  --save_dir runs/monu_vitb
```

### 5) Inference

```bash
python infer.py \
  --input ./samples/monu_001.png \
  --sam_ckpt ./checkpoints/sam_vit_b.pth \
  --ckpt runs/monu_vitb/best.pth \
  --output ./outputs/monu_001_mask.png
```

## Repository Structure

```
HeterSAM/
├─ train.py                 # Training entry
├─ inference.py                 # Inference script
├─ models_swin/
│  ├─ base.py               # Core layers / utilities
│  ├─ model_single.py       # Main model & forward
│  ├─ hardnet.py            # Optional/early prompt encoder
│  └─ swin_prompt_encoder/  # Swin-U dense prompt encoder
├─ segment_anything/        # SAM encoder decoder
├─ fusion/
│  └─ dca.py                # Deformable Two-Way Cross-Attention (with SE-Gate)
├─ configs/                 # Dataset & training configs
├─ cp/                      # SAM and trained weights
└─ dataset/                 # Your datasets
```

## Tips

* By default, SAM’s image encoder and mask head are **frozen**; only prompt/fusion branches are trained.
* For limited GPU memory: prefer ViT-B + AMP (`--amp`) + gradient accumulation (`--grad_accum`).
* Post-processing (e.g., CRF) and input sizes can be set per dataset in `configs/`.

## How to Citation

Please cite if this repository helps your research :

**(A) Manuscript (submitted to The Visual Computer) **
```bibtex
@article{HeterSAM_TVC_2025_submitted,
  title   = {Enhancing Medical Image Segmentation with Heterogeneous Representation Alignment and Deformable Attention},
  author  = {<Your Name> and <Coauthors>},
  journal = {The Visual Computer},
  note    = {submitted},
  year    = {2025}
}
