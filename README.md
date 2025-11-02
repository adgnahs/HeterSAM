# HeterSAM

Prompt-Free Heterogeneous Fusion for Medical Image Segmentation

## Overview

HeterSAM 在 **不微调 SAM 图像编码器与 mask head** 的前提下，引入 **基于 Swin 的稠密提示编码器（U 型结构）** 与 **Deformable 双向 Cross-Attention（含 SE-Gate）**，用影像自身作为“提示”替代人工点/框，实现**零交互**医学图像分割。仅训练提示与融合路径，推理轻量、部署友好，并在多项医学与视频基准上取得具有竞争力的表现。

**Key features**

* **Dense Prompt Encoder（Swin-U）**：由图像自引导生成结构化提示嵌入，摆脱人工提示依赖
* **Deformable Two-Way Cross-Attention**：稀疏可学习偏移采样 + SE-Gate 抑制注意力扩散、强化边界对齐
* **Frozen SAM Backbone**：冻结 SAM 图像编码器与解码头，仅训练提示/融合分支，稳定且易于迁移
* **低时延解码**：提供轻量推理路径，几乎不损伤精度

## Paper

* Preprint（占位）：`TBD`
* 如果您已有 arXiv/期刊链接，直接替换上面的 `TBD`。

## Datasets

本项目实验覆盖多类医学场景；示例数据集如下（与示例一致先列 3 个，更多可在 configs 中扩展）：

* [monu](https://drive.google.com/drive/folders/1bzyHsDWhjhiwzpx_zJ5dpMG3-5F-nhT4?usp=drive_link)
* [glas](https://drive.google.com/drive/folders/1z9xBesNhvuM08yUOpOWcUy7OnBGHenFv?usp=drive_link)
* [polyp](https://drive.google.com/drive/folders/1S11HsauwKO206CPzrGBnTid-nbQMhbZz?usp=drive_link)

> 可选扩展：ISIC 2018、PH2、Kvasir-SEG、ClinicDB、ColonDB、ETIS、Spine1K、VerSe2020 等。

## SAM checkpoints

* [SAM ViT-B (base)](https://drive.google.com/file/d/1ZwKc-7Q8ZaHfbGVKvvkz_LPBemxHyVpf/view?usp=drive_link)
* [SAM ViT-L (large)](https://drive.google.com/file/d/16AhGjaVXrlheeXte8rvS2g2ZstWye3Xx/view?usp=drive_link)
* [SAM ViT-H (huge)](https://drive.google.com/file/d/1tFYGukHxUCbCG3wPtuydO-lYakgpSYDd/view?usp=drive_link)

> 将下载的权重放到 `checkpoints/`（或通过 `--sam_ckpt` 指定路径）。

## Usage

1. **Clone**

   ```bash
   git clone https://github.com/your_username/HeterSAM.git
   cd HeterSAM
   ```

2. **Conda & Requirements**

   ```bash
   conda create -n hetersam python=3.10 -y
   conda activate hetersam
   pip install -r requirements.txt
   ```

3. **Data准备（示例）**

   ```
   data/
     monu/
       images/*.png
       masks/*.png
     glas/
       images/*.png
       masks/*.png
     polyp/
       images/*.png
       masks/*.png
   ```

   > 路径可通过启动参数或配置文件覆盖。

4. **Training（示例命令）**

   ```bash
   # 以 monu 为例，指定 SAM 权重与数据根目录
   python train.py \
     --dataset monu \
     --data_root ./data/monu \
     --sam_ckpt ./checkpoints/sam_vit_b.pth \
     --epochs 300 --batch_size 8 --lr 1e-4 \
     --save_dir runs/monu_vitb
   ```

5. **Evaluation**

   ```bash
   python eval.py \
     --dataset monu \
     --data_root ./data/monu \
     --sam_ckpt ./checkpoints/sam_vit_b.pth \
     --ckpt runs/monu_vitb/best.pth \
     --save_dir runs/monu_vitb_eval
   ```

6. **Inference（单图/文件夹）**

   ```bash
   python infer.py \
     --input ./samples/monu_001.png \
     --sam_ckpt ./checkpoints/sam_vit_b.pth \
     --ckpt runs/monu_vitb/best.pth \
     --output ./outputs/monu_001_mask.png
   ```

## Repository Structure（简要）

```
HeterSAM/
├─ train.py                # 训练入口
├─ eval.py                 # 评测脚本
├─ infer.py                # 推理脚本
├─ models/
│  ├─ base.py              # 基础模块/通用层
│  ├─ model_single.py      # 主干组网 & 前向逻辑
│  ├─ hardnet.py           # 早期/可选提示编码器实现
│  └─ swin_prompt_encoder/ # Swin-U 稠密提示编码器
├─ fusion/
│  └─ dca.py               # Deformable Two-Way Cross-Attention (含 SE-Gate)
├─ configs/                # 数据集与训练配置
├─ checkpoints/            # 放置 SAM 权重或训练权重
└─ data/                   # 数据目录（自备）
```

## Citation

如果本仓库对您的研究有帮助，请引用（占位示例，按实际论文更新）：

```bibtex
@article{your2025hetersam,
  title   = {HeterSAM: Prompt-Free Heterogeneous Fusion for Medical Image Segmentation},
  author  = {Your Name and Coauthors},
  journal = {TBD},
  year    = {2025}
}
```

## Acknowledgements

* [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
* 相关工作：AutoSAM 等

---

> 小贴士：
>
> * 默认冻结 SAM 的 image encoder 与 mask head，仅训练提示与融合模块；如需端到端，可在配置中开启 `--unfreeze_sam`（若实现）。
> * 不同数据集的输入尺寸、归一化与后处理（如 CRF/Post-Refine）可在 `configs/` 中单独指定。
> * 若显存受限，建议使用 ViT-B + 混合精度（`--amp`）与梯度累积（`--grad_accum`）。
