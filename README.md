# 📂 Data Preprocessing

This directory contains the pipeline for multi-modal data alignment, filtering, and feature extraction. The workflow handles both **Whole Slide Images (WSIs)** and **Bulk RNA-seq Transcriptomics Data** to construct paired multi-modal tokens for the downstream conditional MoE network.

---

## 🔬 Part 1: Pathology WSI Feature Extraction

The pathology pipeline follows the structural paradigm of **CLAM** for tissue segmentation, patching, and multi-instance vision feature embedding.

### 1. Adaptive Tissue Segmentation and Patching
Automatically segment tissue regions from WSI files (`.svs`) and crop them into non-overlapping patches of $256 \times 256$ pixels.
```bash
python create_patches_fp.py \
    --source DATA_DIRECTORY \
    --save_dir RESULTS_DIRECTORY \
    --patch_size 256 \
    --seg \
    --process_list CSV_FILE_NAME \
    --patch \
    --stitch
