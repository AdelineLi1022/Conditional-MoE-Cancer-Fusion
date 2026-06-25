# Conditional-MoE-Cancer-Fusion
#Data Preprocessing
This directory contains the pipeline for multi-modal data alignment, filtering, and feature extraction. The workflow handles both Whole Slide Images (WSIs) and Bulk RNA-seq Transcriptomics Data to construct paired multi-modal tokens for the downstream conditional MoE network.
🔬 Part 1: Pathology WSI Feature ExtractionThe pathology pipeline follows the structural paradigm of CLAM for tissue segmentation, patching, and multi-instance vision feature embedding.1. Adaptive Tissue Segmentation and PatchingAutomatically segment tissue regions from WSI files (.svs) and crop them into non-overlapping patches of $256 \times 256$ pixels.Bashpython create_patches_fp.py \
    --source DATA_DIRECTORY \
    --save_dir RESULTS_DIRECTORY \
    --patch_size 256 \
    --seg \
    --process_list CSV_FILE_NAME \
    --patch \
    --stitch
2. Deep Vision Feature EmbeddingExtract high-level visual representations at the patch level using two parallel pre-trained visual backbones: CTransPath and MoCo-ViT.Bashpython extract_features_fp.py \
    --data_h5_dir DIR_TO_COORDS \
    --data_slide_dir DATA_DIRECTORY \
    --csv_path CSV_FILE_NAME \
    --feat_dir FEATURES_DIRECTORY \
    --batch_size 512 \
    --slide_ext .svs
Note: Download the pre-trained weights (TransPath) and utilize the customized timm environment as configured in the project baseline.
🧬 Part 2: Transcriptomics (RNA-seq) RepresentationTo compress high-dimensional genomics data and align it with visual tokens, we explore two parallel molecular representation strategies after matching the clinical file metadata with the gdc_sample_sheet.tsv and filtering for non-tumor tissues.Pre-filtering Common to Both Pipelines:Cohort Alignment: Filters and pairs samples strictly matching TCGA-LUAD (Label: 0) and TCGA-LUSC (Label: 1).Biotype Filtering: Excludes non-coding RNAs, retaining strictly human protein-coding genes prefixed with ENSG.🚀 Paradigm A: Task-Specific Latent Compression via Supervised VAEThis route trains a Supervised Variational Autoencoder (Supervised VAE) from scratch. By adding an auxiliary classification head over the latent layer $z$, the encoder is forced to discard metabolic background noise and retain highly distinct tumor-typing signals.[Raw TPM] -> [Log2(TPM+1)] -> [5000 HVGs] -> [Supervised VAE] -> [128-dim Latent z]
Pipeline Workflow:Abundance Log-Normalization: Modulates the heavily skewed distribution of the raw TPM matrix using $y = \log_2(\text{TPM} + 1)$.Highly Variable Genes (HVG) Selection: Dynamically calculates the variance across all cohorts to adaptively retrieve the top 5,000 highly variable genes.Joint Loss Optimization: Trains the network by minimizing a multi-task loss function containing Reconstruction Error (MSE), Kullback-Leibler (KL) divergence, and Cross-Entropy classification loss:$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{KL}} + \alpha \mathcal{L}_{\text{CE}}$$Run VAE Training & Feature Saving:Bashpython train_rna_vae.py
Output: Saves the compressed task-specific representations into supervised_rna_embedding_128.csv and the checkpoint state dict into supervised_rna_vae.pth.🚀 Paradigm B: Global Context Attention Extraction via GeneformerThis route scales up the molecular modeling into the Foundation Model regime by migrating the pre-trained attention stack from Geneformer (a transformer base model trained on large-scale single-cell transcriptomics data).[TPM Sorting] -> [Rank-to-Token (Top 2048)] -> [Geneformer Stack] -> [Mean Pooling] -> [256/512-dim Context Embedding]
Pipeline Workflow:Rank Transformation: Instead of feeding raw matrix quantities, genes are sorted by their expression values in descending order (timm_unstranded > 0), aligning with Geneformer's rank-based vocabulary.Tokenization & Padding: Retrieves the top 2,048 highly expressed Ensembl IDs, maps them to token indices via token_dictionary_gc104M.pkl, and pads sequences shorter than MAX_LEN with 0 masks.Contextual Encoding & Mean Pooling: Feeds tokens into the local BertModel block. The last hidden state layers (outputs.last_hidden_state) are dynamically averaged across non-zero active masks to yield the robust embedding matrix.Run Geneformer Extraction:Bashpython extract_geneformer.py
Output: Generates geneformer_rna_embedding.csv, representing foundational cellular state priors for each case ID.
