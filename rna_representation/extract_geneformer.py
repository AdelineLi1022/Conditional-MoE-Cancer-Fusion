import os
import pickle
import numpy as np
import pandas as pd
import torch
from transformers import BertModel

#############################################
# Step1: 载入并过滤样本 (TCGA LUSC/LUAD 标签对齐)
#############################################
SHEET_PATH = "./gdc_sample_sheet.2026-06-23.tsv" 
sheet_df = pd.read_csv(SHEET_PATH, sep="\t")
# 过滤：只保留肿瘤样本
sheet_df = sheet_df[sheet_df["Tissue Type"] == "Tumor"]

label_dict = {}
for _, row in sheet_df.iterrows():
    file_name = row["File Name"]
    project_id = row["Project ID"]
    if project_id == "TCGA-LUSC":
        label_dict[file_name] = 1
    elif project_id == "TCGA-LUAD":
        label_dict[file_name] = 0

RNA_DIR = "./RNA"
valid_files = [f for f in os.listdir(RNA_DIR) if f in label_dict]
print(f"Matched {len(valid_files)} valid tumor files for Geneformer extraction.")

#############################################
# Step2: 读取本地的 Geneformer 字典与模型权重
#############################################
LOCAL_MODEL_DIR = "./geneformer_weights"  
# 使用你刚刚下载并确认的新版字典文件名 🌟
LOCAL_DICT_PATH = os.path.join(LOCAL_MODEL_DIR, "token_dictionary_gc104M.pkl")

print(f"Loading Geneformer weights and dictionary from local path: {LOCAL_MODEL_DIR}...")

# 1. 载入本地新版字典
with open(LOCAL_DICT_PATH, "rb") as f:
    gene_token_dict = pickle.load(f)

# 2. 从本地加载模型结构与权重 (自动兼容 .bin 或 .safetensors)
model = BertModel.from_pretrained(LOCAL_MODEL_DIR, output_hidden_states=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Local Geneformer model loaded successfully on {device}.")

#############################################
# Step3: 核心提取循环
#############################################
all_embeddings = []
saved_file_names = []

# Geneformer 默认的最大输入长度是 2048 个基因
MAX_LEN = 2048 

print("Starting feature extraction...")
for file_idx, file in enumerate(valid_files):
    path = os.path.join(RNA_DIR, file)
    df = pd.read_csv(path, sep="\t", comment="#")

    # 1. 清洗并保留有 Ensembl ID 的蛋白编码基因
    df = df[df["gene_id"].astype(str).str.startswith("ENSG", na=False)]
    df = df[df["gene_type"] == "protein_coding"]
    
    # 2. 去除版本号（如 ENSG00000141510.14 -> ENSG00000141510）以对齐大模型字典
    df["pure_gene_id"] = df["gene_id"].apply(lambda x: str(x).split(".")[0])
    
    # 3. 过滤：只保留表达量（TPM）大于0，且在 Geneformer 字典里存在的基因
    df = df[df["tpm_unstranded"] > 0]
    df = df[df["pure_gene_id"].isin(gene_token_dict)]
    
    if len(df) == 0:
        print(f"Warning: No valid genes found for file {file}, skipping.")
        continue
        
    # 4. Geneformer 核心逻辑：按表达量从高到低排序 (Rank) 🌟
    df_sorted = df.sort_values(by="tpm_unstranded", ascending=False)
    
    # 5. 截取前 2048 个最高表达的基因，并转换为 Token ID
    top_genes = df_sorted["pure_gene_id"].head(MAX_LEN).tolist()
    tokens = [gene_token_dict[gene] for gene in top_genes]
    
    # 6. Padding 掩码处理：如果有效基因不够 2048 个，后面补 0
    if len(tokens) < MAX_LEN:
        attention_mask = [1] * len(tokens) + [0] * (MAX_LEN - len(tokens))
        tokens = tokens + [0] * (MAX_LEN - len(tokens))
    else:
        tokens = tokens[:MAX_LEN]
        attention_mask = [1] * MAX_LEN

    # 7. 转为 Tensor 并送入大模型
    input_ids_tensor = torch.LongTensor([tokens]).to(device)
    mask_tensor = torch.LongTensor([attention_mask]).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids_tensor, attention_mask=mask_tensor)
        
        # 8. 特征聚合 (Mean Pooling)：对模型最后一层所有有效基因 Token 求平均 🌟
        # outputs.last_hidden_state 维度是 [1, 2048, 256] 或者是 512/768（取决于大模型隐藏层维度）
        last_hidden = outputs.last_hidden_state[0] 
        
        # 排除 Padding 的 0，只对真实表达的基因做平均
        valid_count = sum(attention_mask)
        mean_embedding = last_hidden[:valid_count].mean(dim=0).cpu().numpy()

    all_embeddings.append(mean_embedding)
    saved_file_names.append(file)
    
    if (file_idx + 1) % 10 == 0 or (file_idx + 1) == len(valid_files):
        print(f"Processed [{file_idx + 1}/{len(valid_files)}] samples.")

#############################################
# Step4: 保存大模型提取出的 RNA 特征
#############################################
embedding_df = pd.DataFrame(all_embeddings, index=saved_file_names)
embedding_df.to_csv("geneformer_rna_embedding.csv")

print("\n====== Extraction Finished ======")
print(f"Saved: geneformer_rna_embedding.csv")
print(f"Embedding matrix shape: {embedding_df.shape}")
print("每个 RNA 样本已经成功转为了预训练大模型级别的稳健特征编码！")