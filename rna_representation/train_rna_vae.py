import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#############################################
# Step1 读取 Sample Sheet 并构建标签映射
#############################################
# 假设你的表格名字是 gdc_sample_sheet.tsv 且是以制表符分隔
SHEET_PATH = "./gdc_sample_sheet.2026-06-23.tsv" 
sheet_df = pd.read_csv(SHEET_PATH, sep="\t")

# 过滤：只保留肿瘤样本 (根据你的实际需求，一般推荐只对比癌种)
sheet_df = sheet_df[sheet_df["Tissue Type"] == "Tumor"]

# 构建 文件名 -> 标签(LUSC=1, LUAD=0) 的映射字典
label_dict = {}
for _, row in sheet_df.iterrows():
    file_name = row["File Name"]
    project_id = row["Project ID"]
    
    if project_id == "TCGA-LUSC":
        label_dict[file_name] = 1
    elif project_id == "TCGA-LUAD":
        label_dict[file_name] = 0

print(f"Sample sheet loaded. Total tumor samples matched: {len(label_dict)}")

#############################################
# Step2 读取 RNA 文件夹
#############################################
RNA_DIR = "./RNA"
all_samples = {}
labels_list = []

print("Loading RNA files...")

for file in os.listdir(RNA_DIR):
    if not file.endswith(".tsv"):
        continue
    
    # 如果这个文件不在我们过滤后的 label_dict 里（比如是Normal或者其他项目），就跳过
    if file not in label_dict:
        continue

    path = os.path.join(RNA_DIR, file)
    df = pd.read_csv(path, sep="\t", comment="#")

    # 过滤非编码基因
    df = df[df["gene_id"].astype(str).str.startswith("ENSG", na=False)]
    df = df[df["gene_type"] == "protein_coding"]

    # 使用完整文件名作为 Key，方便后续对齐标签
    all_samples[file] = df.set_index("gene_name")["tpm_unstranded"]
    labels_list.append(label_dict[file])

# 构建矩阵：行是基因，列是样本
expr = pd.DataFrame(all_samples)
print("Raw matrix shape (Genes x Samples):", expr.shape)

#############################################
# Step3 空值填充、Log转换、高变基因筛选(HVG)
#############################################
expr = expr.fillna(0)
expr = np.log2(expr + 1)

# 筛选前 5000 个高变基因
gene_var = expr.var(axis=1)
top_genes = gene_var.nlargest(5000).index
expr = expr.loc[top_genes]

# 转置：让行变成样本 [Samples, 5000]
X = expr.T.values
y = np.array(labels_list)
print(f"Final Input X shape: {X.shape}, y shape: {y.shape}")
print(f"Class distribution - LUSC(1): {np.sum(y==1)}, LUAD(0): {np.sum(y==0)}")

#############################################
# Step4 严格划分数据集（防止数据泄露 🌟）
#############################################
# 按照 8:2 划分训练集和测试集（你也可以做三路划分 Train/Val/Test）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 只在训练集上 fit，然后同时 transform 训练集和测试集
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

input_dim = X_train.shape[1]

# 构建训练和测试的 DataLoader (现在包含 y 了)
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# 测试集不需要 shuffle
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#############################################
# Step5 Supervised VAE 模型定义（轻量化版 🌟）
#############################################
class SupervisedVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=128, num_classes=2):
        super().__init__()

        # 调小了隐藏层维度 (5000 -> 512 -> 256 -> 128)，并加入了 Dropout 防止过拟合
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # 解码器（用于重构基因）
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        
        # 🌟 新增：辅助分类头 (Auxiliary Classifier)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        
        # 同时预测分类标签
        pred_logits = self.classifier(z)
        return recon, mu, logvar, z, pred_logits

#############################################
# Step6 联合损失函数 (Joint Loss)
#############################################
def supervised_vae_loss(recon, x, mu, logvar, pred_logits, y, beta=0.001, alpha=1.0):
    # 1. 重构损失
    recon_loss = nn.functional.mse_loss(recon, x)
    
    # 2. KL 散度
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 3. 分类损失 (交叉熵)
    ce_loss = nn.functional.cross_entropy(pred_logits, y)
    
    # 联合总损失
    total_loss = recon_loss + beta * kl + alpha * ce_loss
    return total_loss, recon_loss, ce_loss

#############################################
# Step7 模型训练
#############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SupervisedVAE(input_dim=input_dim, latent_dim=128, num_classes=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

epochs = 60 # 轻量化后不需要 100 epoch，60次左右即可稳定

print("Start Training Supervised VAE...")
for epoch in range(epochs):
    model.train()
    train_loss, train_recon, train_ce = 0, 0, 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        recon, mu, logvar, z, pred_logits = model(batch_x)
        
        loss, r_loss, c_loss = supervised_vae_loss(recon, batch_x, mu, logvar, pred_logits, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_recon += r_loss.item()
        train_ce += c_loss.item()
        
        # 计算训练集分类准确率
        preds = torch.argmax(pred_logits, dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)
        
    print(f"Epoch {epoch+1}/{epochs} | Total Loss: {train_loss:.4f} | Recon Loss: {train_recon:.4f} | CE Loss: {train_ce:.4f} | Train Acc: {correct/total:.2%}")

#############################################
# Step8 在测试集上评估（严格的非作弊验证 🌟）
#############################################
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        _, _, _, _, pred_logits = model(batch_x)
        preds = torch.argmax(pred_logits, dim=1)
        test_correct += (preds == batch_y).sum().item()
        test_total += batch_y.size(0)

print(f"==> VAE Classifier Test Accuracy: {test_correct/test_total:.2%}")

#############################################
# Step9 提取并保存特征（全集独立提取 🌟）
#############################################
print("Extracting latent embeddings for all samples...")
all_X_tensor = torch.FloatTensor(scaler.transform(X)).to(device)

with torch.no_grad():
    # 核心：只使用 encode 提取均值 mu，彻底砍掉解码器和分类标签
    mu, _ = model.encode(all_X_tensor)
    embedding = mu.cpu().numpy()

# 保存结果，保持原来的文件名 index，方便后续和 WSI 图像特征按文件名配对
embedding_df = pd.DataFrame(embedding, index=expr.columns)
embedding_df.to_csv("supervised_rna_embedding_128.csv")
torch.save(model.state_dict(), "supervised_rna_vae.pth")
print("Saved supervised_rna_embedding_128.csv and model.")