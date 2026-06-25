import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys, argparse, glob, datetime
import pandas as pd
import numpy as np
import random
import torch.backends.cudnn as cudnn
import json
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# =========================================================
# 1. Dataset (基于 Sample Sheet 的图像与 RNA 三方完美对齐版本)
# =========================================================
class MultiModalBagDataset(Dataset):
    def __init__(self, train_path, rna_df, sheet_df, args, split):
        self.train_path = train_path
        self.args = args
        self.split = split
        
        # 建立一个从“患者ID/图像UUID”到“RNA特征向量”的快捷查找字典
        self.slide_to_rna = {}
        
        print(f"[{split.upper()}] Mapping Image IDs to RNA features via Sample Sheet...")
        
        # 遍历 sample sheet 建立桥梁
        for _, row in sheet_df.iterrows():
            rna_file = row["File Name"]
            
            # 确保这个 rna_file 确实存在于我们提取的 Geneformer 矩阵里
            if rna_file in rna_df.index:
                rna_vector = rna_df.loc[rna_file].values
                
                # 提取核心匹配字段进行患者级别对齐 (例如从 'TCGA-43-3394-01A' 提取出 'TCGA-43-3394')
                if "Sample ID" in row:
                    patient_id = "-".join(str(row["Sample ID"]).split("-")[:3])
                    self.slide_to_rna[patient_id] = rna_vector
                elif "Case ID" in row:
                    self.slide_to_rna[str(row["Case ID"])] = rna_vector

    def get_bag_feats(self, row):
        slide_id = str(row.iloc[0]) # 例如: TCGA-43-3394-01Z-00-DX1.4c2f49b9...
        label_val = row.iloc[1]

        # 1. 读取视觉图像特征
        pt_dir = "/root/autodl-tmp/MExD/ctranspath_feats/pt_files"
        pt_path = os.path.join(pt_dir, f"{slide_id}.pt")
        feats = torch.load(pt_path).float()
        
        # 2. 智能解析复杂的 slide_id 并对齐 RNA 特征
        # 提取当前切片的患者ID (前12位，例如切出 TCGA-43-3394)
        current_patient = "-".join(slide_id.split("-")[:3]) 
        
        # 提取当前切片的图像 UUID (点后面的部分)
        img_uuid = slide_id.split(".")[-1] if "." in slide_id else ""

        # 优先级查找：1. 图像UUID直接匹配  2. 患者ID匹配  3. 兜底补0
        if img_uuid in self.slide_to_rna:
            rna_vec = self.slide_to_rna[img_uuid]
        elif current_patient in self.slide_to_rna:
            rna_vec = self.slide_to_rna[current_patient]
        else:
            # 如果真的完全找不到配对的 RNA，用全 0 向量兜底，256位是 Geneformer 维度
            rna_vec = np.zeros(256) 

        rna_feats = torch.tensor(rna_vec).float()
        label = torch.tensor(int(label_val))

        return label, feats, rna_feats

    def __getitem__(self, idx):
        return self.get_bag_feats(self.train_path.iloc[idx])

    def __len__(self):
        return len(self.train_path)


# =========================================================
# 2. MODEL (创新点：双流自适应可学习门控多模态融合网络)
# =========================================================
class MultiModalGatedMIL(nn.Module):
    def __init__(self, img_dim, rna_dim, num_classes, joint_dim=512):
        super().__init__()
        
        # 图像流编码
        self.img_encoder = nn.Sequential(
            nn.Linear(img_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, joint_dim),
            nn.ReLU()
        )
        
        # RNA流编码
        self.rna_encoder = nn.Sequential(
            nn.Linear(rna_dim, 256),
            nn.ReLU(),
            nn.Linear(256, joint_dim),
            nn.ReLU()
        )
        
        # 动态门控网络 (Gating Mechanism)
        # 拼接两个模态的特征，输出一个 0~1 之间的权重标量
        self.gate = nn.Sequential(
            nn.Linear(joint_dim * 2, 1),
            nn.Sigmoid()
        )
        
        # 多模态分类头
        self.classifier = nn.Linear(joint_dim, num_classes)

    def forward(self, img_x, rna_x):
        # 1. 图像特征聚合 (Mean Pooling) 并映射到联合维度
        img_h = self.img_encoder(img_x)             # [Num_patches, joint_dim]
        img_proto = img_h.mean(dim=0, keepdim=True) # [1, joint_dim]
        
        # 2. RNA 特征映射到联合维度
        rna_proto = self.rna_encoder(rna_x)         # [1, joint_dim]
        
        # 3. 门控自适应融合
        combined = torch.cat([img_proto, rna_proto], dim=-1) # [1, joint_dim * 2]
        g = self.gate(combined)                     # 图像特征的信任权重 g
        
        # 动态加权互补融合
        multimodal_proto = g * img_proto + (1 - g) * rna_proto
        
        # 4. 分类预测
        logits = self.classifier(multimodal_proto)
        return {"logits": logits}


# =========================================================
# 3. TRAIN & TEST 核心循环
# =========================================================
def train(train_loader, milnet, criterion, optimizer, args):
    milnet.train()
    total_loss = 0

    for i, (bag_label, bag_feats, rna_feats) in enumerate(train_loader):
        bag_label = bag_label.cuda()
        bag_feats = bag_feats.cuda()
        rna_feats = rna_feats.cuda()

        # 图像特征安全 reshape
        if len(bag_feats.shape) == 3:
            bag_feats = bag_feats.squeeze(0)
        bag_feats = bag_feats.view(-1, bag_feats.shape[-1])
        
        # RNA 特征安全 reshape
        rna_feats = rna_feats.view(bag_label.size(0), -1)

        optimizer.zero_grad()

        # 模型前向传播 (同时喂入图像与RNA特征)
        output = milnet(bag_feats, rna_feats)
        bag_pred = output['logits']
        loss = criterion(bag_pred, bag_label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        sys.stdout.write(f"\rTrain [{i}/{len(train_loader)}] loss {loss.item():.4f}")

    return total_loss / len(train_loader)


def test(test_loader, milnet, criterion, args):
    milnet.eval()
    preds = []
    labels = []
    total_loss = 0

    with torch.no_grad():
        for i, (bag_label, bag_feats, rna_feats) in enumerate(test_loader):
            bag_label = bag_label.cuda()
            bag_feats = bag_feats.cuda()
            rna_feats = rna_feats.cuda()

            if len(bag_feats.shape) == 3:
                bag_feats = bag_feats.squeeze(0)
            bag_feats = bag_feats.view(-1, bag_feats.shape[-1])
            rna_feats = rna_feats.view(bag_label.size(0), -1)

            output = milnet(bag_feats, rna_feats)
            bag_pred = output['logits']

            loss = criterion(bag_pred, bag_label)
            total_loss += loss.item()

            pred = torch.argmax(bag_pred, dim=-1).cpu().numpy()
            preds.extend(pred)
            labels.extend(bag_label.cpu().numpy())

    acc = np.mean(np.array(preds) == np.array(labels))
    return total_loss / len(test_loader), acc


# =========================================================
# 4. MAIN 主函数
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TCGA-lung-luad-lusc')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--feat_type', type=str, default='vit')
    args = parser.parse_args()

    # 固定随机种子保证实验可重复性
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

    # -----------------------------------------------------
    # 数据载入与三方对齐准备
    # -----------------------------------------------------
    # 1. 载入先前用 Geneformer 提取出来的 256 维 RNA 特征
    rna_csv_path = "/root/autodl-tmp/project/geneformer_rna_embedding.csv"
    if not os.path.exists(rna_csv_path):
        raise FileNotFoundError(f"未找到基因特征文件：{rna_csv_path}，请先运行特征提取脚本！")
    rna_df = pd.read_csv(rna_csv_path, index_col=0)
    rna_dim = rna_df.shape[1]
    print(f"-> Loaded Geneformer RNA features. Shape: {rna_df.shape}")

    # 2. 载入原始的 sample_sheet 作为图像与 RNA 的对齐桥梁
    SHEET_PATH = "/root/autodl-tmp/project/gdc_sample_sheet.2026-06-23.tsv" 
    sheet_df = pd.read_csv(SHEET_PATH, sep="\t")

    # 3. 载入图像分类表格
    csv_path = os.path.join(args.feat_type + "_datasets_new_384_tcga", args.dataset + ".csv")
    data = pd.read_csv(csv_path)

    # 划分训练集、测试集
    train_path, test_path = train_test_split(
        data, test_size=0.2, stratify=data.iloc[:, 1], random_state=args.seed
    )

    # 4. 实例化多模态 Dataset
    train_set = MultiModalBagDataset(train_path, rna_df, sheet_df, args, 'train')
    test_set = MultiModalBagDataset(test_path, rna_df, sheet_df, args, 'test')

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # 自动探测图像视觉特征的维度 (例如 CTransPath 为 768 维)
    sample_pt = torch.load("/root/autodl-tmp/MExD/ctranspath_feats/pt_files/" + f"{data.iloc[0,0]}.pt")
    img_dim = sample_pt.shape[-1]
    print(f"-> Detected Image visual feature dim: {img_dim}")

    # -----------------------------------------------------
    # 模型初始化与训练
    # -----------------------------------------------------
    # 实例化多模态门控模型
    milnet = MultiModalGatedMIL(img_dim=img_dim, rna_dim=rna_dim, num_classes=args.num_classes).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, weight_decay=1e-5)

    best_acc = 0
    print("\n====== Starting Multimodal Gated MIL Training ======")
    for epoch in range(args.num_epochs):
        train_loss = train(train_loader, milnet, criterion, optimizer, args)
        test_loss, acc = test(test_loader, milnet, criterion, args)

        print(f"\nEpoch {epoch}: train_loss {train_loss:.4f} | test_loss {test_loss:.4f} | test_acc {acc:.4f}")

        # 保存表现最好的模型
        if acc > best_acc:
            best_acc = acc
            torch.save(milnet.state_dict(), "best_multimodal_model.pth")
            print(f"==> Saved new best model with acc: {best_acc:.4f}")

    print(f"\nTraining Finished! Best Test Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()