import os
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ==========================================
# 1. 升级版 DATALOADER (提取第20步的类别作为 Target)
# ==========================================
class TaobaoDINDataset(Dataset):
    def __init__(self, pkl_file):
        logging.info("正在加载序列数据到内存中...")
        self.df = pd.read_pickle(pkl_file)
        # 过滤无效空序列
        self.df = self.df[self.df["behavior_seq"].apply(lambda x: sum(x) > 0)]
        self.df.reset_index(drop=True, inplace=True)
        logging.info(f"成功加载有效序列总数: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 历史特征（前19步的类目与行为）
        cat_seq = torch.tensor(row["category_seq"][:-1], dtype=torch.long)
        beh_seq = torch.tensor(row["behavior_seq"][:-1], dtype=torch.long)

        # 🛑 核心升级：提取第20步的商品类目作为 DIN 局部激活的 Target Query
        target_cat = torch.tensor(row["category_seq"][-1], dtype=torch.long)

        # 目标预测标签（第20步的真实行为动作，映射至 0-3）
        target_beh = torch.tensor(row["behavior_seq"][-1] - 1, dtype=torch.long)
        target_beh = torch.clamp(target_beh, min=0)

        return cat_seq, beh_seq, target_cat, target_beh


# ==========================================
# 2. DIN 核心架构 (Deep Interest Network)
# ==========================================
class AttentionUnit(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionUnit, self).__init__()
        # 严格复现阿里 DIN 论文的外积交叉特征拼接：[Query, Fact, Query-Fact, Query*Fact]
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 4, 64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, query, facts):
        # query: [B, embed_dim] (目标商品向量)
        # facts: [B, T, embed_dim] (历史行为序列向量)
        B, T, E = facts.size()

        # 将 target 项进行沿时间轴 T 的广播复制
        query = query.unsqueeze(1).expand(-1, T, -1)  # [B, T, E]

        # 核心交互特征拼接
        combined = torch.cat(
            [query, facts, query - facts, query * facts], dim=-1
        )  # [B, T, E*4]

        # 计算局部激活权重分数
        scores = self.fc(combined)  # [B, T, 1]
        return scores.squeeze(-1)  # [B, T]


class TaobaoDIN(nn.Module):
    def __init__(
        self, num_categories=10000, num_behaviors=5, embed_cat_dim=32, embed_beh_dim=16
    ):
        super(TaobaoDIN, self).__init__()

        # 基础特征嵌入层
        self.cat_embedding = nn.Embedding(num_categories, embed_cat_dim)
        self.beh_embedding = nn.Embedding(num_behaviors, embed_beh_dim)

        total_embed_dim = embed_cat_dim + embed_beh_dim  # 32 + 16 = 48

        # 激活注意力单元
        self.attention = AttentionUnit(total_embed_dim)

        # 最终预测多分类概率的深度网络 (MLP)
        self.fc = nn.Sequential(
            nn.Linear(total_embed_dim * 2, 64),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4),  # 输出 4 种转化动作的对数几率
        )

    def forward(self, cat_seq, beh_seq, target_cat):
        # 1. 映射历史时序特征表征
        cat_embs = self.cat_embedding(cat_seq)  # [B, T, 32]
        beh_embs = self.beh_embedding(beh_seq)  # [B, T, 16]
        hist_embs = torch.cat([cat_embs, beh_embs], dim=-1)  # [B, T, 48]

        # 2. 映射目标预测项的表征（因第20步动作未知，我们用全零占位动作与类目拼接保持特征对齐）
        target_cat_emb = self.cat_embedding(target_cat)  # [B, 32]
        target_beh_placeholder = torch.zeros(
            (target_cat.size(0), 16), device=target_cat.device
        )
        target_emb = torch.cat(
            [target_cat_emb, target_beh_placeholder], dim=-1
        )  # [B, 48]

        # 3. 计算 Local Activation 注意力权重并执行 Softmax 归一化
        attn_weights = self.attention(target_emb, hist_embs)  # [B, T]
        attn_weights = torch.softmax(attn_weights, dim=-1)  # [B, T]

        # 4. 注意力加权池化 (Attention Based Sum Pooling) 替代传统强行压缩的 RNN
        # [B, 1, T] x [B, T, 48] -> [B, 1, 48] -> [B, 48]
        user_interest = torch.bmm(attn_weights.unsqueeze(1), hist_embs).squeeze(1)

        # 5. 聚合当前意图特征，通过顶层多层感知机计算多分类概率
        final_input = torch.cat([user_interest, target_emb], dim=-1)  # [B, 48 * 2]
        output = self.fc(final_input)
        return output


# ==========================================
# 3. 自动化训练引擎
# ==========================================
def train_din_model():
    try:
        # 动态绝对路径硬核降噪，防止任何路径迷路错误
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_file = os.path.join(
            project_root, "data", "processed", "user_sequences.pkl"
        )
        model_save_path = os.path.join(project_root, "models", "taobao_din_model.pth")

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logging.info(f"DIN 模型就绪，正在激活硬件加速设备: [{device.type.upper()}]")

        dataset = TaobaoDINDataset(data_file)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

        model = TaobaoDIN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        epochs = 3

        logging.info("正全面点火运行大厂生产级 DIN 推荐系统架构...")
        for epoch in range(epochs):
            model.train()
            total_loss, correct_predictions, total_samples = 0, 0, 0

            for batch_idx, (cat_seq, beh_seq, target_cat, target_beh) in enumerate(
                dataloader
            ):
                cat_seq, beh_seq, target_cat, target_beh = (
                    cat_seq.to(device),
                    beh_seq.to(device),
                    target_cat.to(device),
                    target_beh.to(device),
                )

                optimizer.zero_grad()
                predictions = model(cat_seq, beh_seq, target_cat)
                loss = criterion(predictions, target_beh)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted_classes = torch.max(predictions, 1)
                correct_predictions += (predicted_classes == target_beh).sum().item()
                total_samples += target_beh.size(0)

                if batch_idx % 50 == 0 and batch_idx > 0:
                    logging.info(
                        f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx} | 实时准确率: {(correct_predictions / total_samples) * 100:.2f}% | Loss: {loss.item():.4f}"
                    )

            final_epoch_acc = (correct_predictions / total_samples) * 100
            logging.info(
                f"=== Epoch {epoch + 1} 完美训练结束 | 最终评估准确率: {final_epoch_acc:.2f}% ==="
            )

        torch.save(model.state_dict(), model_save_path)
        logging.info(f"核心资产落盘成功！DIN 神经中枢已保存在: '{model_save_path}'")

    except Exception as e:
        logging.error(f"DIN 训练引擎发生致命异常: {e}")


if __name__ == "__main__":
    train_din_model()
