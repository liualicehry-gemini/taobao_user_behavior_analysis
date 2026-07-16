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
# 1. 无泄露版 Dataset (防穿越机制)
# ==========================================
class RealWideDINDataset(Dataset):
    def __init__(self, pkl_file, csv_file):
        logging.info("Step 1: 加载序列特征 (Deep 侧) ...")
        seq_df = pd.read_pickle(pkl_file)
        seq_df = seq_df[seq_df["behavior_seq"].apply(lambda x: sum(x) > 0)]
        self.seq_dict = seq_df.set_index("user_id").to_dict("index")
        logging.info(f"成功构建序列字典，用户数: {len(self.seq_dict)}")

        logging.info("Step 2: 加载统计宽表特征 (Wide 侧) ...")
        self.csv_df = pd.read_csv(csv_file)
        self.csv_df = self.csv_df[
            self.csv_df["user_id"].isin(self.seq_dict.keys())
        ].reset_index(drop=True)

        # 🛑 【核心修复：防数据泄露 (Data Leakage Fix)】
        # 坚决剔除直接关联目标商品的交叉统计特征，防止模型“偷看未来答案”
        leakage_columns = [
            "pv_count",
            "fav_count",
            "cart_count",
            "buy_count",
            "total_actions",
            "buy_rate",
        ]

        # 动态识别安全的画像特征 (保留全局 User 画像和全局 Item 画像)
        self.dense_cols = [
            col
            for col in self.csv_df.columns
            if col not in ["user_id", "item_id", "target_label"] + leakage_columns
        ]

        self.num_dense_features = len(self.dense_cols)
        logging.info(f"🛡️ 防泄露机制启动！已拦截高危特征: {leakage_columns}")
        logging.info(
            f"系统保留了 {self.num_dense_features} 个安全的全局画像特征: {self.dense_cols[:5]}..."
        )

    def __len__(self):
        return len(self.csv_df)

    def __getitem__(self, idx):
        csv_row = self.csv_df.iloc[idx]
        user_id = csv_row["user_id"]
        seq_row = self.seq_dict[user_id]

        # --- Deep 侧特征提取 ---
        cat_seq = torch.tensor(seq_row["category_seq"][:-1], dtype=torch.long)
        beh_seq = torch.tensor(seq_row["behavior_seq"][:-1], dtype=torch.long)
        target_cat = torch.tensor(seq_row["category_seq"][-1], dtype=torch.long)

        # --- Wide 侧特征提取 (安全的特征) ---
        raw_dense_values = csv_row[self.dense_cols].values.astype(np.float32)
        # log1p 归一化应对长尾分布
        dense_features = torch.log1p(torch.tensor(raw_dense_values))

        # --- 目标标签 ---
        target_beh = torch.tensor(int(csv_row["target_label"]) - 1, dtype=torch.long)
        target_beh = torch.clamp(target_beh, min=0)

        return cat_seq, beh_seq, target_cat, dense_features, target_beh


# ==========================================
# 2. 真实架构: Wide & Deep + Attention
# ==========================================
class AttentionUnit(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionUnit, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 4, 64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, query, facts):
        B, T, E = facts.size()
        query = query.unsqueeze(1).expand(-1, T, -1)
        combined = torch.cat([query, facts, query - facts, query * facts], dim=-1)
        scores = self.fc(combined)
        return scores.squeeze(-1)


class ActualWideDIN(nn.Module):
    def __init__(
        self,
        num_dense_features,
        num_categories=10000,
        num_behaviors=5,
        embed_cat_dim=32,
        embed_beh_dim=16,
    ):
        super(ActualWideDIN, self).__init__()

        # --- Deep 序列塔 ---
        self.cat_embedding = nn.Embedding(num_categories, embed_cat_dim)
        self.beh_embedding = nn.Embedding(num_behaviors, embed_beh_dim)
        total_embed_dim = embed_cat_dim + embed_beh_dim
        self.attention = AttentionUnit(total_embed_dim)

        # --- Wide 画像塔 ---
        self.dense_layer = nn.Sequential(
            nn.Linear(num_dense_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # --- 顶层融合塔 ---
        fusion_dim = total_embed_dim * 2 + 32
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, cat_seq, beh_seq, target_cat, dense_features):
        cat_embs = self.cat_embedding(cat_seq)
        beh_embs = self.beh_embedding(beh_seq)
        hist_embs = torch.cat([cat_embs, beh_embs], dim=-1)

        target_cat_emb = self.cat_embedding(target_cat)
        target_beh_placeholder = torch.zeros(
            (target_cat.size(0), 16), device=target_cat.device
        )
        target_emb = torch.cat([target_cat_emb, target_beh_placeholder], dim=-1)

        attn_weights = self.attention(target_emb, hist_embs)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        user_interest = torch.bmm(attn_weights.unsqueeze(1), hist_embs).squeeze(1)

        processed_dense = self.dense_layer(dense_features)

        final_input = torch.cat([user_interest, target_emb, processed_dense], dim=-1)
        return self.fc(final_input)


# ==========================================
# 3. 自动化训练引擎
# ==========================================
def train_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)

        seq_data_file = os.path.join(
            project_root, "data", "processed", "user_sequences.pkl"
        )
        csv_data_file = os.path.join(
            project_root, "data", "processed", "master_training_data.csv"
        )
        model_save_path = os.path.join(
            project_root, "models", "taobao_wide_din_actual.pth"
        )

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        dataset = RealWideDINDataset(seq_data_file, csv_data_file)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

        model = ActualWideDIN(num_dense_features=dataset.num_dense_features).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        epochs = 4

        logging.info("点火！双塔全特征融合模型开始训练...")
        for epoch in range(epochs):
            model.train()
            total_loss, correct_predictions, total_samples = 0, 0, 0

            for batch_idx, (
                cat_seq,
                beh_seq,
                target_cat,
                dense_features,
                target_beh,
            ) in enumerate(dataloader):
                cat_seq, beh_seq, target_cat, dense_features, target_beh = (
                    cat_seq.to(device),
                    beh_seq.to(device),
                    target_cat.to(device),
                    dense_features.to(device),
                    target_beh.to(device),
                )

                optimizer.zero_grad()
                predictions = model(cat_seq, beh_seq, target_cat, dense_features)
                loss = criterion(predictions, target_beh)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted_classes = torch.max(predictions, 1)
                correct_predictions += (predicted_classes == target_beh).sum().item()
                total_samples += target_beh.size(0)

                if batch_idx % 50 == 0 and batch_idx > 0:
                    # 🌟 替换成这行（加入了 {len(dataloader)}）：
                    logging.info(
                        f"Epoch {epoch + 1}/{epochs} | Batch [{batch_idx}/{len(dataloader)}] | 准确率: {(correct_predictions / total_samples) * 100:.2f}% | Loss: {loss.item():.4f}"
                    )
                    



            final_epoch_acc = (correct_predictions / total_samples) * 100
            logging.info(
                f"=== Epoch {epoch + 1} 结束 | 整体评估准确率: {final_epoch_acc:.2f}% ==="
            )

        torch.save(model.state_dict(), model_save_path)
        logging.info(f"模型落盘！全视野无泄露 DIN 保存在: '{model_save_path}'")

    except Exception as e:
        logging.error(f"引擎发生异常: {e}")


if __name__ == "__main__":
    train_model()
