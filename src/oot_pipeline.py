import os
import sys
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, classification_report, log_loss

# 💡 智能路径识别机制
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_dir) == "src":
    project_root = os.path.dirname(current_dir)
else:
    project_root = current_dir

sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

# 导入网络架构
from advanced_din import ActualWideDIN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# 🛑 核心重构：专为 Point-in-Time 无泄漏数据打造的 Dataset
class PiTOOTDataset(Dataset):
    def __init__(self, csv_file, mode="train", split_ratio=0.8):
        logging.info(f"加载无穿越 Point-in-Time 数据集 ({mode.upper()})...")
        self.df = pd.read_csv(csv_file)

        # 按数据行物理顺序（已在构建时按时间排序）严格切断过去与未来
        split_idx = int(len(self.df) * split_ratio)

        if mode == "train":
            self.df = self.df.iloc[:split_idx].reset_index(drop=True)
            logging.info(f"📈 [Train] 过去时间段数据量: {len(self.df)}")
        else:
            self.df = self.df.iloc[split_idx:].reset_index(drop=True)
            logging.info(f"🔮 [Test] 未来时间段数据量: {len(self.df)}")

        # 我们有 4 个连续统计特征: pv, fav, cart, buy
        self.num_dense_features = 4

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_id = row["user_id"]
        target_cat = torch.tensor(int(row["item_id"] % 10000), dtype=torch.long)
        target_beh = torch.tensor(int(row["target_label"] - 1), dtype=torch.long)

        # 从特征表提取序列：注意我们要把最远的 shift_5 放在开头，最近的 shift_1 放在最后，符合时序习惯
        cat_seq = torch.tensor(
            [int(row[f"hist_cat_seq_{i}"]) % 10000 for i in range(5, 0, -1)],
            dtype=torch.long,
        )
        beh_seq = torch.tensor(
            [int(row[f"hist_beh_seq_{i}"]) for i in range(5, 0, -1)], dtype=torch.long
        )

        # 提取 4 维全局宏观无穿越特征
        dense_features = torch.tensor(
            [
                row["hist_pv_count"],
                row["hist_fav_count"],
                row["hist_cart_count"],
                row["hist_buy_count"],
            ],
            dtype=torch.float32,
        )

        # 简单平滑归一化，防止极个别超高频用户的数值导致梯度爆炸
        dense_features = torch.log1p(dense_features)

        return user_id, cat_seq, beh_seq, target_cat, dense_features, target_beh


def train_oot_model():
    # 核心：直接读取新生成的无泄漏大表
    csv_file = os.path.join(project_root, "data", "processed", "pit_master_data.csv")
    model_path = os.path.join(project_root, "models", "taobao_wide_din_pit.pth")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logging.info("=" * 50)
    logging.info("🛑 启动真·OOT 闭卷训练 (优化排序能力，追求真实高 GAUC)!")
    logging.info("=" * 50)

    train_dataset = PiTOOTDataset(csv_file, mode="train", split_ratio=0.8)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    model = ActualWideDIN(num_dense_features=train_dataset.num_dense_features).to(
        device
    )

    # 【核心调整1】：温和的类别权重。既要让模型关注稀有类，又不能破坏概率的相对排序 (GAUC)
    class_weights = torch.tensor([1.0, 1.5, 2.0, 3.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 【核心调整2】：降低学习率，让模型在微弱的真实信号中稳健收敛
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 【核心调整3】：大幅增加 Epoch，给予网络充分的时间去挖掘真实的微弱信号
    epochs = 10

    for epoch in range(epochs):
        model.train()
        for batch_idx, (
            user_ids,
            cat_seq,
            beh_seq,
            target_cat,
            dense_features,
            target_beh,
        ) in enumerate(train_loader):
            cat_seq, beh_seq = cat_seq.to(device), beh_seq.to(device)
            target_cat, dense_features = (
                target_cat.to(device),
                dense_features.to(device),
            )
            target_beh = target_beh.to(device)

            optimizer.zero_grad()
            outputs = model(cat_seq, beh_seq, target_cat, dense_features)
            loss = criterion(outputs, target_beh)
            loss.backward()
            optimizer.step()

            if batch_idx % 300 == 0 and batch_idx > 0:
                logging.info(
                    f"⚙️ 闭卷训练中: Epoch [{epoch + 1}/{epochs}] - Batch [{batch_idx}/{len(train_loader)}] - Loss: {loss.item():.4f}"
                )

    torch.save(model.state_dict(), model_path)
    logging.info(f"✅ PiT 无泄漏高优排序模型已保存至: {model_path}")


def run_oot_evaluation():
    csv_file = os.path.join(project_root, "data", "processed", "pit_master_data.csv")
    model_path = os.path.join(project_root, "models", "taobao_wide_din_pit.pth")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    test_dataset = PiTOOTDataset(csv_file, mode="test", split_ratio=0.8)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    model = ActualWideDIN(num_dense_features=test_dataset.num_dense_features).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    all_user_ids, all_trues, all_pred_probs, all_preds = [], [], [], []

    logging.info(f"🚀 启动无泄漏推断... (需处理 {len(test_loader)} 个 Batch)")
    with torch.no_grad():
        for batch_idx, (
            user_ids,
            cat_seq,
            beh_seq,
            target_cat,
            dense_features,
            target_beh,
        ) in enumerate(test_loader):
            cat_seq, beh_seq = cat_seq.to(device), beh_seq.to(device)
            target_cat, dense_features = (
                target_cat.to(device),
                dense_features.to(device),
            )

            outputs = model(cat_seq, beh_seq, target_cat, dense_features)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_user_ids.extend(user_ids.numpy())
            all_trues.extend(target_beh.numpy())
            all_pred_probs.extend(probs)
            all_preds.extend(preds)

            if batch_idx % 200 == 0 and batch_idx > 0:
                logging.info(f"⏳ 推断进度: [{batch_idx}/{len(test_loader)}] Batch")

    all_trues = np.array(all_trues)
    all_pred_probs = np.array(all_pred_probs)
    all_preds = np.array(all_preds)

    print("\n" + "=" * 60)
    print("🛡️ Wide-DIN 终极防线：无穿越 OOT 离线评估报告")
    print("=" * 60)

    target_names = ["View(浏览)", "Fav(收藏)", "Cart(加购)", "Buy(购买)"]
    print("\n1. 细粒度行为转化报告 (PiT OOT):")
    print(classification_report(all_trues, all_preds, target_names=target_names))

    is_buy_true = (all_trues == 3).astype(int)
    buy_probs = all_pred_probs[:, 3]
    df_eval = pd.DataFrame(
        {"user_id": all_user_ids, "true_buy": is_buy_true, "pred_prob": buy_probs}
    )

    user_aucs, user_weights = [], []
    for uid, group in df_eval.groupby("user_id"):
        if len(group["true_buy"].unique()) > 1:
            try:
                user_aucs.append(roc_auc_score(group["true_buy"], group["pred_prob"]))
                user_weights.append(len(group))
            except ValueError:
                pass

    if len(user_aucs) > 0:
        gauc = np.average(user_aucs, weights=user_weights)
        print(
            f"\n2. 核心商业指标 (无泄漏真·GAUC): {gauc:.5f} (基于 {len(user_aucs)} 位验证用户)"
        )
    else:
        print("\n2. 核心商业指标 (无泄漏真·GAUC): 数据不足以计算对比")

    loss = log_loss(all_trues, all_pred_probs)
    print(f"\n3. 置信度偏差 (LogLoss): {loss:.5f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train_oot_model()
    run_oot_evaluation()
