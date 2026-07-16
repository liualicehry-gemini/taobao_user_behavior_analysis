import os
import logging
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, classification_report, log_loss

# 核心联系：导入网络架构和基础 Dataset
from advanced_din import ActualWideDIN, RealWideDINDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ==========================================
# 1. 评估专用 Dataset (引入严格的 OOS 物理隔离)
# ==========================================
class EvalWideDINDataset(RealWideDINDataset):
    def __init__(self, pkl_file, csv_file):
        # 1. 先让父类完成所有基础的加载和对齐工作
        super().__init__(pkl_file, csv_file)

        # 🛑 核心修复：用户级物理隔离 (User-Level Hold-out)
        # 使用哈希取模，稳定隔离出 20% 的用户作为纯净测试集，绝不包含训练集用户
        original_size = len(self.csv_df)
        self.csv_df = self.csv_df[self.csv_df["user_id"] % 5 == 0].reset_index(
            drop=True
        )

        # 同步清理 seq_dict，释放内存
        test_user_ids = set(self.csv_df["user_id"].values)
        self.seq_dict = {k: v for k, v in self.seq_dict.items() if k in test_user_ids}

        logging.info("=" * 50)
        logging.info("🛡️ [数据安全隔离机制] 启动成功")
        logging.info(f"原始全量样本: {original_size} 条")
        logging.info(f"隔离测试样本: {len(self.csv_df)} 条 (精准 20% OOS)")
        logging.info("=" * 50)

    def __getitem__(self, idx):
        # 复用特征提取，额外返回 user_id 计算 GAUC
        cat_seq, beh_seq, target_cat, dense_features, target_beh = super().__getitem__(
            idx
        )
        user_id = self.csv_df.iloc[idx]["user_id"]
        return user_id, cat_seq, beh_seq, target_cat, dense_features, target_beh


# ==========================================
# 2. 核心评估引擎
# ==========================================
def evaluate_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    seq_data_file = os.path.join(
        project_root, "data", "processed", "user_sequences.pkl"
    )
    csv_data_file = os.path.join(
        project_root, "data", "processed", "master_training_data.csv"
    )
    model_save_path = os.path.join(project_root, "models", "taobao_wide_din_actual.pth")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    logging.info("初始化评估引擎，加载测试数据...")
    dataset = EvalWideDINDataset(seq_data_file, csv_data_file)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

    logging.info("初始化 Wide-DIN 模型架构...")
    model = ActualWideDIN(num_dense_features=dataset.num_dense_features).to(device)

    if not os.path.exists(model_save_path):
        logging.error(f"找不到模型权重文件: {model_save_path}")
        return

    logging.info("成功读取模型权重，准备推断...")
    model.load_state_dict(
        torch.load(model_save_path, map_location=device, weights_only=True)
    )
    model.eval()  # 隐藏问题防御：强制关闭训练模式的随机性

    all_user_ids, all_true_labels, all_pred_probs, all_pred_classes = [], [], [], []

    logging.info("开始执行模型大盘推断...")
    with torch.no_grad():
        for batch_idx, (
            user_ids,
            cat_seq,
            beh_seq,
            target_cat,
            dense_features,
            target_beh,
        ) in enumerate(dataloader):
            cat_seq, beh_seq, target_cat, dense_features = (
                cat_seq.to(device),
                beh_seq.to(device),
                target_cat.to(device),
                dense_features.to(device),
            )

            outputs = model(cat_seq, beh_seq, target_cat, dense_features)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            trues = target_beh.cpu().numpy()

            all_user_ids.extend(user_ids.numpy())
            all_true_labels.extend(trues)
            all_pred_probs.extend(probs)
            all_pred_classes.extend(preds)

            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"推断进度: [{batch_idx}/{len(dataloader)}] Batch")

    all_true_labels = np.array(all_true_labels)
    all_pred_probs = np.array(all_pred_probs)
    all_pred_classes = np.array(all_pred_classes)

    print("\n" + "=" * 60)
    print("🎯 Wide-DIN 严格 OOS 离线评估报告")
    print("=" * 60)

    # 1. Classification Report
    target_names = ["View(浏览)", "Fav(收藏)", "Cart(加购)", "Buy(购买)"]
    print("\n1. 细粒度行为转化报告:")
    print(
        classification_report(
            all_true_labels, all_pred_classes, target_names=target_names
        )
    )

    # 2. GAUC
    is_buy_true = (all_true_labels == 3).astype(int)
    buy_probs = all_pred_probs[:, 3]
    df_eval = pd.DataFrame(
        {"user_id": all_user_ids, "true_buy": is_buy_true, "pred_prob": buy_probs}
    )

    user_aucs = []
    user_weights = []

    for uid, group in df_eval.groupby("user_id"):
        # 隐藏问题防御：过滤掉只有正类或负类的无效用户，防止除零报错
        if len(group["true_buy"].unique()) > 1:
            try:
                auc = roc_auc_score(group["true_buy"], group["pred_prob"])
                user_aucs.append(auc)
                user_weights.append(len(group))
            except ValueError:
                pass

    if len(user_aucs) > 0:
        gauc = np.average(user_aucs, weights=user_weights)
        print("\n2. 核心商业推荐指标 (GAUC):")
        print(f"购买转化 GAUC: {gauc:.5f}")
        print(f"(基于 {len(user_aucs)} 位有对比行为的纯净测试用户计算)")
    else:
        print("\n2. 核心商业推荐指标 (GAUC):")
        print("警告：测试集缺乏正负样本对比，无法计算。")

    # 3. LogLoss
    loss = log_loss(all_true_labels, all_pred_probs)
    print(f"\n3. 模型整体置信度偏差 (LogLoss): {loss:.5f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    evaluate_model()
