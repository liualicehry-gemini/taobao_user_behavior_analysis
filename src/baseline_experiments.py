import os
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# 智能路径定位
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = (
    os.path.dirname(current_dir)
    if os.path.basename(current_dir) == "src"
    else current_dir
)


def load_and_prep_data():
    csv_file = os.path.join(project_root, "data", "processed", "pit_master_data.csv")
    logging.info(f"读取无穿越数据集: {csv_file}")
    df = pd.read_csv(csv_file)

    # 构建基线模型使用的特征 (树模型和LR不需要复杂的序列Embedding，直接拉平)
    feature_cols = [
        "hist_pv_count",
        "hist_fav_count",
        "hist_cart_count",
        "hist_buy_count",
        "hist_beh_seq_1",
        "hist_beh_seq_2",
        "hist_beh_seq_3",
        "hist_beh_seq_4",
        "hist_beh_seq_5",
    ]

    X = df[feature_cols]
    # 标签从 1,2,3,4 映射到 0,1,2,3 (符合 sklearn 和 xgb 标准)
    y = df["target_label"] - 1

    # 划分训练集和测试集 (按时间切分逻辑，前80%训练，后20%测试)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 提取测试集的 user_id 用于计算 GAUC
    user_id_test = df["user_id"].iloc[split_idx:]

    return X_train, X_test, y_train, y_test, user_id_test


def evaluate_model(model_name, y_true, y_pred_prob, user_id_test):
    y_pred_class = np.argmax(y_pred_prob, axis=1)

    # 1. 打印 Buy(类别3) 的 Recall
    report = classification_report(
        y_true, y_pred_class, output_dict=True, zero_division=0
    )
    buy_recall = report["3"]["recall"]
    buy_precision = report["3"]["precision"]

    # 2. 计算 GAUC
    df_eval = pd.DataFrame(
        {
            "user_id": user_id_test,
            "true_buy": (y_true == 3).astype(int),
            "pred_prob": y_pred_prob[:, 3],
        }
    )
    user_aucs, user_weights = [], []
    for uid, group in df_eval.groupby("user_id"):
        if len(group["true_buy"].unique()) > 1:
            try:
                user_aucs.append(roc_auc_score(group["true_buy"], group["pred_prob"]))
                user_weights.append(len(group))
            except:
                pass

    gauc = np.average(user_aucs, weights=user_weights) if user_aucs else 0.5

    logging.info(
        f"[{model_name}] -> Buy Recall: {buy_recall:.4f} | Buy Precision: {buy_precision:.4f} | GAUC: {gauc:.5f}"
    )
    return gauc, buy_recall


def run_experiments():
    X_train, X_test, y_train, y_test, user_id_test = load_and_prep_data()

    print("\n" + "=" * 50)
    print("🧪 实验 1: 不同类别权重下的 LightGBM 表现 (解决数据不平衡)")
    print("=" * 50)

    weight_configs = {
        "无权重 (Baseline)": None,
        "温和权重 (1:1.5:2:3)": {0: 1.0, 1: 1.5, 2: 2.0, 3: 3.0},
        "激进权重 (1:10:15:20)": {0: 1.0, 1: 10.0, 2: 15.0, 3: 20.0},
        "自动平衡 (class_weight='balanced')": "balanced",
    }

    for name, cw in weight_configs.items():
        model = lgb.LGBMClassifier(class_weight=cw, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        evaluate_model(f"LGBM - {name}", y_test, probs, user_id_test)

    print("\n" + "=" * 50)
    print("🧪 实验 2: 数据采样策略对比 (SMOTE vs 欠采样)")
    print("=" * 50)

    # 策略 A: 欠采样 (减少 View 样本，使其与 Buy 样本数量接近)
    logging.info("执行随机欠采样 (Undersampling)...")
    rus = RandomUnderSampler(random_state=42)
    X_resampled_rus, y_resampled_rus = rus.fit_resample(X_train, y_train)
    model_rus = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
    model_rus.fit(X_resampled_rus, y_resampled_rus)
    probs_rus = model_rus.predict_proba(X_test)
    evaluate_model("LGBM - 随机欠采样", y_test, probs_rus, user_id_test)

    # 策略 B: SMOTE 过采样 (由于数据量太大，SMOTE 可能很慢，这里仅做演示，真实环境需控制比例)
    logging.info("执行 SMOTE 过采样 (可能需要较长时间)...")
    try:
        smote = SMOTE(random_state=42, n_jobs=-1)
        X_resampled_smote, y_resampled_smote = smote.fit_resample(X_train, y_train)
        model_smote = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
        model_smote.fit(X_resampled_smote, y_resampled_smote)
        probs_smote = model_smote.predict_proba(X_test)
        evaluate_model("LGBM - SMOTE过采样", y_test, probs_smote, user_id_test)
    except Exception as e:
        logging.error(f"SMOTE 执行失败或内存溢出: {e}")

    print("\n" + "=" * 50)
    print("🧪 实验 3: 不同基线模型对比 (Baseline Models)")
    print("=" * 50)

    # 1. 逻辑回归 (LR)
    model_lr = LogisticRegression(class_weight="balanced", max_iter=1000, n_jobs=-1)
    model_lr.fit(X_train, y_train)
    probs_lr = model_lr.predict_proba(X_test)
    evaluate_model("Logistic Regression (LR)", y_test, probs_lr, user_id_test)

    # 2. XGBoost
    model_xgb = xgb.XGBClassifier(
        objective="multi:softprob", num_class=4, n_jobs=-1, random_state=42
    )
    model_xgb.fit(X_train, y_train)
    probs_xgb = model_xgb.predict_proba(X_test)
    evaluate_model("XGBoost", y_test, probs_xgb, user_id_test)


if __name__ == "__main__":
    run_experiments()
