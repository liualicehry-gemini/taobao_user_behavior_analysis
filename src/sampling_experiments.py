import os
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
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
    y = df["target_label"] - 1

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    user_id_test = df["user_id"].iloc[split_idx:]

    return X_train, X_test, y_train, y_test, user_id_test


def evaluate_model(model_name, y_true, y_pred_prob, user_id_test):
    y_pred_class = np.argmax(y_pred_prob, axis=1)

    report = classification_report(
        y_true, y_pred_class, output_dict=True, zero_division=0
    )
    buy_recall = report["3"]["recall"]
    buy_precision = report["3"]["precision"]

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


def run_sampling_experiments():
    X_train, X_test, y_train, y_test, user_id_test = load_and_prep_data()

    # 核心修复：获取各类别在训练集中的真实数量，用于动态计算多分类字典
    counts = y_train.value_counts().to_dict()
    n_view = counts.get(0, 1)  # 类别 0 (浏览) 的数量
    n_buy = counts.get(3, 1)  # 类别 3 (购买) 的数量

    print("\n" + "=" * 60)
    print("专项测试 A: 欠采样 (Under-sampling) 的参数深度探索")
    print("=" * 60)

    # multiplier: 代表我们希望保留的多数类(View)数量，是少数类(Buy)的几倍
    rus_configs = {
        "强力平衡 (1:1)": 1.0,
        "适中平衡 (1:5)": 5.0,
        "轻度平衡 (1:10)": 10.0,
    }

    for name, mult in rus_configs.items():
        logging.info(f"执行随机欠采样 - {name}...")

        # 多分类欠采样必须使用 dict 指定每个类别的目标数量
        target_count = int(n_buy * mult)

        # 仅对真实样本量 大于 target_count 的类别进行欠采样（砍掉多余数据）
        rus_dict = {
            c: target_count for c, count in counts.items() if count > target_count
        }

        rus = RandomUnderSampler(sampling_strategy=rus_dict, random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

        model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
        model.fit(X_resampled, y_resampled)
        probs = model.predict_proba(X_test)
        evaluate_model(f"欠采样 - {name}", y_test, probs, user_id_test)

    print("\n" + "=" * 60)
    print("专项测试 B: SMOTE 过采样 的参数深度探索")
    print("=" * 60)

    # ratio: 代表生成出的少数类(Buy)数量，达到多数类(View)的比例
    smote_configs = {
        "比例 1:10 | 近邻 k=5": (0.1, 5),
        "比例 1:10 | 近邻 k=10": (0.1, 10),
        "比例 1:5 | 近邻 k=5": (0.2, 5),
    }

    for name, (ratio, k) in smote_configs.items():
        logging.info(f"执行 SMOTE - {name} (极其耗时/吃内存，请耐心等待)...")
        try:
            # 多分类过采样必须使用 dict 指定每个类别的目标数量
            target_count = int(n_view * ratio)

            # 仅对真实样本量 小于 target_count 的类别进行过采样（无中生有补充数据）
            smote_dict = {
                c: target_count for c, count in counts.items() if count < target_count
            }

            smote = SMOTE(sampling_strategy=smote_dict, k_neighbors=k, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

            model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
            model.fit(X_resampled, y_resampled)
            probs = model.predict_proba(X_test)
            evaluate_model(f"SMOTE - {name}", y_test, probs, user_id_test)
        except Exception as e:
            logging.error(f"SMOTE [{name}] 执行失败或内存溢出: {e}")


if __name__ == "__main__":
    run_sampling_experiments()
