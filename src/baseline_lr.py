import pandas as pd
import lightgbm as lgb
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_lr_baseline():
    try:
        logging.info(
            "Starting Stage 4: Baseline Model Training (Logistic Regression)..."
        )

        # 1. 加载主训练数据
        logging.info("Loading master training data...")
        df = pd.read_csv("../data/processed/master_training_data.csv")

        # 2. 严格执行特征防泄露 (The Leakage Fix) - 保持与 LightGBM 完全一致
        leakage_columns = [
            "pv_count",
            "fav_count",
            "cart_count",
            "buy_count",
            "total_actions",
            "buy_rate",
        ]
        columns_to_drop = ["user_id", "item_id", "target_label"] + leakage_columns

        X = df.drop(columns=columns_to_drop)
        y = df["target_label"] - 1  # 映射标签 (1,2,3,4) 到 (0,1,2,3) 以符合多分类要求

        # 3. 划分训练集与测试集
        logging.info("Splitting data into Training (80%) and Testing (20%) sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 4. 特征标准化 (Standardization) - 线性模型的关键步骤
        logging.info("Scaling features (Crucial for Logistic Regression)...")
        scaler = StandardScaler()

        # 严防测试集信息泄露：仅在训练集上 fit_transform，测试集上仅进行 transform
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 5. 初始化多元逻辑回归模型
        logging.info("Initializing Multinomial Logistic Regression...")
        model = LogisticRegression(
            multi_class="multinomial",  # 明确指定为多分类任务
            max_iter=1000,  # 增加迭代次数以确保模型收敛
            random_state=42,
            n_jobs=-1,  # 开启多核并行计算加速训练
        )

        # 6. 训练模型
        logging.info("Training the LR baseline model... (No cheating allowed!)")
        model.fit(X_train_scaled, y_train)

        # 7. 评估模型性能
        logging.info("Evaluating model performance...")
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)

        logging.info(f"=====================================")
        logging.info(f"Realistic LR Model Accuracy: {accuracy * 100:.2f}%")
        logging.info(f"=====================================")

        # 打印详细分类指标供后续与 LightGBM 对比
        print("\nClassification Report:\n", classification_report(y_test, predictions))

        # 8. 独立保存模型和对应的 Scaler (与 LightGBM 的 taobao_behavior_model.pkl 区分开)
        model_filename = "../data/processed/taobao_behavior_lr_model.pkl"
        scaler_filename = "../data/processed/taobao_behavior_lr_scaler.pkl"

        joblib.dump(model, model_filename)
        joblib.dump(scaler, scaler_filename)

        logging.info(f"Success! Honest LR model saved as '{model_filename}'")
        logging.info(f"Success! Associated scaler saved as '{scaler_filename}'")

    except Exception as e:
        logging.error(f"An error occurred during LR model training: {e}")
        raise


if __name__ == "__main__":
    train_lr_baseline()
