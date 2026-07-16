import pandas as pd
import lightgbm as lgb
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_model():
    try:
        logging.info("Starting Stage 4: Model Training (Leakage Fixed)...")

        logging.info("Loading master training data...")
        df = pd.read_csv("../data/processed/master_training_data.csv")

        # --- THE LEAKAGE FIX ---
        # We must drop the specific interaction counts that give away the answer.
        # However, notice we KEEP 'buy_count_user' and 'buy_count_item' because general
        # user/item history is perfectly legal and highly predictive!
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
        y = df["target_label"] - 1  # Map labels (1,2,3,4) to (0,1,2,3) for LightGBM

        logging.info("Splitting data into Training (80%) and Testing (20%) sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logging.info("Initializing LightGBM Classifier...")
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=4,
            n_estimators=100,
            learning_rate=0.05,  # Lower learning rate for more careful learning
            random_state=42,
            n_jobs=-1,
            verbose=-1,  # Turns off those annoying "No further splits" warnings
        )

        logging.info("Training the real predictive model... (No cheating allowed!)")
        model.fit(X_train, y_train)

        logging.info("Evaluating model performance...")
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        logging.info(f"Realistic Model Accuracy: {accuracy * 100:.2f}%")

        model_filename = "../data/processed/taobao_behavior_model.pkl"
        joblib.dump(model, model_filename)
        logging.info(f"Success! Honest trained model saved as '{model_filename}'")

    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")
        raise


if __name__ == "__main__":
    train_model()
