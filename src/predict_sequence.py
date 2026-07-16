import pandas as pd
import torch
import torch.nn as nn
import logging
import random
import os

logging.basicConfig(level=logging.INFO, format="%(message)s")


# 必须重定义相同的模型架构，PyTorch 才能把权重准确塞进去
class TaobaoBehaviorGRU(nn.Module):
    def __init__(self, num_categories=10000, num_behaviors=5, hidden_size=64):
        super(TaobaoBehaviorGRU, self).__init__()
        self.cat_embedding = nn.Embedding(
            num_embeddings=num_categories, embedding_dim=32
        )
        self.beh_embedding = nn.Embedding(
            num_embeddings=num_behaviors, embedding_dim=16
        )
        self.gru = nn.GRU(input_size=32 + 16, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 4)

    def forward(self, cat_seq, beh_seq):
        cat_embs = self.cat_embedding(cat_seq)
        beh_embs = self.beh_embedding(beh_seq)
        combined_input = torch.cat([cat_embs, beh_embs], dim=-1)
        gru_out, hidden_state = self.gru(combined_input)
        return self.fc(hidden_state[-1])


def predict_next_action():
    try:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # --- 动态路径导航核心逻辑 (Senior 做法) ---
        # 1. 拿到当前脚本的绝对路径，退回项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)

        # 2. 从项目根目录出发，动态拼接正确的绝对路径
        model_path = os.path.join(project_root, "models", "taobao_sequence_model.pth")
        data_path = os.path.join(
            project_root, "data", "processed", "user_sequences.pkl"
        )
        # ----------------------------------------

        if not os.path.exists(model_path):
            logging.error(f"Cannot find the trained model at: {model_path}")
            logging.error("Please make sure your .pth file is in the models/ folder!")
            return

        # 加载模型大脑
        model = TaobaoBehaviorGRU().to(device)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.eval()  # Put the model in "Test Mode" (turns off learning)

        # 加载用户序列数据
        if not os.path.exists(data_path):
            logging.error(f"Cannot find data at: {data_path}")
            return

        df = pd.read_pickle(data_path)

        # 过滤掉只有 padding 的无效空序列
        df = df[df["behavior_seq"].apply(lambda x: sum(x) > 0)].reset_index(drop=True)

        # 随机抽取一个幸运用户
        random_idx = random.randint(0, len(df) - 1)
        user_row = df.iloc[random_idx]
        user_id = user_row["user_id"]

        # 提取该用户的前 19 步历史轨迹
        cat_history = torch.tensor(
            [user_row["category_seq"][:-1]], dtype=torch.long
        ).to(device)
        beh_history = torch.tensor(
            [user_row["behavior_seq"][:-1]], dtype=torch.long
        ).to(device)

        # 获取真实的第 20 步动作 (作为对比)
        actual_next_step = user_row["behavior_seq"][-1]

        # 模型预测第 20 步！
        with torch.no_grad():  # 预测时不需要计算梯度，省内存
            output = model(cat_history, beh_history)
            probabilities = torch.softmax(output[0], dim=0) * 100
            predicted_class = (
                torch.argmax(output[0]).item() + 1
            )  # Convert back to 1,2,3,4

        # 格式化终端输出打印
        behavior_map = {
            0: "Empty",
            1: " View",
            2: " Favorite",
            3: " Cart",
            4: " Buy",
        }

        print(f"\n=== Deep Learning Sequence Predictor ===")
        print(f"Analyzing User ID: {user_id}")
        print("-" * 40)
        print("User's Recent Timeline (Last 5 steps):")

        # 为了不让终端太长，只打印最后 5 步的历史
        for i in range(14, 19):
            beh = user_row["behavior_seq"][i]
            if beh > 0:
                print(f"Step {i + 1}: {behavior_map[beh]}")

        print("-" * 40)
        print("AI Prediction for Next Action (Step 20):")
        print(f"Predicted: {behavior_map[predicted_class]}")
        print(f"Actual:    {behavior_map[actual_next_step]}")
        print("\nConfidence Breakdown:")
        print(
            f" View: {probabilities[0]:.2f}% |  Fav: {probabilities[1]:.2f}% |  Cart: {probabilities[2]:.2f}% |  Buy: {probabilities[3]:.2f}%"
        )
        print("=======================================\n")

    except Exception as e:
        logging.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    predict_next_action()
