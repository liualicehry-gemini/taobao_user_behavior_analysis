import os
import logging
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ==========================================
# 1. 严格复现的 DIN 架构 (必须一致才能加载权重)
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


class TaobaoDIN(nn.Module):
    def __init__(
        self, num_categories=10000, num_behaviors=5, embed_cat_dim=32, embed_beh_dim=16
    ):
        super(TaobaoDIN, self).__init__()
        self.cat_embedding = nn.Embedding(num_categories, embed_cat_dim)
        self.beh_embedding = nn.Embedding(num_behaviors, embed_beh_dim)
        total_embed_dim = embed_cat_dim + embed_beh_dim
        self.attention = AttentionUnit(total_embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(total_embed_dim * 2, 64),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4),
        )

    # 🛑 核心升级：增加 return_attention 参数，把底层的注意力权重抛出来！
    def forward(self, cat_seq, beh_seq, target_cat, return_attention=False):
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
        final_input = torch.cat([user_interest, target_emb], dim=-1)
        output = self.fc(final_input)

        if return_attention:
            return output, attn_weights
        return output


# ==========================================
# 2. 预测与可视化引擎
# ==========================================
def predict_with_attention():
    try:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # 动态绝对路径护航
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, "models", "taobao_din_model.pth")
        data_path = os.path.join(
            project_root, "data", "processed", "user_sequences.pkl"
        )

        if not os.path.exists(model_path):
            logging.error(f"找不到模型权重文件: {model_path}")
            return

        # 加载模型大脑
        model = TaobaoDIN().to(device)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.eval()

        # 加载数据
        df = pd.read_pickle(data_path)

        # 🛑 魔法过滤：只挑选真实发生购买 (behavior == 4) 且历史动作丰富的用户
        df = df[df["behavior_seq"].apply(lambda x: sum(x) > 0)]
        df = df[df["behavior_seq"].apply(lambda x: x[-1] == 4)].reset_index(drop=True)

        if len(df) == 0:
            logging.error("没有找到符合条件的购买用户数据！")
            return

        # 随机抽取一个幸运买家
        random_idx = random.randint(0, len(df) - 1)
        user_row = df.iloc[random_idx]
        user_id = user_row["user_id"]

        # 提取历史轨迹 (前 19 步) 和 Target 项 (第 20 步)
        cat_seq = torch.tensor([user_row["category_seq"][:-1]], dtype=torch.long).to(
            device
        )
        beh_seq = torch.tensor([user_row["behavior_seq"][:-1]], dtype=torch.long).to(
            device
        )
        target_cat = torch.tensor([user_row["category_seq"][-1]], dtype=torch.long).to(
            device
        )

        actual_next_step = user_row["behavior_seq"][-1]

        # 窥探大脑：不仅要结果，还要拿到底层的注意力分数
        with torch.no_grad():
            output, attn_weights = model(
                cat_seq, beh_seq, target_cat, return_attention=True
            )
            probabilities = torch.softmax(output[0], dim=0) * 100
            predicted_class = torch.argmax(output[0]).item() + 1

        # 将张量转为方便打印的 numpy 数组
        attn_weights = attn_weights[0].cpu().numpy()

        behavior_map = {
            0: "Empty",
            1: "浏览(View)",
            2: "收藏(Fav) ",
            3: "加购(Cart)",
            4: "购买(Buy) ",
        }

        # =======================================
        # 🌟 终端炫酷打印逻辑
        # =======================================
        print(f"\n" + "=" * 55)
        print(f" 🎯 DIN 深度兴趣网络 - 意图透视分析系统")
        print(f" 正在分析用户 ID: {user_id}")
        print(f" 目标预测商品类别: Category {target_cat.item()}")
        print("=" * 55)

        print("\n⏳ 用户的历史行为与 DIN 局部注意力分配 (Local Activation):")
        print("-" * 55)

        # 只打印最后 10 步历史，避免刷屏，且重点展示近期意图
        for i in range(9, 19):
            beh_code = user_row["behavior_seq"][i]
            cat_code = user_row["category_seq"][i]
            weight = attn_weights[i] * 100

            if beh_code > 0:
                # 绘制 ASCII 能量条
                bar_len = int(weight / 2)  # 每 2% 画一个块
                bar = "█" * bar_len + "░" * (20 - bar_len)

                action = behavior_map[beh_code]
                print(
                    f"Step {i + 1:02d}: [{action}] 品类:{cat_code:<5d} | {bar} {weight:>5.2f}%"
                )

        print("-" * 55)
        print(f"🧠 模型推断结果 (Step 20):")
        print(f"实际发生动作: {behavior_map[actual_next_step]}")
        print(f"DIN 预测动作: {behavior_map[predicted_class]}")
        print("\n📊 转化漏斗置信度分布:")
        print(f" {behavior_map[1][:2]}: {probabilities[0]:6.2f}%")
        print(f" {behavior_map[2][:2]}: {probabilities[1]:6.2f}%")
        print(f" {behavior_map[3][:2]}: {probabilities[2]:6.2f}%")
        print(f" {behavior_map[4][:2]}: {probabilities[3]:6.2f}%")
        print("=" * 55 + "\n")

    except Exception as e:
        logging.error(f"预测解析失败: {e}")


if __name__ == "__main__":
    predict_with_attention()
