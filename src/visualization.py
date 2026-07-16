import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# 配置中文字体，兼容 Mac (Arial Unicode MS) 和 Windows (SimHei/Microsoft YaHei)
plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "SimHei",
    "Microsoft YaHei",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

# 智能路径定位
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = (
    os.path.dirname(current_dir)
    if os.path.basename(current_dir) == "src"
    else current_dir
)
figures_dir = os.path.join(project_root, "figures")

# 确保 figures 文件夹存在
os.makedirs(figures_dir, exist_ok=True)


def plot_model_gauc_comparison():
    """
    绘制多模型 GAUC 对比柱状图
    """
    logging.info("正在生成模型 GAUC 对比图...")

    # 基于我们在 baseline_experiments.py 和 oot_pipeline.py 中得到的真实硬核数据
    models = ["LightGBM", "XGBoost", "Logistic Regression", "Wide-DIN (Ours)"]
    gaucs = [0.4013, 0.4071, 0.4297, 0.4791]

    # 颜色区分：基线模型用灰色调，我们的 Wide-DIN 用亮色调突出
    colors = ["#cbd5e1", "#94a3b8", "#64748b", "#ef4444"]

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # 重置中文字体 (seaborn set_theme 会覆盖之前的字体设置)
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei"]

    ax = sns.barplot(x=models, y=gaucs, palette=colors)

    # 在柱子上添加具体的数值标签
    for i, v in enumerate(gaucs):
        ax.text(
            i,
            v + 0.002,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#1e293b",
        )

    plt.title(
        "核心排序能力 GAUC 对比 (Point-in-Time 严格防穿越测试集)", fontsize=16, pad=20
    )
    plt.ylabel("GAUC (购买转化排序能力)", fontsize=14)
    plt.xlabel("模型架构", fontsize=14)
    plt.ylim(0.38, 0.50)  # 截断 Y 轴以放大差距

    # 调整布局并保存
    plt.tight_layout()
    save_path = os.path.join(figures_dir, "model_gauc_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"✅ 对比图已保存至: {save_path}")


def plot_feature_importance():
    """
    绘制 LightGBM 的特征重要性条形图
    """
    logging.info("正在训练模型以提取特征重要性...")
    csv_file = os.path.join(project_root, "data", "processed", "pit_master_data.csv")

    if not os.path.exists(csv_file):
        logging.error(f"未找到数据集 {csv_file}，跳过特征重要性作图。")
        return

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

    # 快速训练一个模型获取特征权重 (不追求调参，只看全局特征重要度)
    model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, n_estimators=50)
    model.fit(X, y)

    # 提取特征重要性 (基于 split 次数或增益 gain)
    importances = model.feature_importances_

    # 映射为更好懂的业务名称
    feature_names_mapping = {
        "hist_pv_count": "历史总浏览量 (Wide)",
        "hist_fav_count": "历史总收藏量 (Wide)",
        "hist_cart_count": "历史总加购量 (Wide)",
        "hist_buy_count": "历史总购买量 (Wide)",
        "hist_beh_seq_1": "倒数第1次行为 (Deep)",
        "hist_beh_seq_2": "倒数第2次行为 (Deep)",
        "hist_beh_seq_3": "倒数第3次行为 (Deep)",
        "hist_beh_seq_4": "倒数第4次行为 (Deep)",
        "hist_beh_seq_5": "倒数第5次行为 (Deep)",
    }
    mapped_features = [feature_names_mapping[col] for col in feature_cols]

    # 构建 DataFrame 并排序
    feat_df = pd.DataFrame({"Feature": mapped_features, "Importance": importances})
    feat_df = feat_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei"]

    sns.barplot(x="Importance", y="Feature", data=feat_df, palette="viridis")

    plt.title("核心业务特征重要度排行 (Feature Importance)", fontsize=16, pad=20)
    plt.xlabel("特征重要性得分 (Feature Splits)", fontsize=14)
    plt.ylabel("特征名称", fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(figures_dir, "feature_importance.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"✅ 特征重要性图已保存至: {save_path}")


if __name__ == "__main__":
    plot_model_gauc_comparison()
    plot_feature_importance()
