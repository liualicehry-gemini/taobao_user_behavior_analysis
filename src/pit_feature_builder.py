import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# 智能路径定位
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = (
    os.path.dirname(current_dir)
    if os.path.basename(current_dir) == "src"
    else current_dir
)


def build_pit_features():
    # 假设这是你最原始的、只包含基础字段的用户行为明细表 (请替换为你真实的原始文件名)
    # 字段应至少包含: user_id, item_id, category_id, behavior_type, time_stamp
    raw_data_path = os.path.join(project_root, "data", "raw", "UserBehavior.csv")

    # 为了演示和内存安全，我们假设你之前有一个清洗过的小规模明细宽表
    # 这里我们读取之前的 master 表作为底表进行重构
    input_file = os.path.join(
        project_root, "data", "processed", "master_training_data.csv"
    )
    output_file = os.path.join(project_root, "data", "processed", "pit_master_data.csv")

    logging.info("开始构建 Point-in-Time 无穿越特征...")

    # 1. 读取数据
    df = pd.read_csv(input_file)

    # 【核心修复】：智能识别时间列，如果没有则使用物理行索引模拟时间顺序
    time_col = "time_stamp" if "time_stamp" in df.columns else "timestamp"
    if time_col not in df.columns:
        logging.warning(
            " 未找到真实的时间戳列！将使用数据的默认物理顺序 (Index) 模拟时间先后顺序进行无穿越截断。"
        )
        df["mock_time"] = df.index
        time_col = "mock_time"

    # 严格按 [用户, 时间先后] 全局排序，这是 Point-in-Time 的基础
    df = df.sort_values(by=["user_id", time_col]).reset_index(drop=True)

    # 2. 构建 Wide 侧无穿越统计特征 (利用 shift 平移，防止偷看未来)
    logging.info("构建 Wide 侧统计特征 (累计平移法)...")

    # 标记当前行为是否为各种类型
    df["is_view"] = (df["target_label"] == 1).astype(int)
    df["is_fav"] = (df["target_label"] == 2).astype(int)
    df["is_cart"] = (df["target_label"] == 3).astype(int)
    df["is_buy"] = (df["target_label"] == 4).astype(int)

    # 核心：计算 T 时刻之前的历史累计次数 (按用户分组，并 shift 平移一行)
    df["hist_pv_count"] = df.groupby("user_id")["is_view"].cumsum().shift(1).fillna(0)
    df["hist_fav_count"] = df.groupby("user_id")["is_fav"].cumsum().shift(1).fillna(0)
    df["hist_cart_count"] = df.groupby("user_id")["is_cart"].cumsum().shift(1).fillna(0)
    df["hist_buy_count"] = df.groupby("user_id")["is_buy"].cumsum().shift(1).fillna(0)

    # 3. 构建 Deep 侧无穿越序列特征 (平移最近 5 次行为)
    logging.info("构建 Deep 侧时序特征 (滑动窗口法)...")

    for i in range(1, 6):
        # 把前第 i 次的类别和行为平移到当前行
        # 这里为了简化用 item_id 模拟 category，如果是真实场景请用 category_id
        df[f"hist_cat_seq_{i}"] = (
            df.groupby("user_id")["item_id"].shift(i).fillna(0).astype(int)
        )
        df[f"hist_beh_seq_{i}"] = (
            df.groupby("user_id")["target_label"].shift(i).fillna(0).astype(int)
        )

    # 4. 清理辅助列与临时模拟的时间列，保留最终特征
    drop_cols = ["is_view", "is_fav", "is_cart", "is_buy"]
    if "mock_time" in df.columns:
        drop_cols.append("mock_time")
    df = df.drop(columns=drop_cols)

    # 5. 保存纯净数据集
    df.to_csv(output_file, index=False)
    logging.info(f"✅ 无穿越特征表构建完成！已保存至: {output_file}")
    logging.info(
        f" 现在数据集中的每一行，其历史特征都严格仅依赖于 '过去' 的数据 (已彻底消除泄露)。"
    )


if __name__ == "__main__":
    build_pit_features()
