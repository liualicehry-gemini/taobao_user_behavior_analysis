import os
import sys
import random
import torch
import torch.nn as nn
import streamlit as st
import pandas as pd
import requests

# ==========================================
# 0. 路径配置与模块动态导入
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = (
    os.path.dirname(current_dir)
    if os.path.basename(current_dir) == "src"
    else current_dir
)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from oot_pipeline import PiTOOTDataset

# ==========================================
# 1. 页面配置与 淘宝风 (Taobao Style) UI 样式
# ==========================================
st.set_page_config(
    page_title="Taobao AI Engine | 淘宝推荐控制台",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 注入淘宝专属 CSS 样式
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=PingFang+SC:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'PingFang SC', 'Helvetica Neue', Arial, sans-serif;
        background-color: #f4f4f4;
    }

    .block-container { padding-top: 3.5rem; padding-bottom: 2rem; }

    /* 淘宝顶部导航栏 */
    .tb-navbar {
        background: linear-gradient(90deg, #ff9000 0%, #ff5000 100%);
        padding: 15px 30px;
        border-radius: 12px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 10px rgba(255, 80, 0, 0.2);
        margin-bottom: 20px;
    }
    .tb-navbar-title { font-size: 24px; font-weight: 700; letter-spacing: 1px; display: flex; align-items: center; gap: 10px;}
    .tb-navbar-subtitle { font-size: 14px; opacity: 0.9; }

    /* 淘宝商品卡片 */
    .tb-product-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.2s;
        border: 1px solid #eee;
    }
    .tb-product-card:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,0.1); }
    .tb-img-placeholder {
        border-radius: 8px;
        height: 180px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 12px;
    }
    .tb-price { color: #ff5000; font-size: 24px; font-weight: 700; margin-top: 8px; }
    .tb-price span { font-size: 14px; margin-right: 2px; }
    .tb-tag { background: #ffe4d0; color: #ff5000; font-size: 11px; padding: 2px 6px; border-radius: 4px; margin-left: 6px; }

    /* 用户信息条 */
    .tb-user-bar {
        background: white;
        border-radius: 12px;
        padding: 20px;
        display: flex;
        gap: 30px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }
    .tb-stat-box { display: flex; flex-direction: column; }
    .tb-stat-label { color: #999; font-size: 13px; margin-bottom: 4px; }
    .tb-stat-value { color: #333; font-size: 20px; font-weight: 600; }

    /* 引擎对比面板 */
    .engine-panel {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-top: 4px solid #ff5000;
    }
    .engine-panel-gru { border-top: 4px solid #94a3b8; }

    /* big size buy button style*/
    .tb-buy-btn {
        background: linear-gradient(90deg, #ff9000, #ff5000);
        color: white;
        text-align: center;
        padding: 10px 0;
        border-radius: 20px;
        font-weight: 600;
        font-size: 16px;
        margin-top: 15px;
        box-shadow: 0 4px 6px rgba(255, 80, 0, 0.2);
        cursor: pointer;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# progress bar
def draw_tb_progress(label, percentage, color, icon=""):
    html = f"""
    <div style="margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <span style="font-size: 14px; color: #666;">{icon} {label}</span>
            <span style="font-size: 14px; font-weight: 600; color: #333;">{percentage:.2f}%</span>
        </div>
        <div style="width: 100%; background-color: #f0f0f0; border-radius: 10px; height: 8px; overflow: hidden;">
            <div style="background: {color}; height: 100%; border-radius: 10px; width: {percentage}%;"></div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ==========================================
# 2. old version GRU architecture
# ==========================================
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
        _, hidden_state = self.gru(combined_input)
        return self.fc(hidden_state[-1])


# ==========================================
# 3. data and benchmark model loading
# ==========================================
@st.cache_resource(show_spinner=False)
def load_gru_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    gru_model = TaobaoBehaviorGRU().to(device)
    gru_path = os.path.join(project_root, "models", "taobao_sequence_model.pth")
    if os.path.exists(gru_path):
        gru_model.load_state_dict(
            torch.load(gru_path, map_location=device, weights_only=True)
        )
    gru_model.eval()
    return gru_model, device


@st.cache_data(show_spinner=False)
def load_dataset():
    csv_file = os.path.join(project_root, "data", "processed", "pit_master_data.csv")
    return PiTOOTDataset(csv_file, mode="test")


# ==========================================
# 4. 淘宝导航栏 & 控制流
# ==========================================
st.markdown(
    """
    <div class="tb-navbar">
        <div class="tb-navbar-title">🛒 淘宝网 AI 算法控制台 (Taobao RecSys)</div>
        <div class="tb-navbar-subtitle">V2.0 微服务架构 · 实时流量监测中</div>
    </div>
""",
    unsafe_allow_html=True,
)

with st.spinner("正在连接淘宝大数据集群..."):
    dataset = load_dataset()
    gru_model, device = load_gru_model()

st.sidebar.image(
    "https://img.alicdn.com/tfs/TB1_uT8a5ERMeJjSspiXXbZLFXa-143-59.png", width=140
)
st.sidebar.markdown("### 线上流量模拟器")
if st.sidebar.button("捕获下一个活跃用户", use_container_width=True, type="primary"):
    st.session_state["random_idx"] = random.randint(0, len(dataset) - 1)
if "random_idx" not in st.session_state:
    st.session_state["random_idx"] = 0

idx = st.session_state["random_idx"]
user_id, cat_seq, beh_seq, target_cat, dense_features, target_beh = dataset[idx]
actual_action = target_beh.item() + 1
behavior_map = {1: "浏览", 2: "收藏", 3: "加购", 4: "购买"}

# 用户画像条
st.markdown(
    f"""
    <div class="tb-user-bar">
        <div class="tb-stat-box"><span class="tb-stat-label">当前活跃用户</span><span class="tb-stat-value">UID: {int(user_id)}</span></div>
        <div class="tb-stat-box"><span class="tb-stat-label">历史累计浏览</span><span class="tb-stat-value">{int(dense_features[0].item())} 次</span></div>
        <div class="tb-stat-box"><span class="tb-stat-label">历史累计收藏</span><span class="tb-stat-value">{int(dense_features[1].item())} 次</span></div>
        <div class="tb-stat-box"><span class="tb-stat-label">历史累计加购</span><span class="tb-stat-value">{int(dense_features[2].item())} 次</span></div>
        <div class="tb-stat-box"><span class="tb-stat-label">历史成功购买</span><span class="tb-stat-value" style="color:#ff5000;">{int(dense_features[3].item())} 单</span></div>
    </div>
""",
    unsafe_allow_html=True,
)


st.markdown("###  正在曝光的商品 (Candidate Item)")
col_prod, col_blank1, col_blank2 = st.columns([1, 1, 2])
with col_prod:
    # ----------------------------------------------------
    #  算法伪随机生成方案：通过类目 ID 确定性映射商品图标和背景色
    # ----------------------------------------------------
    cat_id_int = int(target_cat.item())

    # 准备一套电商核心品类的图标与标题词库
    product_mocks = [
        ("👗", "法式复古收腰连衣裙 2026新款"),
        ("📱", "全新 5G 智能旗舰手机 512GB"),
        ("👟", "男女同款 运动透气老爹鞋"),
        ("💄", "丝绒哑光唇泥 显白持久不掉色"),
        ("💻", "轻薄本 游戏本 16英寸高性能"),
        ("🍔", "双层吉士汉堡 充饥代餐速食"),
        ("🧸", "大号毛绒玩具 熊公仔 生日礼物"),
        ("📚", "精装版 世界名著 全套正版"),
        ("⌚️", "智能手表 运动计步 蓝牙通话"),
        ("👓", "防蓝光辐射眼镜 男女平光镜"),
        ("👜", "真皮单肩斜挎包 百搭通勤款"),
        ("🎧", "主动降噪 蓝牙无线头戴式耳机"),
        ("🛋️", "北欧极简布艺沙发 客厅小户型"),
        ("🧴", "氨基酸洗面奶 温和控油深层清洁"),
    ]
    # 取模保证同一个类目ID每次渲染出的都是同一个商品
    mock_item = product_mocks[cat_id_int % len(product_mocks)]

    # 根据类目ID生成动态的、固定的优美渐变色背景 (通过质数打乱色相)
    hue1 = (cat_id_int * 137) % 360
    hue2 = (cat_id_int * 211) % 360
    gradient_bg = (
        f"linear-gradient(135deg, hsl({hue1}, 80%, 85%), hsl({hue2}, 80%, 75%))"
    )

    # 模拟一个逼真的淘宝商品卡片 (免图片版)
    st.markdown(
        f"""
        <div class="tb-product-card">
            <div class="tb-img-placeholder" style="background: {gradient_bg};">
                <span style="font-size: 70px; filter: drop-shadow(2px 4px 6px rgba(0,0,0,0.15));">{mock_item[0]}</span>
            </div>
            <div style="font-size: 15px; color: #333; line-height: 1.4; font-weight: 500;">
                {mock_item[1]} <span style="color:#999;font-size:12px;">(类目: {cat_id_int})</span> 
                <span class="tb-tag">包邮</span> <span class="tb-tag" style="background:#fff1f0; color:#ff4d4f;">极速退款</span>
            </div>
            <div class="tb-price"><span>￥</span>{cat_id_int % 400 + 29}.00</div>
            <div style="font-size: 12px; color: #999; margin-top: 4px;">月销 {(cat_id_int * 7) % 9000 + 100}+ &nbsp;|&nbsp; 浙江杭州</div>
            <div class="tb-buy-btn">立即抢购</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<br>### ⚡ 推荐算法引擎实时推断", unsafe_allow_html=True)
col_v2, col_v1 = st.columns(2)

# ----------------------------------------------------
# 🌟 [链路 A]: 现代微服务链路 (FastAPI) - 淘宝核心版
# ----------------------------------------------------
with col_v2:
    st.markdown("<div class='engine-panel'>", unsafe_allow_html=True)
    st.markdown(
        "<h4 style='margin-top:0; color:#ff5000;'>线上核心引擎: Wide-DIN (远程 API)</h4>",
        unsafe_allow_html=True,
    )
    st.caption("网络层: HTTP 毫秒级请求 | 算法层: 注意力机制 & 无穿越特征")

    api_payload = {
        "user_id": int(user_id),
        "category_seq": cat_seq.tolist(),
        "behavior_seq": beh_seq.tolist(),
        "target_category": int(target_cat.item()),
        "dense_features": dense_features.tolist(),
    }

    API_URL = "http://127.0.0.1:8000/predict"

    try:
        response = requests.post(API_URL, json=api_payload, timeout=2)
        if response.status_code == 200:
            result = response.json()
            preds = result["predictions"]

            prob_din = [
                float(preds["View (浏览)"].strip("%")),
                float(preds["Fav (收藏)"].strip("%")),
                float(preds["Cart (加购)"].strip("%")),
                float(preds["Buy (购买)"].strip("%")),
            ]
            pred_din = int(torch.tensor(prob_din).argmax().item()) + 1

            st.markdown(
                f"<div style='background:#fff0e6; padding:12px; border-radius:8px; margin-bottom:15px; border:1px solid #ffccb3;'><span style='color:#ff5000; font-weight:600;'>基准事实 (真实用户动作): </span><span style='font-size:18px; font-weight:bold;'>{behavior_map.get(actual_action, '未知')}</span></div>",
                unsafe_allow_html=True,
            )

            status_icon = " 精准命中" if pred_din == actual_action else " 意图偏移"
            status_color = "#10b981" if pred_din == actual_action else "#f59e0b"
            st.markdown(
                f"**Wide-DIN 预判动作:** <span style='color:{status_color}; font-weight:bold;'>{behavior_map.get(pred_din, '未知')} ({status_icon})</span>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<hr style='margin:12px 0; border-color:#eee;'>", unsafe_allow_html=True
            )

            # 使用淘宝专属颜色的进度条
            draw_tb_progress("浏览概率", prob_din[0], "#ffd591", "👁️")
            draw_tb_progress("收藏概率", prob_din[1], "#ffa940", "❤️")
            draw_tb_progress("加购概率", prob_din[2], "#fa541c", "🛒")
            draw_tb_progress(
                "购买转化率 (CVR)",
                prob_din[3],
                "linear-gradient(90deg, #ff9000, #ff5000)",
                "💰",
            )

        else:
            st.error(f"API 异常: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error(" 服务断开：请确认 `python src/api.py` 正在后台运行。")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# [链路 B]: 传统单体链路 (GRU)
# ----------------------------------------------------
with col_v1:
    st.markdown("<div class='engine-panel engine-panel-gru'>", unsafe_allow_html=True)
    st.markdown(
        "<h4 style='margin-top:0; color:#475569;'>降级备用引擎: GRU (本地计算)</h4>",
        unsafe_allow_html=True,
    )
    st.caption("网络层: 本地同步阻塞 | 算法层: 存在特征穿越缺陷")

    with torch.no_grad():
        c_seq_t = cat_seq.unsqueeze(0).to(device)
        b_seq_t = beh_seq.unsqueeze(0).to(device)
        out_gru = gru_model(c_seq_t, b_seq_t)
        prob_gru = torch.softmax(out_gru[0], dim=0) * 100
        pred_gru = torch.argmax(out_gru[0]).item() + 1

    st.markdown(
        f"<div style='background:#f8fafc; padding:12px; border-radius:8px; margin-bottom:15px; border:1px solid #e2e8f0;'><span style='color:#64748b; font-weight:600;'>基准事实 (真实用户动作): </span><span style='font-size:18px; font-weight:bold;'>{behavior_map.get(actual_action, '未知')}</span></div>",
        unsafe_allow_html=True,
    )

    status_icon = " 命中" if pred_gru == actual_action else " 意图偏移"
    status_color = "#10b981" if pred_gru == actual_action else "#f59e0b"
    st.markdown(
        f"**GRU 预判动作:** <span style='color:{status_color}; font-weight:bold;'>{behavior_map.get(pred_gru, '未知')} ({status_icon})</span>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<hr style='margin:12px 0; border-color:#eee;'>", unsafe_allow_html=True
    )

    prob_gru_np = prob_gru.cpu().numpy()
    # gray and white cool color scheme contrast with core workflow
    draw_tb_progress("浏览概率", prob_gru_np[0], "#cbd5e1", "👁️")
    draw_tb_progress("收藏概率", prob_gru_np[1], "#cbd5e1", "❤️")
    draw_tb_progress("加购概率", prob_gru_np[2], "#cbd5e1", "🛒")
    draw_tb_progress("购买转化率 (CVR)", prob_gru_np[3], "#94a3b8", "💰")

    st.markdown("</div>", unsafe_allow_html=True)
