import os
import sys
import torch
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# 智能路径识别
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_dir) == "src":
    project_root = os.path.dirname(current_dir)
else:
    project_root = current_dir

sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

# 导入我们的 Wide-DIN 模型
from advanced_din import ActualWideDIN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# ==========================================
# 1. 定义 API 接收的 JSON 数据结构 (使用 Pydantic 进行严格校验)
# ==========================================
class UserRequest(BaseModel):
    user_id: int = Field(..., description="用户 ID")
    category_seq: List[int] = Field(
        ..., min_items=5, max_items=5, description="最近 5 次访问的类目 ID 序列"
    )
    behavior_seq: List[int] = Field(
        ...,
        min_items=5,
        max_items=5,
        description="最近 5 次动作序列 (1浏览 2收藏 3加购 4购买)",
    )
    target_category: int = Field(..., description="当前曝光/候选商品的类目 ID")
    # 无泄漏的 4 个统计特征: hist_pv, hist_fav, hist_cart, hist_buy
    dense_features: List[float] = Field(
        ..., min_items=4, max_items=4, description="Point-in-Time 历史行为计数"
    )


# ==========================================
# 2. 初始化 FastAPI 应用与全局模型
# ==========================================
app = FastAPI(
    title="Taobao AI Recommendation Engine",
    description="Wide-DIN 在线推理服务 V1.0",
    version="1.0.0",
)

# 全局变量，避免每次请求都重新加载模型
model = None
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@app.on_event("startup")
def load_model():
    """在服务器启动时把模型加载到内存中 (单例模式)"""
    global model
    model_path = os.path.join(project_root, "models", "taobao_wide_din_pit.pth")

    logging.info(f"🚀 正在启动推理服务... 加载模型: {model_path}")
    if not os.path.exists(model_path):
        raise RuntimeError(f"找不到模型文件: {model_path}，请确认路径是否正确。")

    # 初始化网络架构并加载权重
    model = ActualWideDIN(num_dense_features=4).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()  # 必须切换到 eval 模式
    logging.info("✅ AI 引擎加载完毕，准备接收并发请求！")


# ==========================================
# 3. 核心推理接口 /predict
# ==========================================
@app.post("/predict")
def predict_action(request: UserRequest):
    """
    接收用户当前状态 JSON，返回下一个动作的概率分布
    """
    if model is None:
        raise HTTPException(status_code=500, detail="模型未成功加载")

    try:
        # 1. 将 JSON 数据转换为 PyTorch Tensor (增加 batch 维度)
        # 注意对 category 进行取模防止越界 (模拟 embedding hash)
        c_seq = torch.tensor(
            [[c % 10000 for c in request.category_seq]], dtype=torch.long
        ).to(device)
        b_seq = torch.tensor([request.behavior_seq], dtype=torch.long).to(device)
        t_cat = torch.tensor([request.target_category % 10000], dtype=torch.long).to(
            device
        )

        # 对连续特征做 log1p 平滑 (与训练时保持一致)
        d_feats = torch.tensor([request.dense_features], dtype=torch.float32)
        d_feats = torch.log1p(d_feats).to(device)

        # 2. 前向传播
        with torch.no_grad():
            outputs = model(c_seq, b_seq, t_cat, d_feats)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        # 3. 封装返回结果
        return {
            "status": "success",
            "user_id": request.user_id,
            "target_category": request.target_category,
            "predictions": {
                "View (浏览)": f"{probs[0] * 100:.2f}%",
                "Fav (收藏)": f"{probs[1] * 100:.2f}%",
                "Cart (加购)": f"{probs[2] * 100:.2f}%",
                "Buy (购买)": f"{probs[3] * 100:.2f}%",
            },
            "recommended_action": ["View", "Fav", "Cart", "Buy"][int(probs.argmax())],
        }

    except Exception as e:
        logging.error(f"推理请求失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"数据处理异常: {str(e)}")


# ==========================================
# 4. 健康检查接口
# ==========================================
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Taobao Wide-DIN Inference API"}


if __name__ == "__main__":
    # 在本地的 8000 端口启动服务
    logging.info("启动 Uvicorn 服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
