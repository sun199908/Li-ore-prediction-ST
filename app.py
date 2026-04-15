import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="锂矿预测模型", layout="wide", page_icon="💎")

# --- 1. 配置映射关系 (请确保这里的 0,1,2,3 顺序与训练数据一致) ---
ERA_OPTIONS = ["早二叠世", "晚二叠世", "早石炭世", "中石炭世", "晚石炭世"]
ERA_MAP = {val: i for i, val in enumerate(ERA_OPTIONS)}

ZONE_OPTIONS = ["黔中", "滇东-贵西", "黔北-渝南", "河南", "山西"]
ZONE_MAP = {val: i for i, val in enumerate(ZONE_OPTIONS)}

# 必须与训练时的 features 列表顺序完全一致
FINAL_FEATURES = ["Al2O3", "SiO2", "Fe2O3", "A/S", "成矿时代", "成矿区带"]
CLASS_MAPPING = {0: '无矿', 1: '矿化'}

# --- 2. 加载模型 ---
@st.cache_resource
def load_assets():
    model = joblib.load('SVC_model4.15.joblib')
    scaler = joblib.load('scalerLi4.15.joblib')
    return model, scaler

model, scaler = load_assets()

# --- 3. 预处理函数 ---
def preprocess_data(df_input):
    df = df_input.copy()
    
    # 处理文本映射
    if '成矿时代' in df.columns and df['成矿时代'].dtype == object:
        df['成矿时代'] = df['成矿时代'].map(ERA_MAP)
    if '成矿区带' in df.columns and df['成矿区带'].dtype == object:
        df['成矿区带'] = df['成矿区带'].map(ZONE_MAP)
        
    # 计算 A/S
    eps = 1e-5
    df['A/S'] = df["Al2O3"] / np.where(df["SiO2"] == 0, eps, df["SiO2"])
    
    # 按照训练时的特征顺序提取，并处理缺失值
    df = df[FINAL_FEATURES].fillna(0)
    
    # 【核心修复】返回 .values (numpy数组)，避开 sklearn 的特征名检查
    return df.astype(float)

# --- 4. UI 逻辑 ---
st.title("💎 铝土矿伴生锂预测模型")

# 侧边栏单样品输入
with st.sidebar:
    st.header("📥 单样品输入")
    al = st.number_input("Al2O3 (wt%)", value=60.0)
    si = st.number_input("SiO2 (wt%)", value=10.0)
    fe = st.number_input("Fe2O3 (wt%)", value=5.0)
    era = st.selectbox("成矿时代", ERA_OPTIONS)
    zone = st.selectbox("成矿区带", ZONE_OPTIONS)

# 预测按钮
if st.button("开始预测"):
    # 构建输入数据
    single_data = pd.DataFrame([{
        "Al2O3": al, "SiO2": si, "Fe2O3": fe, "成矿时代": era, "成矿区带": zone
    }])
    
    processed_df = preprocess_data(single_data)
    
    # 【关键】使用 .values 确保不带列名进入 scaler
    input_scaled = scaler.transform(processed_df.values) 
    
    pred = model.predict(input_scaled)[0]
    
    st.success(f"### 预测结果: **{CLASS_MAPPING.get(int(pred), '未知')}**")
    
    # 只有当模型训练时开启了 probability=True 才能运行以下代码
    try:
        probs = model.predict_proba(input_scaled)[0]
        st.write("#### 预测概率分布")
        st.bar_chart(pd.Series(probs, index=["无矿", "矿化"]))
    except AttributeError:
        st.warning("提示：当前模型文件不支持概率预测，请在训练时设置 SVC(probability=True)。")

# 批量预测部分 (略，逻辑同上，使用 preprocess_data(batch_raw).values 即可)