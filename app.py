import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. 页面配置 ---
st.set_page_config(page_title="锂矿预测系统", layout="wide", page_icon="💎")

# --- 2. 常量定义 (必须与训练模型时的编码一致) ---
# 定义映射字典：将中文选项映射为训练时使用的数值
ERA_OPTIONS = ["早石炭世", "晚石炭世", "早二叠世", "晚二叠世"]
ERA_MAP = {val: float(i) for i, val in enumerate(ERA_OPTIONS)}

ZONE_OPTIONS = ["黔北", "黔中", "山西", "河南", "山东", "桂北"]
ZONE_MAP = {val: float(i) for i, val in enumerate(ZONE_OPTIONS)}

# 对应训练脚本中的英文特征名
FINAL_FEATURES = ["Al2O3", "SiO2", "Fe2O3", "A/S", "Mineralization age", "Mineralization belt"]
CLASS_MAPPING = {0: '无矿', 1: '矿化'}

# --- 3. 加载模型资产 ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('SVC_model4.15.joblib')
        scaler = joblib.load('scalerLi4.15.joblib')
        return model, scaler
    except Exception as e:
        st.error(f"无法加载模型文件，请检查目录下是否存在对应的 .joblib 文件。错误: {e}")
        st.stop()

model, scaler = load_assets()

# --- 4. 核心预处理逻辑 ---
def preprocess_data(df_input):
    """
    将原始输入(包含中文和数值)转换为模型可读取的纯数字格式
    """
    df = df_input.copy()
    
    # A. 确保数值列是数字类型 (处理 Excel 读取时可能的格式问题)
    numeric_cols = ["Al2O3", "SiO2", "Fe2O3"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # B. 计算 A/S
    eps = 1e-5
    df['A/S'] = df["Al2O3"] / (df["SiO2"] + eps)
    
    # C. 映射文本列为数值 (处理‘成矿时代’和‘成矿区带’)
    # 如果列里已经是数字，则保留；如果是中文，则按字典映射
    if '成矿时代' in df.columns:
        df['Mineralization age'] = df['成矿时代'].map(ERA_MAP).fillna(df['成矿时代'])
    if '成矿区带' in df.columns:
        df['Mineralization belt'] = df['成矿区带'].map(ZONE_MAP).fillna(df['成矿区带'])
    
    # D. 最终校验：提取特征列并强制转为 float
    # errors='coerce' 会将无法转换的残余文本变为 NaN，随后用 0 填充
    df_final = df[FINAL_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return df_final

# --- 5. 用户界面 ---
st.title("💎 铝土矿伴生锂预测模型")
st.markdown("---")

# 侧边栏：单样品输入
with st.sidebar:
    st.header("📥 单样品输入")
    input_al = st.number_input("Al2O3 (wt%)", value=60.0, step=0.1)
    input_si = st.number_input("SiO2 (wt%)", value=10.0, step=0.1)
    input_fe = st.number_input("Fe2O3 (wt%)", value=5.0, step=0.1)
    input_era = st.selectbox("成矿时代", ERA_OPTIONS)
    input_zone = st.selectbox("成矿区带", ZONE_OPTIONS)

# 主界面：展示预测结果
col_left, col_right = st.columns([1, 1])

with col_left:
    if st.button("🚀 开始单样品预测"):
        # 构建 DataFrame
        single_df = pd.DataFrame([{
            "Al2O3": input_al, "SiO2": input_si, "Fe2O3": input_fe,
            "成矿时代": input_era, "成矿区带": input_zone
        }])
        
        # 预处理
        processed_data = preprocess_data(single_df)
        
        # 标准化与预测 (使用 .values 避免特征名警告)
        scaled_data = scaler.transform(processed_data.values)
        prediction = model.predict(scaled_data)[0]
        
        # 结果显示
        res_label = CLASS_MAPPING.get(int(prediction), "未知")
        st.success(f"### 预测结论: **{res_label}**")
        st.write("计算得到的特征值：")
        st.dataframe(processed_data)
        
        # 概率预测 (仅当训练时开启了 probability=True)
        try:
            probs = model.predict_proba(scaled_data)[0]
            st.write("#### 预测概率")
            prob_df = pd.DataFrame([probs], columns=["无矿概率", "矿化概率"])
            st.bar_chart(prob_df.T)
        except:
            st.info("提示：模型未开启概率预测功能。")

# 批量预测模块
with col_right:
    st.header("📂 批量预测")
    uploaded_file = st.file_uploader("上传 Excel 文件", type=["xlsx"])
    
    if uploaded_file:
        batch_raw = pd.read_excel(uploaded_file)
        # 检查必要的列名是否存在
        required = ["Al2O3", "SiO2", "Fe2O3", "成矿时代", "成矿区带"]
        if all(col in batch_raw.columns for col in required):
            if st.button("📊 执行批量分析"):
                with st.spinner("处理中..."):
                    batch_processed = preprocess_data(batch_raw)
                    batch_scaled = scaler.transform(batch_processed.values)
                    batch_preds = model.predict(batch_scaled)
                    
                    # 结果合并
                    results = batch_raw.copy()
                    results['预测结果'] = [CLASS_MAPPING.get(int(p), "未知") for p in batch_preds]
                    
                    st.dataframe(results)
                    # 下载按钮
                    csv = results.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 下载结果表格", data=csv, file_name="batch_results.csv")
        else:
            st.error(f"上传文件必须包含以下列：{required}")
