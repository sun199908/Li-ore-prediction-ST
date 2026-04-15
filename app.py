import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. 页面配置与常量定义 ---
st.set_page_config(page_title="铝土矿伴生锂预测", layout="wide", page_icon="💎")

st.title("💎 铝土矿伴生锂矿化预测模型")
st.markdown("""
本系统基于主量元素含量及地质背景数据，预测铝土矿的锂矿化分类。
**操作说明：** 在左侧输入单样品数据进行预测，或在下方上传 Excel 文件进行批量处理。
""")

# 分类标签映射
CLASS_MAPPING = {0: '无矿', 1: '矿化'}

# 特征约束映射 (根据你的要求设定)
# 注意：后台转换的数字 (0, 1, 2...) 需确保与你模型训练时的编码顺序一致
ERA_OPTIONS = ["早石炭世", "晚石炭世", "早二叠世", "晚二叠世"]
ERA_MAP = {val: i for i, val in enumerate(ERA_OPTIONS)}

ZONE_OPTIONS = ["黔北", "黔中", "山西", "河南", "山东", "桂北"]
ZONE_MAP = {val: i for i, val in enumerate(ZONE_OPTIONS)}

# 模型预期的最终特征顺序
FINAL_FEATURES = ["Al2O3", "SiO2", "Fe2O3", "A/S", "成矿时代", "成矿区带"]

# --- 2. 加载模型资产 ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('SVC_model4.15.joblib')
        scaler = joblib.load('scalerLi4.15.joblib')
        return model, scaler
    except Exception as e:
        st.error(f"模型文件加载失败，请检查目录下是否存在 .joblib 文件。错误: {e}")
        st.stop()

model, scaler = load_assets()

# --- 3. 核心计算函数 ---
def preprocess_data(df_input):
    """
    处理原始数据：计算 A/S，并将中文文本映射为数值
    """
    df = df_input.copy()
    
    # 1. 文本标签转数值 (针对批量上传的情况)
    if df['成矿时代'].dtype == object:
        df['成矿时代'] = df['成矿时代'].map(ERA_MAP)
    if df['成矿区带'].dtype == object:
        df['成矿区带'] = df['成矿区带'].map(ZONE_MAP)
        
    # 2. 计算铝硅比 (A/S)，处理除以0的情况
    eps = 1e-5
    df['A/S'] = df["Al2O3"] / np.where(df["SiO2"] == 0, eps, df["SiO2"])
    
    # 3. 严格按照模型特征顺序排序
    return df[FINAL_FEATURES]

# --- 4. 侧边栏：单样品输入 ---
st.sidebar.header("📥 单样品数据输入")

def get_single_input():
    # 主量元素输入
    al2o3 = st.sidebar.number_input("Al2O3 (wt%)", min_value=0.0, max_value=100.0, value=60.0, format="%.2f")
    sio2 = st.sidebar.number_input("SiO2 (wt%)", min_value=0.0, max_value=100.0, value=10.0, format="%.2f")
    fe2o3 = st.sidebar.number_input("Fe2O3 (wt%)", min_value=0.0, max_value=100.0, value=5.0, format="%.2f")
    
    # 约束选项输入
    era = st.sidebar.selectbox("成矿时代", options=ERA_OPTIONS)
    zone = st.sidebar.selectbox("成矿区带", options=ZONE_OPTIONS)
    
    # 构建初始 DataFrame (此时成矿时代/区带还是中文)
    data = {
        "Al2O3": al2o3,
        "SiO2": sio2,
        "Fe2O3": fe2o3,
        "成矿时代": era,
        "成矿区带": zone
    }
    return pd.DataFrame([data])

input_df = get_single_input()

# --- 5. 单样品预测逻辑 ---
st.subheader("📊 单样品预测结果")

if st.button("开始预测"):
    # 数据预处理
    processed_input = preprocess_data(input_df)
    
    # 标准化与预测
    scaled_input = scaler.transform(processed_input)
    pred_idx = int(model.predict(scaled_input)[0])
    
    # 获取概率 (需确认模型支持 predict_proba)
    try:
        pred_proba = model.predict_proba(scaled_input)[0]
    except:
        pred_proba = None

    # 结果展示
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        result_text = CLASS_MAPPING.get(pred_idx, "未知")
        st.success(f"### 预测结论：\n# **{result_text}**")
        st.info(f"**计算得到的 A/S:** {processed_input['A/S'].iloc[0]:.2f}")
        
    with res_col2:
        if pred_proba is not None:
            st.write("#### 预测概率分布")
            proba_df = pd.DataFrame([pred_proba], columns=[CLASS_MAPPING[c] for c in model.classes_])
            st.bar_chart(proba_df.T)

# --- 6. 批量预测模块 ---
st.markdown("---")
st.subheader("📂 批量预测")
st.write(f"上传的 Excel 必须包含以下列：`Al2O3, SiO2, Fe2O3, 成矿时代, 成矿区带`")
st.caption(f"注：成矿时代必须为 {ERA_OPTIONS} 中的值；成矿区带必须为 {ZONE_OPTIONS} 中的值。")

uploaded_file = st.file_uploader("选择 Excel 文件 (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        batch_raw = pd.read_excel(uploaded_file)
        required_cols = ["Al2O3", "SiO2", "Fe2O3", "成矿时代", "成矿区带"]
        
        # 检查缺失列
        missing = [c for c in required_cols if c not in batch_raw.columns]
        if missing:
            st.error(f"文件格式错误，缺少以下列: {missing}")
        else:
            with st.spinner('正在处理数据...'):
                # 1. 预处理
                batch_processed = preprocess_data(batch_raw)
                # 2. 标准化
                batch_scaled = scaler.transform(batch_processed)
                # 3. 预测
                batch_preds = model.predict(batch_scaled)
                
                # 4. 结果整理
                result_df = batch_raw.copy()
                result_df['预测结果'] = [CLASS_MAPPING.get(int(i), "未知") for i in batch_preds]
                
                # 5. 展示与下载
                st.success("批量预测完成！")
                st.dataframe(result_df)
                
                csv = result_df.to_csv(index=False).encode('utf-8-sig') # 使用 utf-8-sig 防止 Excel 打开中文乱码
                st.download_button(
                    label="📥 下载预测报告 (CSV)",
                    data=csv,
                    file_name="Li_Mineralization_Results.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.error(f"批量处理过程中发生错误: {e}")