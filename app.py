import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="锂矿预测模型", layout="wide", page_icon="💎")

st.title("💎 铝土矿伴生锂预测模型")
st.markdown("""
Welcome! This system predicts the genesis or source classification of pyrite based on its trace element composition.
**Instructions:** Input the raw elemental concentrations (in wt%). The system will automatically calculate the necessary elemental ratios and provide a prediction.
""")

# --- Define Class Mapping ---
# 字典映射：将模型输出的 0, 1, 2 转换为具体的地质名词
CLASS_MAPPING = {
    0: '无矿',
    1: '矿化',
}

# --- Load Models and Scaler ---
@st.cache_resource
def load_assets():
    # 只需要加载模型和Scaler，不再需要 LabelEncoder
    model = joblib.load('SVC_model4.15.joblib')
    scaler = joblib.load('scalerLi4.15.joblib')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Failed to load required model files. Please ensure the .joblib files exist in the directory. Error: {e}")
    st.stop()

# --- Define Feature Lists ---
# 1. Final features required by the XGBoost model (must match exactly)
final_feature_names = [
    "Al2O3", "SiO2", "Fe2O3", "A/S", "成矿时代", "成矿区带"
]

# 2. Raw elements needed from the user (Notice we added 'Zn' because it's needed for 'Cu/Zn')
raw_elements = ["Al2O3", "SiO2", "Fe2O3", "成矿时代", "成矿区带"]

# --- Helper Function to Calculate Ratios ---
def compute_features(df_raw):
    """
    Takes a DataFrame of raw elements and calculates the required ratios.
    Handles division by zero by replacing 0 with a very small number (1e-5).
    """
    df = df_raw.copy()
    eps = 1e-5 # Proxy for detection limit to avoid ZeroDivisionError
    
    df['A/S'] = df["Al2O3"] / np.where(df["SiO2"] == 0, eps, df["SiO2"])
    
    # Return strictly the columns the model expects in the correct order
    return df[final_feature_names]

# --- Sidebar: Single Sample Input ---
st.sidebar.header("📥 Input Raw Elements (ppm)")

def user_input_features():
    data = {}
    col1, col2 = st.sidebar.columns(2)
    
    # Create input fields for the 11 raw elements
    for i, element in enumerate(raw_elements):
        if i % 2 == 0:
            val = col1.number_input(f"{element}", min_value=0.0, value=0.0, format="%.4f")
        else:
            val = col2.number_input(f"{element}", min_value=0.0, value=0.0, format="%.4f")
        data[element] = val
        
    return pd.DataFrame([data])

input_raw_df = user_input_features()

# --- Prediction Logic ---
st.subheader("📊 单样品预测")

if st.button("点击预测"):
    # 1. Compute ratios internally
    input_processed = compute_features(input_raw_df)
    
    # 2. Scale features (Requires column names to match)
    input_scaled = scaler.transform(input_processed)
    
    # 3. Model Prediction (Outputs 0, 1, or 2)
    pred_idx = int(model.predict(input_scaled)[0])
    pred_proba = model.predict_proba(input_scaled)[0]
    
    # 4. Map numeric label to String using the Dictionary
    predicted_class = CLASS_MAPPING.get(pred_idx, f"Unknown Class ({pred_idx})")
    
    # Map the probability labels for the chart
    class_labels = [CLASS_MAPPING.get(int(cls), f"Class {cls}") for cls in model.classes_]

    # 5. Display Results
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.success(f"### Predicted Source:\n# **{predicted_class}**")
        st.info("💡 **Calculated Ratios:**")
        # Display the calculated features for the user's reference
        ratios_only = input_processed.drop(columns=[col for col in raw_elements if col != 'Zn' and col in input_processed.columns])
        st.dataframe(ratios_only.T.style.format("{:.4f}"))
        
    with col_res2:
        st.write("#### 预测概率")
        proba_df = pd.DataFrame([pred_proba], columns=class_labels)
        st.bar_chart(proba_df.T)

# --- Batch Upload (Advanced Feature) ---
st.markdown("---")
st.subheader("📂 批量预测")
st.write("上传文件. 文件中必须包含以下列: ‘Al2O3, SiO2, Fe2O3, 成矿时代, 成矿区带’")

uploaded_file = st.file_uploader("上传Excel文件 (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        batch_raw = pd.read_excel(uploaded_file)
        
        # Check if all required raw elements are present
        missing_cols = [col for col in raw_elements if col not in batch_raw.columns]
        
        if missing_cols:
            st.error(f"文件中缺少需要的列: {', '.join(missing_cols)}")
        else:
            st.write("上传数据预览:")
            st.dataframe(batch_raw.head())
            
            with st.spinner('Calculating ratios and predicting...'):
                # 1. Calculate ratios
                batch_processed = compute_features(batch_raw)
                
                # 2. Scale and Predict (Outputs array of 0, 1, 2)
                batch_scaled = scaler.transform(batch_processed)
                batch_preds_idx = model.predict(batch_scaled)
                
                # 3. Map numbers to string labels using list comprehension
                batch_preds = [CLASS_MAPPING.get(int(idx), "未知") for idx in batch_preds_idx]
                
                # 4. Append to original dataframe
                result_df = batch_raw.copy()
                result_df['Predicted_Source'] = batch_preds
                
            st.success("批量预测完成")
            st.write(result_df)
            
            # Allow user to download results as CSV
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 下载预测结果",
                data=csv,
                file_name="pyrite_predictions.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"预测过程发生错误: {e}")