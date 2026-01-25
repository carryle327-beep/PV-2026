import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob

# --- 1. 页面配置 ---
st.set_page_config(page_title="2026光伏信贷风控驾驶舱", layout="wide")

# --- 2. 自动定位文件 ---
current_folder = os.path.dirname(os.path.abspath(__file__))
xlsx_files = glob.glob(os.path.join(current_folder, "*.xlsx"))

if not xlsx_files:
    st.error("❌ 严重错误：找不到Excel文件！")
    st.stop()
file_path = xlsx_files[0]

# --- 3. 侧边栏：Sheet 选择 ---
st.sidebar.header("📂 1. 数据源选择")
try:
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names
    # 让用户选择
    selected_sheet = st.sidebar.selectbox("选择 Sheet:", sheet_names)
except Exception as e:
    st.error(f"Excel 读取失败: {e}")
    st.stop()

# --- 4. 数据加载 (绝对容错模式) ---
@st.cache_data
def load_data_safe(sheet_name):
    # 读取
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # ⚠️ 关键修复：强制保留原始行数，不做任何 dropna！
    
    # 1. 补全字符串列 (防止空值被过滤)
    str_cols = ["信贷评级", "公司名称", "股票代码"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', '未知').replace('NaN', '未知')
            
    # 2. 补全数值列 (防止图表报错)
    num_cols = ["技术壁垒(毛利率%)", "综合得分", "资产负债率(%)"]
    for col in num_cols:
        if col in df.columns:
            # 强制转数字，出错的变 NaN，然后填 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    return df

# 加载数据
df = load_data_safe(selected_sheet)

# --- 5. 侧边栏：筛选器 (默认全放行) ---
st.sidebar.header("🔍 2. 筛选条件")

# 5.1 评级筛选
if "信贷评级" in df.columns:
    all_ratings = sorted(list(df["信贷评级"].unique()))
    selected_rating = st.sidebar.multiselect(
        "信贷评级:",
        options=all_ratings,
        default=all_ratings # 默认全选
    )
else:
    st.error("❌ 缺少 '信贷评级' 列")
    st.stop()

# 5.2 毛利率筛选 (默认 0)
# 修改前：min_margin = st.sidebar.slider("最低毛利率:", 0, 60, 0)

# 👇 修改后：允许负数（最低 -50%），默认从 -50 开始，保证亏损企业也能显示
min_margin = st.sidebar.slider("最低毛利率要求 (%):", -80,80, -80)

# 5.3 执行筛选
filtered_df = df[
    (df["信贷评级"].isin(selected_rating)) & 
    (df["技术壁垒(毛利率%)"] >= min_margin)
]

# --- 6. 核心指标卡 ---
st.title("☀️ 2026 光伏行业信贷生存压力测试")

# 🔴 显眼包：直接把数字打在公屏上
st.info(f"📊 数据核对：Excel 原始读取 **{len(df)}** 行 | 筛选后显示 **{len(filtered_df)}** 行")

col1, col2, col3, col4 = st.columns(4)
col1.metric("监测企业总数", f"{len(filtered_df)} 家")
col2.metric("A类优质资产", f"{len(filtered_df[filtered_df['综合得分']>=80])} 家")
col3.metric("平均毛利率", f"{filtered_df['技术壁垒(毛利率%)'].mean():.1f}%")
col4.metric("平均负债率", f"{filtered_df['资产负债率(%)'].mean():.1f}%")

st.markdown("---")

# --- 7. 图表与数据 ---
tab1, tab2 = st.tabs(["📊 行业全景", "📋 完整数据表"])

with tab1:
    if not filtered_df.empty:
        fig = px.treemap(
            filtered_df,
            path=[px.Constant("全行业"), '信贷评级', '公司名称'],
            values='综合得分', # 只要这个不是0就能显示
            color='综合得分',
            color_continuous_scale='RdYlGn',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # 直接显示表格，这是最直观的证据
    st.dataframe(filtered_df, use_container_width=True)
    st.download_button("📥 下载数据", filtered_df.to_csv().encode('utf-8-sig'), "data.csv")


