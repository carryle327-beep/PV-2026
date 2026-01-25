import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob

# --- 1. 页面配置 ---
st.set_page_config(page_title="2026光伏信贷风控驾驶舱", layout="wide")

# --- 2. 智能文件读取 ---
current_folder = os.path.dirname(os.path.abspath(__file__))
xlsx_files = glob.glob(os.path.join(current_folder, "*.xlsx"))

if not xlsx_files:
    st.error(f"❌ 找不到Excel文件！请确认文件在: {current_folder}")
    st.stop()
else:
    file_path = xlsx_files[0]

# --- 3. 数据加载与清洗 (关键修复步骤！) ---
@st.cache_data
def load_and_clean_data():
    try:
        # 读取原始数据
        df = pd.read_excel(file_path)
        
        # 🕵️‍♂️ 第一层防丢失：重命名列（防止Excel里有空格）
        # 自动去掉列名里的空格
        df.columns = [c.strip() for c in df.columns]
        
        # 🕵️‍♂️ 第二层防丢失：处理空值 (NaN)
        # 如果评级是空的，填上 "未分级"
        if "信贷评级" in df.columns:
            df["信贷评级"] = df["信贷评级"].fillna("未分级").astype(str)
        
        # 如果毛利率是空的，填上 0
        if "技术壁垒(毛利率%)" in df.columns:
            # 先把非数字的（比如"--"）强制转成NaN，再填0
            df["技术壁垒(毛利率%)"] = pd.to_numeric(df["技术壁垒(毛利率%)"], errors='coerce').fillna(0)
            
        return df
    except Exception as e:
        st.error(f"❌ 数据清洗失败: {e}")
        st.stop()

df = load_and_clean_data()

# --- 4. 调试信息 (让你看见真相) ---
# 这一行字会显示在网页最上面，告诉你到底读到了几行
st.success(f"✅ 成功读取 Excel 原始数据：共 {len(df)} 家企业 (目标 52 家)")

# --- 5. 侧边栏：筛选器 ---
st.sidebar.header("🔍 筛选控制台")

# 5.1 评级筛选 (默认全选)
if "信贷评级" in df.columns:
    all_ratings = sorted(list(df["信贷评级"].unique()))
    selected_rating = st.sidebar.multiselect(
        "选择信贷评级:",
        options=all_ratings,
        default=all_ratings # 默认全选！
    )
else:
    st.error("Excel中缺少'信贷评级'列")
    st.stop()

# 5.2 毛利率筛选 (默认从0开始)
min_margin = st.sidebar.slider("最低毛利率要求 (%):", 0, 60, 0) # 默认0

# 5.3 执行筛选
filtered_df = df[
    (df["信贷评级"].isin(selected_rating)) & 
    (df["技术壁垒(毛利率%)"] >= min_margin)
]

# --- 6. 核心指标 ---
st.title("☀️ 2026 光伏行业信贷生存压力测试")
st.markdown(f"**数据源**: {os.path.basename(file_path)}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("监测企业总数", f"{len(filtered_df)} 家", delta=f"原始 {len(df)} 家")
col2.metric("A类优质资产", f"{len(filtered_df[filtered_df['综合得分']>=80])} 家")
col3.metric("平均毛利率", f"{filtered_df['技术壁垒(毛利率%)'].mean():.1f}%")
col4.metric("平均负债率", f"{filtered_df['资产负债率(%)'].mean():.1f}%")

st.markdown("---")

# --- 7. 图表展示 ---
tab1, tab2, tab3 = st.tabs(["📊 行业全景", "🔬 风险矩阵", "📋 详细数据"])

with tab1:
    if not filtered_df.empty:
        fig = px.treemap(
            filtered_df,
            path=[px.Constant("全行业"), '信贷评级', '公司名称'],
            values='综合得分',
            color='综合得分',
            color_continuous_scale='RdYlGn',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    if not filtered_df.empty:
        fig_bubble = px.scatter(
            filtered_df,
            x="技术壁垒(毛利率%)",
            y="综合得分",
            size="综合得分",
            color="信贷评级",
            hover_name="公司名称",
            height=500
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

with tab3:
    st.dataframe(filtered_df, use_container_width=True)
