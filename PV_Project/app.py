import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob

# --- 1. 页面配置 ---
st.set_page_config(page_title="2026光伏信贷风控驾驶舱", layout="wide")

# --- 2. 自动找文件 ---
current_folder = os.path.dirname(os.path.abspath(__file__))
xlsx_files = glob.glob(os.path.join(current_folder, "*.xlsx"))

if not xlsx_files:
    st.error("❌ 没找到Excel文件！")
    st.stop()
file_path = xlsx_files[0]

# --- 3. 强力加载与清洗 (Fixed) ---
@st.cache_data
def load_data_force():
    # 强制读取所有行，不忽略任何错误
    df = pd.read_excel(file_path)
    
    # 1. 强制保留所有行，哪怕全是空的
    original_count = len(df)
    
    # 2. 清洗列名 (去掉空格)
    df.columns = [str(c).strip() for c in df.columns]
    
    # 3. 处理毛利率 (最容易出错的地方)
    if "技术壁垒(毛利率%)" in df.columns:
        # 强制转换为数字，把无法转换的（比如"--"）变成 NaN
        df["技术壁垒(毛利率%)"] = pd.to_numeric(df["技术壁垒(毛利率%)"], errors='coerce')
        # 把 NaN 填补为 0 (这样就不会被过滤掉了！)
        df["技术壁垒(毛利率%)"] = df["技术壁垒(毛利率%)"].fillna(0)
    
    # 4. 处理评级
    if "信贷评级" in df.columns:
        df["信贷评级"] = df["信贷评级"].fillna("未分级").astype(str)
        
    return df, original_count

# 加载数据
df, raw_count = load_data_force()

# --- 4. 显眼包调试条 ---
st.success(f"📊 Excel 原始行数：{raw_count} 行 | 当前显示：{len(df)} 行")
if raw_count != 52:
    st.warning(f"⚠️ 注意：你的 Excel 里只有 {raw_count} 行数据，不是 52 行！请检查 Excel 文件内容。")

# --- 5. 侧边栏 ---
st.sidebar.header("🔍 筛选")
if "信贷评级" in df.columns:
    all_ratings = sorted(list(df["信贷评级"].unique()))
    selected = st.sidebar.multiselect("评级", all_ratings, default=all_ratings) # 默认全选
    
    # 筛选逻辑
    mask_rating = df["信贷评级"].isin(selected)
    filtered_df = df[mask_rating]
else:
    filtered_df = df

# --- 6. 展示数据表 (直接看这里有没有 52) ---
st.title("☀️ 光伏企业全量数据")
st.metric("当前显示数量", f"{len(filtered_df)} 家")

st.dataframe(filtered_df, use_container_width=True)
