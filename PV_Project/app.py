import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob

# --- 1. 页面基础设置 ---
st.set_page_config(page_title="2026光伏信贷风控驾驶舱", layout="wide")

# --- 2. 自动定位文件 ---
current_folder = os.path.dirname(os.path.abspath(__file__))
xlsx_files = glob.glob(os.path.join(current_folder, "*.xlsx"))

if not xlsx_files:
    st.error("❌ 严重错误：找不到Excel文件！")
    st.stop()
file_path = xlsx_files[0]

# --- 3. 侧边栏：数据源设置 (关键！) ---
st.sidebar.header("📂 数据源设置")

# 读取 Excel 的所有 Sheet 名字
try:
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names
    # 让用户选择正确的 Sheet (就是你刚才找到 52 行的那个！)
    selected_sheet = st.sidebar.selectbox("选择包含数据的 Sheet:", sheet_names)
except Exception as e:
    st.error(f"Excel 读取失败: {e}")
    st.stop()

# --- 4. 数据加载与清洗 ---
@st.cache_data
def load_data(sheet_name):
    # 读取用户选中的 Sheet
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # --- 强力清洗逻辑 (防止数据丢失) ---
    # 1. 去掉列名空格
    df.columns = [str(c).strip() for c in df.columns]
    
    # 2. 补全评级 (空值 -> 未分级)
    if "信贷评级" in df.columns:
        df["信贷评级"] = df["信贷评级"].fillna("未分级").astype(str)
    
    # 3. 补全毛利率 (空值/横杠 -> 0)
    if "技术壁垒(毛利率%)" in df.columns:
        df["技术壁垒(毛利率%)"] = pd.to_numeric(df["技术壁垒(毛利率%)"], errors='coerce').fillna(0)
        
    return df

# 加载数据
df = load_data(selected_sheet)

# --- 5. 侧边栏：业务筛选 ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 业务筛选")

# 5.1 评级筛选 (默认全选)
if "信贷评级" in df.columns:
    all_ratings = sorted(list(df["信贷评级"].unique()))
    selected_rating = st.sidebar.multiselect(
        "选择信贷评级:",
        options=all_ratings,
        default=all_ratings # ✅ 默认全选，保证显示 52 家
    )
else:
    st.error("❌ 数据表中缺少 '信贷评级' 列，请检查 Excel！")
    st.stop()

# 5.2 毛利率筛选 (默认从 0 开始)
min_margin = st.sidebar.slider("最低毛利率要求 (%):", 0, 60, 0) # ✅ 默认 0

# 5.3 执行筛选
filtered_df = df[
    (df["信贷评级"].isin(selected_rating)) & 
    (df["技术壁垒(毛利率%)"] >= min_margin)
]

# --- 6. 驾驶舱核心指标 ---
st.title("☀️ 2026 光伏行业信贷生存压力测试")
st.markdown(f"**当前数据源**: `{os.path.basename(file_path)}` - `{selected_sheet}`")

# 顶部指标卡
col1, col2, col3, col4 = st.columns(4)
col1.metric("监测企业总数", f"{len(filtered_df)} 家", delta=f"原始 {len(df)} 家")

# A类资产计算
a_class_count = len(filtered_df[filtered_df['综合得分']>=80]) if '综合得分' in filtered_df.columns else 0
col2.metric("A类优质资产 (得分≥80)", f"{a_class_count} 家")

# 平均值计算
avg_margin = filtered_df['技术壁垒(毛利率%)'].mean()
col3.metric("平均毛利率", f"{avg_margin:.1f}%")

avg_debt = filtered_df['资产负债率(%)'].mean() if '资产负债率(%)' in filtered_df.columns else 0
col4.metric("平均负债率", f"{avg_debt:.1f}%", delta_color="inverse")

st.markdown("---")

# --- 7. 图表展示区 ---
tab1, tab2, tab3 = st.tabs(["📊 行业全景图", "🔬 风险矩阵", "📋 数据明细"])

with tab1:
    st.subheader("信贷评级分布 (TreeMap)")
    if not filtered_df.empty and '综合得分' in filtered_df.columns:
        fig_tree = px.treemap(
            filtered_df,
            path=[px.Constant("光伏全行业"), '信贷评级', '公司名称'],
            values='综合得分',
            color='综合得分',
            color_continuous_scale='RdYlGn',
            height=550
        )
        st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.info("数据不足或缺少'综合得分'列，无法生成热力图")

with tab2:
    st.subheader("技术壁垒 vs 经营风险")
    if not filtered_df.empty and '资产负债率(%)' in filtered_df.columns:
        fig_bubble = px.scatter(
            filtered_df,
            x="技术壁垒(毛利率%)",
            y="综合得分",
            size="综合得分",
            color="信贷评级",
            hover_name="公司名称",
            hover_data=["资产负债率(%)"],
            height=500
        )
        # 添加辅助线
        fig_bubble.add_vline(x=30, line_dash="dash", line_color="green", annotation_text="高壁垒区")
        st.plotly_chart(fig_bubble, use_container_width=True)

with tab3:
    st.subheader("筛选结果列表")
    st.dataframe(filtered_df, use_container_width=True)
    
    # CSV 下载
    csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 下载当前筛选结果", csv, "risk_report.csv", "text/csv")
