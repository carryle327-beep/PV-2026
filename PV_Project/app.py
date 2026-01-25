import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- 1. 页面配置 ---
st.set_page_config(page_title="2026光伏信贷风控驾驶舱", layout="wide")

# --- 2. 万能路径加载法 (最关键的一步) ---
# 不管是在你的 Mac 上，还是在云端服务器上，这句话都能自动找到当前文件夹
current_folder = os.path.dirname(os.path.abspath(__file__))
# 拼接文件名 (确保你的 Excel 文件名和这个一模一样)
file_path = os.path.join(current_folder, "光伏全行业_完整信贷评级表.xlsx")

# --- 3. 读取数据函数 ---
@st.cache_data
def load_data():
    try:
        # 尝试读取
        df = pd.read_excel(file_path)
        return df
    except FileNotFoundError:
        st.error(f"❌ 找不到文件！程序试图读取的路径是：{file_path}")
        st.stop()
    except Exception as e:
        st.error(f"❌ 读取错误: {e}")
        st.stop()

# 加载数据
df = load_data()

# --- 4. 侧边栏：筛选器 ---
st.sidebar.header("🔍 筛选控制台")

# 4.1 评级筛选
if "信贷评级" in df.columns:
    all_ratings = list(df["信贷评级"].unique())
    selected_rating = st.sidebar.multiselect(
        "选择信贷评级:",
        options=all_ratings,
        default=all_ratings[:2] if len(all_ratings) > 1 else all_ratings
    )
else:
    st.error("Excel中缺少'信贷评级'列")
    st.stop()

# 4.2 毛利率筛选
min_margin = st.sidebar.slider("最低毛利率要求 (%):", 0, 60, 10)

# 4.3 执行筛选
filtered_df = df[
    (df["信贷评级"].isin(selected_rating)) & 
    (df["技术壁垒(毛利率%)"] >= min_margin)
]

# --- 5. 核心指标卡片 ---
st.title("☀️ 2026 光伏行业信贷生存压力测试")
st.markdown(f"**当前筛选**: {len(filtered_df)} 家企业 | **基准**: 2026 Q1 预测数据")

col1, col2, col3, col4 = st.columns(4)
col1.metric("监测企业总数", f"{len(filtered_df)} 家")
col2.metric("A类优质资产", f"{len(filtered_df[filtered_df['综合得分']>=80])} 家")
col3.metric("平均毛利率", f"{filtered_df['技术壁垒(毛利率%)'].mean():.1f}%")
col4.metric("平均负债率", f"{filtered_df['资产负债率(%)'].mean():.1f}%")

st.markdown("---")

# --- 6. 图表展示 ---
tab1, tab2, tab3 = st.tabs(["📊 行业全景", "🔬 风险矩阵", "📋 详细数据"])

with tab1:
    st.subheader("信贷评级分布 (TreeMap)")
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
    else:
        st.info("请调整筛选条件以查看数据")

with tab2:
    st.subheader("技术壁垒 vs 经营风险")
    if not filtered_df.empty:
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
        fig_bubble.add_vline(x=30, line_dash="dash", line_color="green", annotation_text="护城河")
        st.plotly_chart(fig_bubble, use_container_width=True)

with tab3:
    st.dataframe(filtered_df, use_container_width=True)
    # 下载按钮
    csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 下载筛选数据", csv, "report.csv", "text/csv")
