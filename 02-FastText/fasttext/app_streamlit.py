'''import streamlit as st
import requests
import time


st.write("欢迎使用微博情绪预测系统")
user_input = st.text_input("请输入文章内容:")
button = st.button("预测")
if button:
    start_time = time.time()
    url = "http://192.168.12.116:5000/NewCls_handler"
    re = requests.post(url, json={"text":user_input})

    st.write('预测的类别是：',re.json()["pred_class"])
    print(re.json())
    end_time = time.time()
    st.write(f'预测消耗时间{(end_time - start_time)*1000} ms')
'''


import streamlit as st
import requests
import time

# 页面全局基础配置
st.set_page_config(
    page_title="微博情绪预测系统",
    page_icon="🪁",
    layout="centered"
)

# 全套国风文艺CSS，修复右侧文字挤压问题
st.markdown("""
<style>
/* 全局页面底色 温柔米杏 */
.stApp {
    background-color: #f9f6ef;
}
/* 主标题样式 宋体文艺字、宽字间距 */
.title-main {
    font-family: "SimSun", serif;
    font-size: 46px;
    color: #363430;
    letter-spacing: 6px;
    text-align: center;
    margin: 20px 0 10px;
}
/* 副标题小字 */
.title-sub {
    font-family: "SimSun", serif;
    font-size: 15px;
    color: #8a847b;
    text-align: center;
    letter-spacing: 2px;
    margin-bottom: 30px;
}
/* 通用卡片容器 柔和阴影圆角 */
div[data-testid="stContainer"] {
    border-radius: 20px !important;
    box-shadow: 0 6px 20px rgba(170, 155, 140, 0.15) !important;
    border: 1px solid #e4dfd7 !important;
    padding: 18px 22px !important;
    background-color: #fcfaf5 !important;
}
/* 右侧侧边卡片单独加宽内边距，防止文字贴边 */
div[data-testid="stVerticalBlock"] > div:nth-child(2) div[data-testid="stContainer"] {
    padding: 22px 20px !important;
}
/* 输入文本框美化 */
textarea {
    border-radius: 14px !important;
    border: 1px solid #d8cdc0 !important;
    background-color: #f7f3ec !important;
    font-family: "SimSun", serif;
    font-size: 16px;
}
/* 主按钮豆沙红渐变 */
button[kind="primary"] {
    background: linear-gradient(135deg, #d17878, #c96565) !important;
    border: none !important;
    border-radius: 14px !important;
    height: 52px;
    font-size: 18px;
    font-family: "SimSun", serif;
    letter-spacing: 3px;
}
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #c76868, #b85555) !important;
}
/* 分割线浅棕柔和 */
hr {
    border-color: #e0d9cf !important;
    margin: 30px 0;
}
/* 提示文字、底部注释 */
.bottom-text {
    color: #7c766e;
    font-size: 13px;
    letter-spacing: 1px;
    text-align: center;
    margin-top: 10px;
}
/* 结果大指标美化 */
[data-testid="stMetricValue"] {
    font-size: 32px !important;
    font-family: "SimSun", serif;
    color: #b85555;
}
/* 侧边说明模块，修复文字拥挤 */
.info-card {
    background-color: #f3eee5;
    border-radius: 16px;
    padding: 24px 20px;
    border-left: 4px solid #d17878;
    line-height: 1.85 !important;
}
/* 右侧列表行高加宽，避免文字堆叠 */
.info-card li {
    margin: 10px 0 !important;
}
/* 右侧示例按钮间距 */
.right-btn button {
    margin: 6px 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ========== 页面头部 丰富标题区 ==========
st.markdown('<p class="title-main">🪁 微博情绪预测系统</p>', unsafe_allow_html=True)
st.markdown('<p class="title-sub">以文字为舟，打捞藏在短句里的万千心绪</p>', unsafe_allow_html=True)
st.divider()

# ========== 分栏比例调整 [2.8,1] 左侧更宽，右侧收窄适配文字 ==========
col_left, col_right = st.columns([2.8, 1], gap="large")

with col_left:
    # 文本输入主卡片
    with st.container(border=True):
        st.subheader("✉️ 输入待解析微博文本")
        st.write("")
        user_input = st.text_area(
            label="文本内容",
            placeholder="将细碎心绪写在这里，如：晚风温柔，今日诸事顺心 / 世事难料，难免心生烦闷",
            height=180,
            label_visibility="visible"
        )
        st.write("")
        predict_btn = st.button("开始解析情绪", type="primary", use_container_width=True)

        # 预测逻辑
        if predict_btn:
            if not user_input.strip():
                st.warning("🍂 请填入一段文字，再进行情绪解析")
            else:
                start_time = time.time()
                api_url = "http://192.168.12.116:5000/NewCls_handler"
                with st.spinner("正在细读文字里的情绪碎片……"):
                    try:
                        resp = requests.post(url=api_url, json={"text": user_input}, timeout=8)
                        resp.raise_for_status()
                        res_data = resp.json()
                        end_time = time.time()
                        cost_ms = round((end_time - start_time) * 1000, 2)

                        st.success("🍁 情绪解析完成")
                        # 结果展示独立卡片
                        res_box = st.container(border=True)
                        with res_box:
                            st.metric(label="文字蕴藏情绪", value=res_data["pred_class"])
                            st.caption(f"文字解读耗时：{cost_ms} 毫秒")

                    except requests.exceptions.ConnectionError:
                        st.error("❌ 未能连接情绪解析服务，请检查后端接口是否启动")
                    except requests.exceptions.Timeout:
                        st.error("⏳ 解析超时，服务响应缓慢，请稍后重试")
                    except Exception as e:
                        st.error(f"解析异常：{str(e)}")

with col_right:
    # 右侧信息卡片，加宽内边距、行高解决文字挤压
    st.markdown("""
    <div class="info-card">
    <h4 style="font-family:SimSun;color:#363430;margin-top:0;margin-bottom:12px;">📜 系统说明</h4>
    <p style="color:#666;font-size:14px;line-height:1.85;margin:0 0 20px;">
    本系统基于FastText文本分类模型训练，针对微博日常短句、网络随笔进行情感识别，可自动判别文本背后的情绪倾向。
    </p>
    <h4 style="font-family:SimSun;color:#363430;margin-bottom:12px;">✨ 使用小贴士</h4>
    <ul style="color:#666;font-size:13px;line-height:1.8;padding-left:18px;margin:0;">
        <li>输入单句、长段微博均可识别</li>
        <li>包含表情、网络词汇不影响识别效果</li>
        <li>完整短句，情绪判断结果更精准</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # 示例按钮单独容器，增加上下间距，不和说明文字挤在一起
    st.write("<div class='right-btn'></div>", unsafe_allow_html=True)
    st.subheader("📝 示例文案一键填充")
    sample1 = st.button("示例1：愉悦文案", use_container_width=True)
    sample2 = st.button("示例2：低落文案", use_container_width=True)
    sample3 = st.button("示例3：愤怒吐槽", use_container_width=True)

    # 示例填充逻辑
    if sample1:
        st.session_state["fill_text"] = "今天出门遇见晚霞，晚风温柔，一整天都觉得舒心自在"
    if sample2:
        st.session_state["fill_text"] = "诸事不顺，期待的事情全部落空，心里闷闷的提不起劲"
    if sample3:
        st.session_state["fill_text"] = "办事遇到蛮不讲理的人，沟通完全无效，实在让人恼火"

    # 回填到输入框
    if "fill_text" in st.session_state:
        user_input = st.session_state["fill_text"]

# 底部分割线+页脚文案
st.divider()
st.markdown(
    '<p class="bottom-text">基于FastText文本分类模型 | 捕捉短句里藏着的喜怒悲欢 · 国风文艺情感解析工具</p>',
    unsafe_allow_html=True
)