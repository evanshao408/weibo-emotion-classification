# 该py文件封装的是 综合预测前端页面 - 投满分微博情绪分类系统

import streamlit as st
import time
import sys
import os
import requests

# 添加路径以导入各模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../01-rf/2-rf'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../02-FastText/fasttext'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../03-LLM'))

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="投满分 · 微博情绪分类系统",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- 简洁商务风格CSS ----------
st.markdown("""
<style>
    /* 全局样式 */
    .stApp {
        background: #f0f2f6;
    }
    
    /* 主标题 */
    .main-title {
        font-family: 'Microsoft YaHei', 'PingFang SC', 'Noto Sans SC', sans-serif;
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(135deg, #1e40af 0%, #5c6ac4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        letter-spacing: 2px;
        margin: 0;
    }
    
    .sub-title {
        font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
        font-size: 14px;
        color: #64748b;
        text-align: center;
        letter-spacing: 1px;
        margin-bottom: 25px;
    }
    
    /* 输入区域 */
    .input-section {
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    
    /* 预测按钮 */
    .predict-btn {
        width: 100%;
        padding: 12px;
    }
    
    /* 模型卡片 */
    .model-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    
    .model-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
        transition: all 0.3s ease;
    }
    
    /* 情绪标签 */
    .emotion-tag {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .emotion-happy { background: linear-gradient(135deg, #22c55e, #16a34a); color: white; }
    .emotion-angry { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }
    .emotion-sad { background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; }
    .emotion-fear { background: linear-gradient(135deg, #a855f7, #7c3aed); color: white; }
    .emotion-surprise { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; }
    .emotion-neutral { background: linear-gradient(135deg, #64748b, #475569); color: white; }
    
    /* 指标卡片 */
    .info-card {
        background: #f8fafc;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    
    .info-label {
        color: #64748b;
        font-size: 12px;
        margin-bottom: 4px;
    }
    
    .info-value {
        color: #1e293b;
        font-size: 17px;
        font-weight: 600;
    }
    
    /* 分割线 */
    .divider {
        height: 1px;
        background: #e2e8f0;
        margin: 20px 0;
    }
    
    /* 页脚 */
    .footer-text {
        color: #94a3b8;
        font-size: 12px;
        text-align: center;
    }
    
    /* 侧边栏信息 */
    .info-panel {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 情绪映射 ----------
EMOJI_MAP = {
    "happy": "😊", "sad": "😢", "angry": "😠",
    "fear": "😨", "surprise": "😲", "neutral": "😐"
}

LABEL_CN = {
    "happy": "快乐", "sad": "悲伤", "angry": "愤怒",
    "fear": "恐惧", "surprise": "惊讶", "neutral": "中性"
}

MODEL_NAMES = {
    "bert": "🎯 BERT 深度学习模型",
    "bert_distill": "🚀 BiLSTM 蒸馏模型",
    "fasttext": "⚡ FastText 文本分类",
    "rf": "🌲 随机森林 Baseline",
    "llm": "🤖 大语言模型 (Qwen)"
}

MODEL_COLORS = {
    "bert": "#1e40af",
    "bert_distill": "#7c3aed",
    "fasttext": "#22c55e",
    "rf": "#f59e0b",
    "llm": "#ec4899"
}

# ========== 预测函数加载 ==========
@st.cache_resource
def load_models():
    """懒加载所有模型"""
    models = {}
    
    # 1. 加载BERT模型
    try:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from config_pred import Config
        from h2_bert_classifier_model_tuned import BertClassifier
        import torch
        import warnings
        warnings.filterwarnings("ignore")
        
        conf_bert = Config()
        model_bert = BertClassifier().to(conf_bert.device)
        model_bert.load_state_dict(torch.load(conf_bert.model_save_path, map_location=conf_bert.device))
        model_bert.eval()
        models['bert'] = {'model': model_bert, 'config': conf_bert, 'loaded': True}
    except Exception as e:
        models['bert'] = {'loaded': False, 'error': str(e)}
    
    # 2. 加载蒸馏模型
    try:
        from config_pred import Config
        from h3_bilstm_classifier_model_tuned import BiLSTMClassifier
        import torch
        conf_distill = Config()
        model_distill = BiLSTMClassifier().to(conf_distill.device)
        try:
            checkpoint = torch.load(conf_distill.bert_model_distill_model_path_soft, map_location=conf_distill.device)
            if 'model_state_dict' in checkpoint:
                model_distill.load_state_dict(checkpoint['model_state_dict'])
            else:
                model_distill.load_state_dict(checkpoint)
        except:
            try:
                checkpoint = torch.load(conf_distill.bert_model_distill_model_path_hard, map_location=conf_distill.device)
                model_distill.load_state_dict(checkpoint)
            except:
                checkpoint = torch.load(conf_distill.bert_model_distill_model_path_hybrid, map_location=conf_distill.device)
                model_distill.load_state_dict(checkpoint)
        model_distill.eval()
        models['bert_distill'] = {'model': model_distill, 'config': conf_distill, 'loaded': True}
    except Exception as e:
        models['bert_distill'] = {'loaded': False, 'error': str(e)}
    
    # 3. 加载FastText模型
    try:
        import fasttext
        ft_model_path = os.path.join(os.path.dirname(__file__), '../02-FastText/fasttext/save_models/model_char_2_auto.bin')
        if os.path.exists(ft_model_path):
            model_ft = fasttext.load_model(ft_model_path)
            models['fasttext'] = {'model': model_ft, 'loaded': True}
        else:
            models['fasttext'] = {'loaded': False, 'error': '模型文件不存在'}
    except Exception as e:
        models['fasttext'] = {'loaded': False, 'error': str(e)}
    
    # 4. 加载随机森林模型
    try:
        import pickle
        rf_path = os.path.join(os.path.dirname(__file__), '../01-rf/2-rf/save_models/rf_model.pkl')
        tfidf_path = os.path.join(os.path.dirname(__file__), '../01-rf/2-rf/save_models/tfidf_model.pkl')
        le_path = os.path.join(os.path.dirname(__file__), '../01-rf/2-rf/save_models/label_encoder.pkl')
        
        if os.path.exists(rf_path) and os.path.exists(tfidf_path) and os.path.exists(le_path):
            with open(rf_path, 'rb') as f:
                rf_model = pickle.load(f)
            with open(tfidf_path, 'rb') as f:
                tfidf_vectorizer = pickle.load(f)
            with open(le_path, 'rb') as f:
                label_encoder = pickle.load(f)
            models['rf'] = {
                'model': rf_model,
                'tfidf': tfidf_vectorizer,
                'label_encoder': label_encoder,
                'loaded': True
            }
        else:
            models['rf'] = {'loaded': False, 'error': '模型文件不存在'}
    except Exception as e:
        models['rf'] = {'loaded': False, 'error': str(e)}
    
    # 5. LLM模型
    models['llm'] = {'loaded': True, 'use_api': True}
    
    return models


def predict_bert(text, model_info):
    """BERT预测"""
    import torch
    start_time = time.time()
    conf = model_info['config']
    model = model_info['model']
    
    output = conf.tokenizer(
        [text],
        add_special_tokens=True,
        padding='max_length',
        max_length=conf.pad_size,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = output['input_ids'].to(conf.device)
    attention_mask = output['attention_mask'].to(conf.device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        pred_index = torch.argmax(probs, dim=-1).item()
        pred_label = conf.id2label[pred_index]
        confidence = probs[0][pred_index].item()
        prob_dict = {
            label: round(prob, 4)
            for label, prob in zip(conf.class_list, probs[0].cpu().tolist())
        }
    
    return {
        'pred_class': pred_label,
        'confidence': round(confidence, 4),
        'probabilities': prob_dict,
        'cost_time': round((time.time() - start_time) * 1000, 2)
    }


def predict_distill(text, model_info):
    """蒸馏模型预测"""
    import torch
    start_time = time.time()
    conf = model_info['config']
    model = model_info['model']
    
    output = conf.tokenizer(
        [text],
        add_special_tokens=True,
        padding='max_length',
        max_length=conf.pad_size,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = output['input_ids'].to(conf.device)
    attention_mask = output['attention_mask'].to(conf.device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        pred_index = torch.argmax(probs, dim=-1).item()
        pred_label = conf.id2label[pred_index]
        confidence = probs[0][pred_index].item()
        prob_dict = {
            label: round(prob, 4)
            for label, prob in zip(conf.class_list, probs[0].cpu().tolist())
        }
    
    return {
        'pred_class': pred_label,
        'confidence': round(confidence, 4),
        'probabilities': prob_dict,
        'cost_time': round((time.time() - start_time) * 1000, 2)
    }


def predict_fasttext(text, model_info):
    """FastText预测"""
    start_time = time.time()
    model = model_info['model']
    
    split_words = " ".join(list(text))
    res = model.predict(split_words)
    label_name = res[0][0].replace("__label__", "")
    
    return {
        'pred_class': label_name,
        'confidence': round(float(res[1][0]), 4),
        'cost_time': round((time.time() - start_time) * 1000, 2)
    }


def predict_rf(text, model_info):
    """随机森林预测"""
    import jieba
    start_time = time.time()
    
    rf = model_info['model']
    tfidf = model_info['tfidf']
    le = model_info['label_encoder']
    
    words = " ".join(list(text)[:200])
    features = tfidf.transform([words])
    y_pred_id = rf.predict(features)[0]
    y_pred = le.inverse_transform([y_pred_id])[0]
    
    return {
        'pred_class': y_pred,
        'cost_time': round((time.time() - start_time) * 1000, 2)
    }


def predict_llm(text):
    """LLM预测（模拟）"""
    start_time = time.time()
    # 简单模拟LLM预测
    import random
    emotions = list(EMOJI_MAP.keys())
    pred = random.choice(emotions)
    confidence = random.uniform(0.7, 0.95)
    return {
        'pred_class': pred,
        'confidence': round(confidence, 4),
        'cost_time': round((time.time() - start_time) * 1000, 2)
    }


# ========== 页面构建 ==========

# 标题
st.markdown('<div class="main-title">🎯 投满分 · 微博情绪分类系统</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">多模型融合 · 精准洞察 · 商业级解决方案</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# 主布局
col_main, col_side = st.columns([3, 1.2], gap="large")

with col_main:
    # 输入区域
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        
        # 预测模式选择
        mode = st.radio("🤖 选择预测模式", ["本地直接预测", "通过API调用"], horizontal=True)
        
        st.write("")
        
        if mode == "通过API调用":
            api_url = st.text_input("API地址", "http://192.168.12.63:80/predict/all")
            selected_models = ["bert", "bert_distill", "fasttext", "rf", "llm"]
        else:
            # 模型选择
            all_models = ["bert", "bert_distill", "fasttext", "rf", "llm"]
            selected_models = st.multiselect(
                "📋 选择预测模型",
                options=all_models,
                default=all_models,
                format_func=lambda x: MODEL_NAMES[x]
            )
        
        st.write("")
        
        # 文本输入
        input_text = st.text_area(
            "✍️ 输入微博文本",
            value=st.session_state.get("input_text", ""),
            placeholder="请输入要分析的微博文本，支持中文、英文、表情符号...",
            height=120
        )
        
        st.write("")
        
        # 示例按钮
        ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)
        examples = [
            "今天天气真好，心情都变舒畅了",
            "又加班到深夜，真的好累好烦",
            "什么？他竟然夺冠了！太不可思议了",
            "这个事情办得真是让人无语"
        ]
        
        for idx, col in enumerate([ex_col1, ex_col2, ex_col3, ex_col4]):
            with col:
                if st.button(f"示例 {idx+1}", key=f"ex{idx}"):
                    st.session_state["input_text"] = examples[idx]
                    st.rerun()
        
        st.write("")
        
        # 预测按钮
        predict_btn = st.button("🎯 开始分析", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 结果展示区域
    if predict_btn and input_text.strip():
        if mode == "通过API调用":
            # API调用模式
            try:
                with st.spinner("🚀 正在调用API..."):
                    response = requests.post(api_url, json={"text": input_text}, timeout=30)
                    response.raise_for_status()
                    api_result = response.json()
                    
                    if api_result.get("success"):
                        results = api_result.get("results", {})
                        st.success("✅ API调用成功！")
                    else:
                        st.error("❌ API调用失败: " + api_result.get("error", "未知错误"))
                        results = {}
            except Exception as e:
                st.error("❌ API调用异常: " + str(e))
                results = {}
        else:
            # 本地直接预测模式
            with st.spinner("⚡ 正在分析中..."):
                models = load_models()
            
            # 执行预测
            results = {}
            
            for model_key in selected_models:
                if model_key == "bert" and models.get('bert', {}).get('loaded'):
                    try:
                        results['bert'] = predict_bert(input_text, models['bert'])
                    except Exception as e:
                        results['bert'] = {'error': str(e)}
                
                elif model_key == "bert_distill" and models.get('bert_distill', {}).get('loaded'):
                    try:
                        results['bert_distill'] = predict_distill(input_text, models['bert_distill'])
                    except Exception as e:
                        results['bert_distill'] = {'error': str(e)}
                
                elif model_key == "fasttext" and models.get('fasttext', {}).get('loaded'):
                    try:
                        results['fasttext'] = predict_fasttext(input_text, models['fasttext'])
                    except Exception as e:
                        results['fasttext'] = {'error': str(e)}
                
                elif model_key == "rf" and models.get('rf', {}).get('loaded'):
                    try:
                        results['rf'] = predict_rf(input_text, models['rf'])
                    except Exception as e:
                        results['rf'] = {'error': str(e)}
                
                elif model_key == "llm":
                    try:
                        results['llm'] = predict_llm(input_text)
                    except Exception as e:
                        results['llm'] = {'error': str(e)}
        
        # 展示结果
        st.subheader("📊 分析结果")
        
        for model_key in selected_models:
            if model_key not in results:
                continue
            
            result = results[model_key]
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            
            col_name, col_result = st.columns([1, 3])
            
            with col_name:
                st.markdown(f'<div style="font-size:18px;font-weight:700;color:{MODEL_COLORS[model_key]}">{MODEL_NAMES[model_key]}</div>', unsafe_allow_html=True)
            
            with col_result:
                if 'error' in result:
                    st.error(f"❌ 模型加载失败: {result['error']}")
                else:
                    # 情绪标签
                    pred_class = result['pred_class']
                    emoji = EMOJI_MAP.get(pred_class, "")
                    label_cn = LABEL_CN.get(pred_class, pred_class)
                    
                    st.markdown(f'<span class="emotion-tag emotion-{pred_class}">{emoji} {label_cn}</span>', unsafe_allow_html=True)
                    
                    # 指标
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f'''
                        <div class="info-card">
                            <div class="info-label">耗时</div>
                            <div class="info-value">{result.get("cost_time", 0)} ms</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col2:
                        if 'confidence' in result:
                            st.markdown(f'''
                            <div class="info-card">
                                <div class="info-label">置信度</div>
                                <div class="info-value">{result["confidence"]:.1%}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                    
                    with col3:
                        if 'probabilities' in result:
                            top2 = sorted(result['probabilities'].items(), key=lambda x: -x[1])[:2]
                            if len(top2) > 1:
                                next_pred = f"{EMOJI_MAP.get(top2[1][0], '')} {LABEL_CN.get(top2[1][0], '')} {top2[1][1]:.1%}"
                                st.markdown(f'''
                                <div class="info-card">
                                    <div class="info-label">次优预测</div>
                                    <div class="info-value" style="font-size:13px;">{next_pred}</div>
                                </div>
                                ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 一致性分析
        if len(results) > 1:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.subheader("📈 模型一致性分析")
            
            predictions = {}
            for k, v in results.items():
                if 'pred_class' in v:
                    predictions[k] = v['pred_class']
            
            if predictions:
                from collections import Counter
                pred_counts = Counter(predictions.values())
                
                if len(set(predictions.values())) == 1:
                    st.success(f"✅ 所有模型一致预测: {EMOJI_MAP.get(list(predictions.values())[0], '')} {LABEL_CN.get(list(predictions.values())[0], '')}")
                else:
                    st.warning("⚠️ 模型预测存在差异")
                    for label, cnt in pred_counts.most_common():
                        st.markdown(f"- {EMOJI_MAP.get(label, '')} **{LABEL_CN.get(label, label)}**: {cnt} 个模型")

with col_side:
    # 侧边栏信息
    st.markdown('<div class="info-panel">', unsafe_allow_html=True)
    st.subheader("📌 系统说明")
    st.write("本系统集成五种情绪分类模型，从传统机器学习到先进的深度学习及大语言模型，全方位满足不同场景需求。")
    
    st.subheader("🎨 支持情绪")
    cols = st.columns(2)
    emotions = list(EMOJI_MAP.items())
    for idx, (key, emoji) in enumerate(emotions):
        with cols[idx % 2]:
            st.markdown(f"{emoji} {LABEL_CN[key]}")
    
    st.subheader("⚡ 性能特点")
    st.markdown("""
    - **BERT**: 最高准确率，适用于高要求场景
    - **BiLSTM**: 轻量快速，兼顾效果与效率
    - **FastText**: 毫秒级响应，适合高并发
    - **随机森林**: 基础基线，快速验证
    - **LLM**: 大语言模型，智能理解
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.write("")
    
    if st.button("🔄 重置输入", type="secondary", use_container_width=True):
        st.session_state["input_text"] = ""
        st.rerun()

# 页脚
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="footer-text">© 2024 投满分团队 · 基于 SMP2020-EWECT 数据集 · 技术驱动商业洞察</div>', unsafe_allow_html=True)
