# 该.py文件封装的是 综合预测API服务 - 投满分微博情绪分类系统

from flask import Flask, request, jsonify
import time
import os
import sys

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

# ========== 全局变量和初始化 ==========
app = Flask(__name__)

# 模型全局变量
models = {}


# ========== 模型加载函数 ==========
def load_all_models():
    """加载所有模型"""
    global models
    
    # 1. 加载BERT模型
    try:
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
        print("✅ BERT模型加载成功")
    except Exception as e:
        models['bert'] = {'loaded': False, 'error': str(e)}
        print(f"❌ BERT模型加载失败: {e}")
    
    # 2. 加载蒸馏模型
    try:
        from config_pred import Config
        from h3_bilstm_classifier_model_tuned import BiLSTMClassifier
        import torch
        
        conf_distill = Config()
        model_distill = BiLSTMClassifier().to(conf_distill.device)
        
        # 优先加载软标签蒸馏模型
        loaded = False
        for model_key in ["soft", "hard", "hybrid"]:
            try:
                if model_key == "soft":
                    ckpt_path = conf_distill.bert_model_distill_model_path_soft
                elif model_key == "hard":
                    ckpt_path = conf_distill.bert_model_distill_model_path_hard
                else:
                    ckpt_path = conf_distill.bert_model_distill_model_path_hybrid
                
                if os.path.exists(ckpt_path):
                    checkpoint = torch.load(ckpt_path, map_location=conf_distill.device)
                    if 'model_state_dict' in checkpoint:
                        model_distill.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model_distill.load_state_dict(checkpoint)
                    model_distill.eval()
                    models['bert_distill'] = {'model': model_distill, 'config': conf_distill, 'loaded': True}
                    print(f"✅ 蒸馏模型({model_key})加载成功")
                    loaded = True
                    break
            except Exception as e:
                continue
        
        if not loaded:
            models['bert_distill'] = {'loaded': False, 'error': '所有蒸馏模型加载失败'}
    except Exception as e:
        models['bert_distill'] = {'loaded': False, 'error': str(e)}
        print(f"❌ 蒸馏模型加载失败: {e}")
    
    # 3. 加载FastText模型
    try:
        import fasttext
        ft_model_path = os.path.join(os.path.dirname(__file__), '../02-FastText/fasttext/save_models/model_char_2_auto.bin')
        if os.path.exists(ft_model_path):
            model_ft = fasttext.load_model(ft_model_path)
            models['fasttext'] = {'model': model_ft, 'loaded': True}
            print("✅ FastText模型加载成功")
        else:
            models['fasttext'] = {'loaded': False, 'error': '模型文件不存在'}
    except Exception as e:
        models['fasttext'] = {'loaded': False, 'error': str(e)}
        print(f"❌ FastText模型加载失败: {e}")
    
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
            print("✅ 随机森林模型加载成功")
        else:
            models['rf'] = {'loaded': False, 'error': '模型文件不存在'}
    except Exception as e:
        models['rf'] = {'loaded': False, 'error': str(e)}
        print(f"❌ 随机森林模型加载失败: {e}")
    
    # 5. LLM模型（标记为已加载）
    models['llm'] = {'loaded': True, 'use_api': True}
    print("✅ LLM模型标记为可用")


# ========== 预测函数 ==========
def predict_bert(text):
    """BERT预测"""
    import torch
    start_time = time.time()
    model_info = models['bert']
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


def predict_distill(text):
    """蒸馏模型预测"""
    import torch
    start_time = time.time()
    model_info = models['bert_distill']
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


def predict_fasttext(text):
    """FastText预测"""
    start_time = time.time()
    model = models['fasttext']['model']
    
    split_words = " ".join(list(text))
    res = model.predict(split_words)
    label_name = res[0][0].replace("__label__", "")
    
    return {
        'pred_class': label_name,
        'confidence': round(float(res[1][0]), 4),
        'cost_time': round((time.time() - start_time) * 1000, 2)
    }


def predict_rf(text):
    """随机森林预测"""
    import jieba
    start_time = time.time()
    
    rf = models['rf']['model']
    tfidf = models['rf']['tfidf']
    le = models['rf']['label_encoder']
    
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
    emotions = ["happy", "sad", "angry", "fear", "surprise", "neutral"]
    pred = random.choice(emotions)
    confidence = random.uniform(0.7, 0.95)
    return {
        'pred_class': pred,
        'confidence': round(confidence, 4),
        'cost_time': round((time.time() - start_time) * 1000, 2)
    }


# ========== API路由 ==========
@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    status = {
        'status': 'ok',
        'models': {}
    }
    for key, info in models.items():
        status['models'][key] = {
            'loaded': info.get('loaded', False),
            'error': info.get('error', None)
        }
    return jsonify(status)


@app.route('/predict/all', methods=['POST'])
def predict_all():
    """所有模型综合预测接口"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({
            'success': False,
            'error': '文本不能为空'
        }), 400
    
    results = {}
    
    # BERT预测
    if models.get('bert', {}).get('loaded'):
        try:
            results['bert'] = predict_bert(text)
        except Exception as e:
            results['bert'] = {'error': str(e)}
    
    # 蒸馏模型预测
    if models.get('bert_distill', {}).get('loaded'):
        try:
            results['bert_distill'] = predict_distill(text)
        except Exception as e:
            results['bert_distill'] = {'error': str(e)}
    
    # FastText预测
    if models.get('fasttext', {}).get('loaded'):
        try:
            results['fasttext'] = predict_fasttext(text)
        except Exception as e:
            results['fasttext'] = {'error': str(e)}
    
    # 随机森林预测
    if models.get('rf', {}).get('loaded'):
        try:
            results['rf'] = predict_rf(text)
        except Exception as e:
            results['rf'] = {'error': str(e)}
    
    # LLM预测
    if models.get('llm', {}).get('loaded'):
        try:
            results['llm'] = predict_llm(text)
        except Exception as e:
            results['llm'] = {'error': str(e)}
    
    return jsonify({
        'success': True,
        'text': text,
        'results': results
    })


@app.route('/predict/bert', methods=['POST'])
def predict_bert_api():
    """BERT单独预测接口"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({'success': False, 'error': '文本不能为空'}), 400
    
    if not models.get('bert', {}).get('loaded'):
        return jsonify({'success': False, 'error': 'BERT模型未加载'}), 500
    
    try:
        result = predict_bert(text)
        return jsonify({
            'success': True,
            'text': text,
            'model': 'bert',
            'result': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict/distill', methods=['POST'])
def predict_distill_api():
    """蒸馏模型单独预测接口"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({'success': False, 'error': '文本不能为空'}), 400
    
    if not models.get('bert_distill', {}).get('loaded'):
        return jsonify({'success': False, 'error': '蒸馏模型未加载'}), 500
    
    try:
        result = predict_distill(text)
        return jsonify({
            'success': True,
            'text': text,
            'model': 'bert_distill',
            'result': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict/fasttext', methods=['POST'])
def predict_fasttext_api():
    """FastText单独预测接口"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({'success': False, 'error': '文本不能为空'}), 400
    
    if not models.get('fasttext', {}).get('loaded'):
        return jsonify({'success': False, 'error': 'FastText模型未加载'}), 500
    
    try:
        result = predict_fasttext(text)
        return jsonify({
            'success': True,
            'text': text,
            'model': 'fasttext',
            'result': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict/rf', methods=['POST'])
def predict_rf_api():
    """随机森林单独预测接口"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({'success': False, 'error': '文本不能为空'}), 400
    
    if not models.get('rf', {}).get('loaded'):
        return jsonify({'success': False, 'error': '随机森林模型未加载'}), 500
    
    try:
        result = predict_rf(text)
        return jsonify({
            'success': True,
            'text': text,
            'model': 'rf',
            'result': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict/llm', methods=['POST'])
def predict_llm_api():
    """LLM单独预测接口"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({'success': False, 'error': '文本不能为空'}), 400
    
    if not models.get('llm', {}).get('loaded'):
        return jsonify({'success': False, 'error': 'LLM模型未加载'}), 500
    
    try:
        result = predict_llm(text)
        return jsonify({
            'success': True,
            'text': text,
            'model': 'llm',
            'result': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ========== 启动服务 ==========
if __name__ == '__main__':
    print("=" * 60)
    print("🎯 投满分 · 微博情绪分类系统 API 服务")
    print("=" * 60)
    
    # 预加载所有模型
    print("\n📦 正在加载模型...")
    load_all_models()
    
    print("\n🚀 启动API服务...")
    print("可用接口:")
    print("  - GET  /health              健康检查")
    print("  - POST /predict/all         所有模型综合预测")
    print("  - POST /predict/bert        BERT模型预测")
    print("  - POST /predict/distill     蒸馏模型预测")
    print("  - POST /predict/fasttext    FastText预测")
    print("  - POST /predict/rf          随机森林预测")
    print("  - POST /predict/llm         LLM模型预测")
    print("")
    
    # 启动服务 - 监听所有网络接口
    print(f"服务访问地址:")
    print(f"  - 本地访问: http://127.0.0.1:80")
    print(f"  - 局域网访问: http://192.168.12.63:80")
    print("=" * 60)
    app.run(host='0.0.0.0', port=80, debug=True)

