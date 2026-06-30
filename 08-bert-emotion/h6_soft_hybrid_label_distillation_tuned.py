# 该.py文件的作用 -> 模型蒸馏之 -> 软硬标签混合蒸馏（优化版）

import torch  # 深度学习框架, 提供张量计算, 神经网络构建等...
import torch.nn as nn  # 神经网络模块, 损失函数, 网络层
from torch.optim import AdamW  # 优化器, 适用于Transformer类模型的优化器, 缓解梯度消失问题.
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F  # 函数模块, 激活函数, 损失函数等.

# 用于评估模型性能的库
from tqdm import tqdm  # 进度条
from config_tuned import Config  # 配置文件类
from model2dev_utils_tuned import model2dev  # 导入自定义验证函数(例如: 精确率, 召回率...)
from h1_dataloader_utils_tuned import build_dataloader  # 获取数据集加载器

from h2_bert_classifier_model_tuned import BertClassifier  # 教师模型, 导入BERT分类模型
from h3_bilstm_classifier_model_tuned import BiLSTMClassifier  # 学生模型, 导入Bilstm分类模型

# 忽略的警告信息
import warnings

warnings.filterwarnings("ignore")

# todo 1. 加载配置对象，包含模型参数、路径等
conf = Config()


# todo 2. 定义模型训练函数, 封装完整的训练流程(数据加载, 模型训练, 验证, 保存)
def model2train():
    # 1. 准备训练/验证/测试数据, 获取其对应的 数据集加载器.
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()

    # 2. 初始化并配置模型.
    # 2.1 实例化教师模型.
    # 创建教师对象.
    teacher_model = BertClassifier().to(conf.device)
    # 加载教师模型的预训练参数.
    teacher_model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device))

    # 2.2 创建学生模型.
    student_model = BiLSTMClassifier().to(conf.device)

    # 3. 定义损失函数, 使用: 交叉熵损失.
    criterion = nn.CrossEntropyLoss()

    # 4. 定义优化器, 使用: AdamW 优化器.
    optimizer = AdamW(student_model.parameters(), lr=conf.lstm_learning_rate, weight_decay=1e-4)

    # 5. 定义学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=conf.student_epochs, eta_min=1e-5)

    # 6. 初始化最优F1分数, 用于筛选性能最好的模型(即: 初始值为0, 后续更新)
    best_macro_f1 = 0.0

    # 7. 具体的训练过程, 外层循环表示训练的轮数. 每轮都要遍历所有训练数据.
    for epoch in range(conf.student_epochs):
        # 7.1 设置教师模型为评估模式, 学生模型为训练模式.
        teacher_model.eval()
        student_model.train()

        # 7.2 初始化训练过程中的统计变量..total_loss(损失值), batch_count(迭代次数),
        total_loss = 0.0  # 累计当前批次的损失值.
        batch_count = 0

        # todo 2. 动态调整温度和软标签权重
        # 随着训练进行，逐渐增加硬标签权重，减少软标签权重
        progress = epoch / conf.student_epochs
        T = conf.distill_temperature * (1 - 0.5 * progress)  # 温度从2.0逐渐降到1.0
        alpha = conf.distill_alpha * (1 - 0.6 * progress)  # alpha从0.5逐渐降到0.2
        beta = 1.0 - alpha  # 硬标签权重从0.5逐渐升到0.8

        print(f"\n第 {epoch + 1}/{conf.student_epochs} 轮训练, T={T:.2f}, alpha={alpha:.2f}, beta={beta:.2f}")

        # 7.3 内层循环, 遍历训练集, 获取到每个批次, 逐批次更新模型.
        for i, batch in enumerate(tqdm(train_dataloader, desc=f'混合蒸馏第{epoch + 1}轮训练')):
            # 7.3.1 提取批次数据并移动到指定设备
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(conf.device)  # token id序列(模型输入)
            attention_mask = attention_mask.to(conf.device)  # 表示哪些token有效, 哪些无效
            labels = labels.to(conf.device)  # 标签(模型输出)

            # ----------------------------- 学生模型前向传播 -----------------------------
            # 7.3.2 前向传播：模型预测
            student_logits = student_model(input_ids,
                                           attention_mask=attention_mask)  # logits: 未归一化的分类得分, 形状: [batch_size, num_classes]

            # ----------------------------- 老师模型前向传播 -----------------------------
            # 7.3.3 教师模型预测值与预测标签, 禁用梯度计算以提高效率并减少内存占用
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids, attention_mask=attention_mask)
                teacher_labels = torch.argmax(teacher_logits, dim=1)

            # todo 7.3.4 计算真实标签损失（使用原始标签）
            real_label_loss = criterion(student_logits, labels)

            # todo 7.3.5 计算教师硬标签损失
            teacher_hard_loss = criterion(student_logits, teacher_labels)

            # todo 7.3.6 计算软标签损失: KL散度损失, 衡量学生和教师的平滑概率分布的差异.
            teacher_log_soft_labels = F.log_softmax(teacher_logits / T, dim=-1)
            student_log_soft_labels = F.log_softmax(student_logits / T, dim=-1)

            # todo 7.3.7 具体的计算KL散度的动作.标准知识蒸馏损失中，软标签部分要乘以T**2, 确保软标签损失与硬标签损失在梯度量级上可比
            soft_loss = F.kl_div(
                student_log_soft_labels,  # 学生的平滑对数概率(预测分布)
                teacher_log_soft_labels,  # 教师的平滑对数概率(真实分布)
                reduction='batchmean',  # 批次平均
                log_target=True  # 是否使用对数概率作为目标
            ) * (T * T)

            # todo 7.3.8 混合损失：软标签 + 教师硬标签 + 真实标签
            # 权重分配：软标签(alpha) + 教师硬标签(beta * 0.5) + 真实标签(beta * 0.5)
            loss = (alpha * soft_loss + 
                    beta * 0.6 * teacher_hard_loss + 
                    beta * 0.4 * real_label_loss)

            # 7.3.9 累计损失值. 累计迭代次数
            total_loss += loss.item()
            batch_count += 1
            # 7.3.10 梯度清零, 保证参数更新准确
            optimizer.zero_grad()
            # 7.3.11 反向传播：计算梯度, 链式求导.
            loss.backward()
            # 7.3.12 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            # 7.3.13 参数更新(梯度更新), 基于梯度和优化器规则(AdamW)更新模型权重
            optimizer.step()

            # 7.4 每100个批次或者轮次末尾, 验证模型效果(在验证集评估模型并保存最优模型)
            if i % 100 == 0 or i == len(train_dataloader) - 1:
                # 7.4.1 计算平均损失
                avg_loss = total_loss / batch_count  # 总损失/批次数=平均损失
                # 7.4.2 调用验证函数, 计算验证集的评估报告.
                report, macro_f1, accuracy, precision, recall = model2dev(student_model, dev_dataloader, conf.device)
                # 7.4.3 打印验证集日志, 查看详细评估报告.
                print(f"\nepoch:{epoch + 1}轮次, 批次:{i + 1}, 平均损失是:{avg_loss:.4f}")
                print(f"验证集评估报告:\n{report}")
                print(f"Macro-F1: {macro_f1:.4f}, 准确度: {accuracy:.4f}, Macro-Precision: {precision:.4f}, Macro-Recall: {recall:.4f}")
                # 7.4.4 重置模型为训练模式.
                student_model.train()
                # 7.5保存模型. 如果 f1值高于历史最优分数, 则保存模型
                if macro_f1 > best_macro_f1:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': student_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_macro_f1': macro_f1,
                        'accuracy': accuracy
                    }, conf.bert_model_distill_model_path_hybrid)
                    print(f"混合蒸馏模型已保存到: {conf.bert_model_distill_model_path_hybrid}")
                    best_macro_f1 = macro_f1

        # 7.6 每个epoch结束后更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"当前学习率: {current_lr:.8f}")

    print(f"\n混合蒸馏训练完成, 最佳Macro-F1: {best_macro_f1:.4f}")


# todo 3. 主程序入口.
if __name__ == '__main__':
    model2train()
