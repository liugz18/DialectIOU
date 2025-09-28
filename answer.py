import os
import json
import re
import importlib
import sys
from datetime import datetime

# 从配置文件导入所有配置
import config

def get_model_instance():
    """动态导入并实例化所选的模型。"""
    try:
        model_name = config.SELECTED_MODEL
        model_config = config.MODEL_CONFIGS[model_name]
        
        # 动态地从 models 包中导入对应的模块
        module_name = model_config['module_name']
        model_module = importlib.import_module(module_name)
        
        # 从模块中获取模型类
        ModelClass = getattr(model_module, model_name)
        
        # 实例化模型
        print(f"正在实例化模型: {model_name}")
        instance = ModelClass(
            device=config.DEVICE,
            **model_config
        )
        return instance

    except (ImportError, KeyError, AttributeError) as e:
        print(f"错误: 无法加载模型 '{config.SELECTED_MODEL}'。请检查 'config.py' 和 'models' 文件夹。")
        print(f"详细错误: {e}")
        return None

def load_quizzes(quizzes_path):
    """加载Quizzes.json文件"""
    if not os.path.exists(quizzes_path):
        print(f"错误: Quizzes.json文件未找到 at {quizzes_path}")
        return None
    
    try:
        with open(quizzes_path, 'r', encoding='utf-8') as f:
            quizzes = json.load(f)
        return quizzes
    except Exception as e:
        print(f"错误: 加载Quizzes.json失败: {e}")
        return None

def find_audio_file(sentence_id, audio_base_path):
    """根据句子ID查找对应的音频文件"""
    # 尝试不同的音频文件扩展名
    extensions = ['.wav', '.mp3', '.flac', '.m4a']
    
    for ext in extensions:
        audio_path = os.path.join(audio_base_path, f"dialect_{sentence_id}{ext}")
        if os.path.exists(audio_path):
            return audio_path
    
    # 如果没找到，尝试查找其他可能的命名模式
    for ext in extensions:
        # 尝试查找以句子ID开头的文件
        for file in os.listdir(audio_base_path):
            if file.startswith(sentence_id) and file.endswith(ext):
                return os.path.join(audio_base_path, file)
    
    return None

def run_quizzes_evaluation(model, quizzes, audio_base_path):
    """运行测验评估，按句子长度分类统计准确率"""
    if not model or not quizzes:
        return
    
    # 初始化统计变量
    total_stats = {
        'short': {'total': 0, 'correct': 0, 'sentences': 0},
        'medium': {'total': 0, 'correct': 0, 'sentences': 0},
        'long': {'total': 0, 'correct': 0, 'sentences': 0}
    }
    
    print("=" * 80)
    print("开始评估测验题目...")
    print("=" * 80)
    
    for idx, quiz in enumerate(quizzes):
        sentence_id = quiz.get("sentence_id", "未知")
        sentence = quiz.get("sentence", "")
        quiz_data = quiz.get("quiz_data", {})
        questions = quiz_data.get("questions", [])
        dialect_explanations = quiz.get("dialect_explanations", None)  # 获取 dialect_explanations 参数
        
        print(f"\n处理句子 ID: {sentence_id}")
        print(f"句子: {sentence}")
        
        # 根据句子长度确定类别
        sentence_length = len(sentence)
        if sentence_length < 30:
        # if idx < 200:  # 前20题作为短句测试
            category = 'short'
        elif sentence_length < 100:
            category = 'medium'
        else:
            category = 'long'
        
        total_stats[category]['sentences'] += 1
        
        # 查找对应的音频文件
        audio_path = find_audio_file(sentence_id, audio_base_path)
        if not audio_path:
            print(f"警告: 未找到句子 {sentence_id} 的音频文件")
            continue
        
        print(f"使用音频文件: {os.path.basename(audio_path)}")
        print(f"句子长度: {sentence_length} 字符，类别: {category}")
        
        # 处理每个问题
        for i, question_data in enumerate(questions):
            question = question_data.get("question", "")
            options = question_data.get("options", [])
            correct_answer = question_data.get("answer", "")
            
            print(f"\n问题 {i+1}: {question}")
            for option in options:
                print(f"  {option}")
            
            # 调用模型的answer函数
            try:
                if config.USE_DIALECT_EXPLANATIONS and dialect_explanations:
                    model_answer = model.answer(audio_path, question, options, dialect_explanations=dialect_explanations)
                else:
                    model_answer = model.answer(audio_path, question, options)
                
                print(f"模型答案: {model_answer}")
                print(f"正确答案: {correct_answer}")
                
                # 检查答案是否正确
                if model_answer == correct_answer:
                    print("✓ 正确")
                    total_stats[category]['correct'] += 1
                else:
                    print("✗ 错误")
                
                total_stats[category]['total'] += 1
                
            except Exception as e:
                print(f"处理问题时出错: {e}")
                continue
    
    # 打印总体结果
    print("\n" + "=" * 80)
    print("评估结果")
    print("=" * 80)
    
    # 总统计
    total_questions = sum(stats['total'] for stats in total_stats.values())
    total_correct = sum(stats['correct'] for stats in total_stats.values())
    
    print(f"总问题数: {total_questions}")
    print(f"总正确回答: {total_correct}")
    
    if total_questions > 0:
        overall_accuracy = total_correct / total_questions * 100
        print(f"总体准确率: {overall_accuracy:.2f}%")
    else:
        print("没有处理任何问题，无法计算准确率")
    
    # 分类别统计
    print("\n按句子长度分类统计:")
    print("-" * 40)
    
    for category, stats in total_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total'] * 100
            print(f"{category.capitalize()}类别 (<30字: {category=='short'}, 30-99字: {category=='medium'}, ≥100字: {category=='long'}):")
            print(f"  句子数量: {stats['sentences']}")
            print(f"  问题数量: {stats['total']}")
            print(f"  正确回答: {stats['correct']}")
            print(f"  准确率: {accuracy:.2f}%")
        else:
            print(f"{category.capitalize()}类别: 无相关问题")

if __name__ == '__main__':
    # 打印配置信息
    config.print_config()
    
    # 检查配置的路径
    if not os.path.isdir(config.AUDIO_BASE_PATH):
        print(f"错误: 配置的音频根目录 AUDIO_BASE_PATH 不存在: '{config.AUDIO_BASE_PATH}'")
    else:
        # 动态加载并实例化模型
        model_instance = get_model_instance()
        
        if model_instance:
            # 加载测验数据
            quizzes_path = getattr(config, 'QUIZZES_PATH', 'ShangHaiQuizzes.json')
            quizzes = load_quizzes(quizzes_path)
            
            if quizzes:
                # 运行评估
                run_quizzes_evaluation(
                    model=model_instance,
                    quizzes=quizzes,
                    audio_base_path=config.AUDIO_BASE_PATH
                )