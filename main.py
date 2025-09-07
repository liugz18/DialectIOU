# your_project_folder/main.py

import os
import re
import importlib
import sys

# 从配置文件导入所有配置
import config

# 从工具模块导入函数
from my_utils.text_processing import process_line_to_ground_truth, calculate_text_iou, calculate_word_metrics

def get_model_instance():
    """动态导入并实例化所选的模型。"""
    try:
        model_name = config.SELECTED_MODEL
        model_config = config.MODEL_CONFIGS[model_name]
        
        # 动态地从 models 包中导入对应的模块
        # 例如，如果 model_name 是 "QwenAudioModel"，则导入 models.qwen_model
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


def run_evaluation(model, text_file_path, audio_base_path):
    """主评估流程。"""
    text_file_path = os.path.join(audio_base_path, text_file_path)
    print("-" * 80)
    print(f"正在处理文本文件: {text_file_path}")
    print(f"音频文件根目录: {audio_base_path}")
    print("-" * 80)

    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误: 文本文件 '{text_file_path}' 未找到。")
        return

    used_external_evaluator = False

    if config.USE_WORD_COMPARISON:
        # 使用词汇级别的比对方法
        total_recall = 0
        total_precision = 0
        total_f1 = 0
        processed_lines = 0

        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            filename, transcription, gt_words = process_line_to_ground_truth(line, use_word_comparison=True)
            if filename is None or not filename:
                continue
            
            full_audio_path = os.path.join(audio_base_path, filename)
            
            # 调用模型的 process 方法，现在返回的是逗号分隔的词汇
            hyp_words = model.process(full_audio_path, transcription)
            
            recall, precision, f1 = calculate_word_metrics(gt_words, hyp_words)
            total_recall += recall
            total_precision += precision
            total_f1 += f1
            processed_lines += 1
            
            print(f"文件: {filename}")
            print(f"  GT文本: {transcription}")
            print(f"  GT词汇: {gt_words}")
            print(f"  HYP词汇: {hyp_words}")
            print(f"  召回率: {recall:.4f}, 准确率: {precision:.4f}, F1: {f1:.4f}\n")
    else:
        # 使用外部分段评估器（若启用），否则回退到 IoU
        evaluator = None
        if getattr(config, 'USE_EXTERNAL_SEGMENT_EVALUATOR', False):
            try:
                import importlib.util
                evaluator_file = getattr(config, 'EXTERNAL_EVALUATOR_FILE')
                # 将外部工程根目录加入 sys.path，确保其内部引用可解析（如 import power）
                project_root = os.path.dirname(evaluator_file)
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)

                spec = importlib.util.spec_from_file_location(
                    "external_evaluator", evaluator_file
                )
                if spec and spec.loader:
                    external_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(external_module)
                    EvaluatorClass = getattr(external_module, getattr(config, 'EXTERNAL_EVALUATOR_CLASS'))
                    evaluator = EvaluatorClass()
                    used_external_evaluator = True
                else:
                    print("警告: 无法加载外部评估器模块，回退到 IoU 评估。")
            except Exception as e:
                print(f"警告: 加载外部评估器失败，回退到 IoU 评估。错误: {e}")

        total_iou = 0
        processed_lines = 0

        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            filename, gt_text, _ = process_line_to_ground_truth(line, use_word_comparison=False)
            if filename is None or not filename:
                continue
            
            full_audio_path = os.path.join(audio_base_path, filename)
            if not os.path.exists(full_audio_path):
                print(f"错误: 音频文件未找到 at {full_audio_path}")
                continue
            plain_transcription = re.sub(r'[<>]', '', gt_text)
            
            # 调用模型的 process 方法
            hyp_text = model.process(full_audio_path, plain_transcription)

            print(f"文件: {filename}")
            print(f"  GT  : {gt_text}")
            print(f"  HYP : {hyp_text}")

            if evaluator is not None:
                try:
                    # 打印报告；外部评估器内部负责比对【】区间
                    _ = evaluator.print_evaluation_report(gt_text, hyp_text)
                except Exception as e:
                    print(f"  外部评估器运行失败: {e}")
            else:
                # 回退到 IoU
                iou = calculate_text_iou(gt_text, hyp_text)
                total_iou += iou
                print(f"  IoU : {iou:.4f}\n")
                processed_lines += 1

        # 仅在 IoU 回退路径下打印平均值
        if evaluator is None and processed_lines > 0:
            average_iou = total_iou / processed_lines
            print("-" * 80)
            print(f"处理完成！")
            print(f"总计处理行数: {processed_lines}")
            print(f"平均 IoU: {average_iou:.4f}")
            print("-" * 80)
            return

    # 使用词级/IoU路径的汇总输出；外部评估器路径已在上面逐条输出，且不需要此处的汇总
    if not used_external_evaluator and processed_lines > 0:
        if config.USE_WORD_COMPARISON:
            average_recall = total_recall / processed_lines
            average_precision = total_precision / processed_lines
            average_f1 = total_f1 / processed_lines
            print("-" * 80)
            print(f"处理完成！")
            print(f"总计处理行数: {processed_lines}")
            print(f"平均召回率: {average_recall:.4f}")
            print(f"平均准确率: {average_precision:.4f}")
            print(f"平均F1分数: {average_f1:.4f}")
            print("-" * 80)
        else:
            # 在使用外部评估器路径下，逐条打印评估报告，上面已输出
            pass
    elif not used_external_evaluator:
        print("没有可处理的有效数据行。")


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
            # 如果模型加载成功，则开始评估
            run_evaluation(
                model=model_instance,
                text_file_path=config.TEXT_FILE_PATH,
                audio_base_path=config.AUDIO_BASE_PATH
            )