# eval_w_checkpoint.py
# 基于main.py的checkpoint版本，只保留USE_EXTERNAL_SEGMENT_EVALUATOR分支
# 能够从log文件恢复评估状态并继续处理

import os
import re
import importlib
import sys
import glob
import datetime
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 从配置文件导入所有配置
import config

# 从工具模块导入函数
from my_utils.text_processing import process_line_to_ground_truth

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

def find_latest_log_file():
    """查找最新的log文件（格式：YYYY-MM-DD_HH-MM-SS.log）"""
    log_files = glob.glob("20*.log")
    if not log_files:
        return None
    
    # 按文件名排序，取最新的
    log_files.sort(reverse=True)
    return log_files[0]

def parse_log_file(log_file_path):
    """解析log文件，提取最后处理的文件和rolling平均值状态"""
    if not os.path.exists(log_file_path):
        print(f"警告: log文件不存在: {log_file_path}")
        return None, None, {}, 0
    
    print(f"正在解析log文件: {log_file_path}")
    
    last_processed_file = None
    last_rolling_values = {}
    processed_count = 0
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 统计已处理的文件数量
        for line in lines:
            if line.strip().startswith("文件: ") and line.strip().endswith(".wav"):
                processed_count += 1
        
        # 从后往前查找最后处理的文件
        for line in reversed(lines):
            if line.strip().startswith("文件: ") and line.strip().endswith(".wav"):
                match = re.search(r"文件: (.+\.wav)", line.strip())
                if match:
                    last_processed_file = match.group(1)
                    break
        
        # 查找最后的rolling平均值
        for line in reversed(lines):
            if "rolling_recall_avg :" in line:
                # 从这一行开始向前查找所有rolling值
                current_line_idx = lines.index(line)
                for i in range(current_line_idx, len(lines)):
                    current_line = lines[i].strip()
                    if current_line.startswith("rolling_") and " : " in current_line:
                        parts = current_line.split(" : ")
                        if len(parts) == 2:
                            key = parts[0].strip()
                            try:
                                value = float(parts[1].strip())
                                last_rolling_values[key] = value
                            except ValueError:
                                pass
                    elif current_line.startswith("rolling_"):
                        # 继续查找rolling值
                        continue
                    else:
                        # 遇到非rolling行，停止
                        break
                break
                
    except Exception as e:
        print(f"解析log文件时出错: {e}")
        return None, None, {}, 0
    
    print(f"最后处理的文件是 {last_processed_file}")
    print(f"已处理文件数量: {processed_count}")
    print(f"找到的rolling值数量: {len(last_rolling_values)}")
    
    return last_processed_file, last_rolling_values, lines, processed_count

def restore_evaluator_state(evaluator, rolling_values, processed_count):
    """恢复评估器的rolling平均值状态"""
    if not evaluator:
        return
    
    print("正在恢复评估器状态...")
    
    # 恢复rolling计数
    evaluator._rolling_count = processed_count
    print(f"  恢复 _rolling_count = {processed_count}")

    
    # 尝试设置rolling平均值
    if rolling_values:
        for key, value in rolling_values.items():
            setattr(evaluator, key, value)
            print(f"  恢复 {key} = {value}")


def setup_logging(log_file_path):
    """设置日志输出，将输出同时显示在终端和追加到log文件"""
    if not log_file_path:
        return None
    
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        
        def flush(self):
            for f in self.files:
                f.flush()
    
    try:
        log_file = open(log_file_path, 'a', encoding='utf-8')
        tee = TeeOutput(sys.stdout, log_file)
        return tee, log_file
    except Exception as e:
        print(f"警告: 无法打开日志文件 {log_file_path}: {e}")
        return None, None

def run_evaluation_with_checkpoint(model, text_file_path, audio_base_path, log_file_path=None):
    """带checkpoint的主评估流程"""
    text_file_path = os.path.join(audio_base_path, text_file_path)
    
    # 设置日志文件路径
    if log_file_path is None:
        log_file_path = find_latest_log_file()
    
    # 设置日志输出
    tee_output, log_file = setup_logging(log_file_path)
    original_stdout = sys.stdout
    
    try:
        if tee_output:
            sys.stdout = tee_output
        
        # 添加续接标识
        print("\n" + "="*80)
        print(f"======== 续接执行: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ========")
        print("="*80)
        
        print("-" * 80)
        print(f"正在处理文本文件: {text_file_path}")
        print(f"音频文件根目录: {audio_base_path}")
        if log_file_path:
            print(f"将输出追加到日志文件: {log_file_path}")
        print("-" * 80)

        try:
            with open(text_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"错误: 文本文件 '{text_file_path}' 未找到。")
            return

        # 解析log文件获取checkpoint信息
        last_processed_file = None
        rolling_values = {}
        processed_count = 0
        
        if log_file_path:
            last_processed_file, rolling_values, _, processed_count = parse_log_file(log_file_path)
        else:
            # 自动查找最新的log文件
            latest_log = find_latest_log_file()
            if latest_log:
                last_processed_file, rolling_values, _, processed_count = parse_log_file(latest_log)

        # 初始化外部评估器
        evaluator = None
        used_external_evaluator = False
        
        if getattr(config, 'USE_EXTERNAL_SEGMENT_EVALUATOR', False):
            try:
                import importlib.util
                evaluator_file = getattr(config, 'EXTERNAL_EVALUATOR_FILE')
                # 将外部工程根目录加入 sys.path
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
                    
                    # 恢复评估器状态
                    restore_evaluator_state(evaluator, rolling_values, processed_count)
                    
                else:
                    print("警告: 无法加载外部评估器模块。")
            except Exception as e:
                print(f"警告: 加载外部评估器失败。错误: {e}")
        
        if not used_external_evaluator:
            print("错误: 外部评估器未启用或加载失败，无法继续。")
            return

        # 确定从哪个文件开始处理
        start_processing = False
        processed_lines = 0
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            filename, gt_text, _ = process_line_to_ground_truth(line, use_word_comparison=False)
            if filename is None or not filename:
                continue
            
            # 检查是否应该开始处理
            if not start_processing:
                if last_processed_file is None or filename == last_processed_file:
                    start_processing = True
                    if last_processed_file is not None:
                        print(f"从文件 {filename} 之后开始处理...")
                        continue
                    else:
                        print("从头开始处理所有文件...")
                else:
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
                # try:
                    # 打印报告；外部评估器内部负责比对【】区间
                _ = evaluator.print_evaluation_report(gt_text, hyp_text)
                # except Exception as e:
                #     print(f"  外部评估器运行失败: {e}")
            
            processed_lines += 1

        print("-" * 80)
        print(f"处理完成！")
        print(f"本次处理行数: {processed_lines}")
        print("="*80)
        print(f"======== 续接执行结束: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ========")
        print("="*80)
        
    finally:
        # 恢复原始输出
        if tee_output:
            sys.stdout = original_stdout
        if log_file:
            log_file.close()

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
            # 自动查找最新的log文件，或使用命令行参数指定的文件
            log_file_path = None
            if len(sys.argv) > 1:
                log_file_path = sys.argv[1]
                print(f"使用指定的log文件: {log_file_path}")
            else:
                log_file_path = find_latest_log_file()
                if log_file_path:
                    print(f"自动找到最新的log文件: {log_file_path}")
                else:
                    print("未找到log文件，将从头开始处理")
            
            # 如果模型加载成功，则开始评估
            run_evaluation_with_checkpoint(
                model=model_instance,
                text_file_path=config.TEXT_FILE_PATH,
                audio_base_path=config.AUDIO_BASE_PATH,
                log_file_path=log_file_path
            )
