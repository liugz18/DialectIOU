# your_project_folder/config.py
import torch
import types
import pprint
import os

# --- 模型选择 ---
SELECTED_MODEL = "ParaformerLlmApiModel"
# 如果未来有新模型，例如: SELECTED_MODEL = "WhisperLargeV3Model"

# --- 路径配置 ---
AUDIO_BASE_PATH = "/mnt/sda/20250403来自HDD的备份/YuYinDuoMoTai/XiNanData"#"/mnt/sda/ASR/DataSets/LabelData/20250606fangyanciku_tiqu/filtered_audio_huangzhou/huangzhou"
TEXT_FILE_PATH = "text.txt"#'fangyan_text.txt'

# --- 不同模型的具体配置 ---
MODEL_CONFIGS = {
    "QwenAudioModel": {
        "module_name": "models.qwen_model",
        "model_path": "../Qwen2-Audio-7B-Instruct",
        "processor_path": "../Qwen2-Audio-7B-Instruct"
    },
    "KimiAudioModel": {
        "module_name": "models.kimi_model",
        "model_path": "Kimi-Audio-7B-Instruct",
        "processor_path": None,
        "sampling_params": {
            "audio_temperature": 0.8,
            "audio_top_k": 10,
            "text_temperature": 0.0,
            "text_top_k": 5,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
        }
    },
    "ParaformerLlmApiModel": {
        "module_name": "models.paraformer_llm_api_model",
        "processor_path": None,
        "model_path": "/mnt/sda/ASR/model/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",#"/mnt/sda/ASR/zhanghui/FunASR/inference_model/secondmodel/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-jingzhou",
        "llm_api_url": "https://api.siliconflow.cn/v1/chat/completions",
        "llm_model_name": "THUDM/GLM-4.1V-9B-Thinking",
        "llm_input_source": "paraformer"
    }
}

# --- 评估方法配置 ---
USE_WORD_COMPARISON = False

# 外部分段评估器配置（用于带括号文本的匹配评估）
# 若启用，则在非词级别评估分支中调用指定脚本里的评估类
USE_EXTERNAL_SEGMENT_EVALUATOR = True
EXTERNAL_EVALUATOR_FILE = "/mnt/sda/20250403来自HDD的备份/YuYinDuoMoTai/PhonemizerTest/power-asr/chinese_segment_evaluator.py"
EXTERNAL_EVALUATOR_CLASS = "ChineseSegmentEvaluator"

# --- 通用硬件配置 ---
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

def print_config():
    """打印当前所有配置项，美化输出"""
    from pprint import pformat
    
    print(f"\n{'-'*60}")
    print(f"配置文件路径: {__file__}")
    
    print("\n当前加载的配置项:")
    
    # 收集所有非系统属性
    config_items = [
        (k, v) for k, v in globals().items()
        if not k.startswith('__') 
        and not isinstance(v, types.ModuleType) 
        and not callable(v)
        and k != 'print_config'  # 排除自己
    ]
    
    # 找出最长键名用于对齐
    max_key_len = max(len(k) for k, _ in config_items) if config_items else 0
    
    for key, value in config_items:
        if key == 'MODEL_CONFIGS':
            # 特殊处理嵌套配置
            print(f"  {key:<{max_key_len}} →")
            for model, config in value.items():
                print(f"    ├── {model}:")
                for k, v in config.items():
                    if isinstance(v, dict):
                        print(f"    │   ├── {k}:")
                        for sk, sv in v.items():
                            print(f"    │   │   ├── {sk} = {sv}")
                    else:
                        print(f"    │   ├── {k} = {v}")
            continue
        
        # 普通配置项处理
        if isinstance(value, str) and ('/' in value or '\\' in value):
            # 路径类型的美化
            print(f"  {key:<{max_key_len}} → ├─{value}")
        elif isinstance(value, (list, dict, tuple, set)):
            # 复杂结构的格式化
            formatted = pformat(value, width=100, compact=True, indent=2)
            print(f"  {key:<{max_key_len}} → \n{formatted}")
        else:
            print(f"  {key:<{max_key_len}} → {value}")
    
    print('-'*60 + '\n')

# 初始化时自动打印配置（可选）
if __name__ == '__main__':
    print_config()