# your_project_folder/config.py
import torch

# --- 模型选择 ---
# 在这里指定要使用的模型类名 (与 models/ 文件夹下的类名对应)
SELECTED_MODEL = "ParaformerLlmApiModel"#"QwenAudioModel" #"KimiAudioModel"#
# 如果未来有新模型，例如: SELECTED_MODEL = "WhisperLargeV3Model"

# --- 路径配置 ---
# 存放 .wav 音频文件的文件夹绝对路径
AUDIO_BASE_PATH = "/mnt/sda/ASR/DataSets/LabelData/20250606fangyanciku_tiqu/filtered_audio_huangzhou/huangzhou"
# 存放标注文件的路径
TEXT_FILE_PATH = 'fangyan_text.txt'

# --- 不同模型的具体配置 ---
MODEL_CONFIGS = {
    "QwenAudioModel": {
        "module_name": "models.qwen_model",
        "model_path": "../Qwen2-Audio-7B-Instruct",
        "processor_path": "../Qwen2-Audio-7B-Instruct"
    },
    "KimiAudioModel": {
        "module_name": "models.kimi_model",
        "model_path": "Kimi-Audio-7B-Instruct", # Kimi 模型的路径
        "processor_path": None, # 此模型不使用单独的 processor
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
        "processor_path": None, # 此模型不使用单独的 processor
        # --- Stage 1: FunASR 配置 ---
        "model_path": "/mnt/sda/ASR/zhanghui/FunASR/inference_model/secondmodel/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-jingzhou", # 您可以换成其他 FunASR 模型，例如本地路径

        # --- Stage 2: 远程 LLM API 配置 ---
        "llm_api_url": "https://api.siliconflow.cn/v1/chat/completions",
        "llm_model_name": "Qwen/Qwen3-32B", # 确保您的 key 支持此模型
        
        # LLM 的输入源: 'asr' (使用FunASR的结果) 或 'gt' (使用标准答案文本)
        "llm_input_source": "gt", 
    }
    # "WhisperLargeV3Model": {
    #     "model_path": "openai/whisper-large-v3",
    #     "processor_path": "openai/whisper-large-v3"
    # },
    # 未来可以在这里添加更多模型的配置
}


# --- 通用硬件配置 ---
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"