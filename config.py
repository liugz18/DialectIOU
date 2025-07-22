# your_project_folder/config.py
import torch

# --- 模型选择 ---
# 在这里指定要使用的模型类名 (与 models/ 文件夹下的类名对应)
SELECTED_MODEL = "QwenAudioModel" 
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
    # "WhisperLargeV3Model": {
    #     "model_path": "openai/whisper-large-v3",
    #     "processor_path": "openai/whisper-large-v3"
    # },
    # 未来可以在这里添加更多模型的配置
}


# --- 通用硬件配置 ---
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"