# your_project_folder/models/qwen_model.py

import os
import torch
import librosa
import re
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from .base_model import MultimodalModel # 从同一目录下的 base_model 导入基类
from utils.text_processing import mark_words_in_text

class QwenAudioModel(MultimodalModel):
    """Qwen2-Audio-7B-Instruct 模型的具体实现。"""

    def __init__(self, model_path: str, processor_path: str, device: str, **kwargs):
        """
        加载 Qwen2-Audio 模型和处理器。
        """
        print("="*50)
        print(f"正在从 '{model_path}' 加载 Qwen 模型和处理器...")
        print(f"将使用设备: {device}")
        
        try:
            self.processor = AutoProcessor.from_pretrained(processor_path)
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, device_map=device)
            self.device = device
            print("Qwen 模型和处理器加载成功！")
        except Exception as e:
            print(f"错误: 加载 Qwen 模型或处理器失败。请检查路径。")
            raise e # 重新抛出异常，让主程序知道失败了
        print("="*50)

    def process(self, audio_path: str, transcription: str) -> str:
        """
        两阶段流程：
        1) 用 Qwen2-Audio 做 ASR 得到 asr_text；
        2) 仅基于 asr_text 让模型输出用逗号分隔的方言特有词汇；
        3) 在 asr_text 中使用 mark_words_in_text 标记这些词汇。
        """
        # try:
        if not os.path.exists(audio_path):
            print(f"错误: 音频文件未找到 at {audio_path}")
            return f"[错误: 音频文件未找到] {transcription}"

        # --- 阶段一：ASR 转写 ---
        asr_conversation = [
            {"role": "user", "content": "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n Transcribe the speech to texts"},
            # {"role": "user", "content": [
            #     {"type": "audio", "audio_url": audio_path},
            # ]}
            
        ]
        asr_text_template = self.processor.apply_chat_template(asr_conversation, add_generation_prompt=True, tokenize=False)
        audio_data, _ = librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)
        asr_inputs = self.processor(text=asr_text_template, audios=[audio_data], return_tensors="pt", padding=True)
        asr_inputs = asr_inputs.to(self.device)
        asr_ids = self.model.generate(**asr_inputs, max_length=2048)
        asr_ids = asr_ids[:, asr_inputs.input_ids.size(1):]
        asr_text = self.processor.batch_decode(asr_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        if not asr_text:
            asr_text = transcription or ""
        
        try:
            asr_text_ = asr_text.split("'")[1]
        except:
            asr_text_ = asr_text
            print(f"警告: 处理文本时未找到单引号，使用完整文本: {asr_text_}")

        # --- 阶段二：提取方言特有词汇 ---
        extract_conversation = [
            {"role": "system", "content": [
                {"type": "text", "text": f"""Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n {asr_text_} Extract all dialect-specific words from the speech, and separate them with commas. Do not include any other content. \n Example output: \n汾波, 门子\n """},
            ]},
            # {"role": "user", "content": [
            #     {"type": "text", "text": """你三不孜儿地看下停电短信息，是不是门子跳了""" },
            # ]},
            # {"role": "assistant", "content": [
            #     {"type": "text", "text": """三不孜儿地，门子""" + asr_text},
            # ]},
            # {"role": "user", "content": [
            #     {"type": "audio", "audio_url": audio_path},
            # ]},
            # {"role": "user", "content": [
            #     {"type": "text", "text": asr_text },
            # ]},
        ]
        # import pdb; pdb.set_trace()
        extract_text_template = self.processor.apply_chat_template(extract_conversation, add_generation_prompt=True, tokenize=False)
        extract_inputs = self.processor(text=extract_text_template, audios=[audio_data], return_tensors="pt", padding=True) #
        extract_inputs = {k: v.to(self.device) for k, v in extract_inputs.items()}
        extract_ids = self.model.generate(**extract_inputs, max_length=1024)
        extract_ids = extract_ids[:, extract_inputs["input_ids"].size(1):]
        dialect_words_text = self.processor.batch_decode(extract_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print("\n\n\n\nOriginal OUTPUT: \n", asr_text, '\n', asr_text_, '\n', dialect_words_text)
        # --- 阶段三：标记 ---
        dialect_words = [w.strip() for w in re.split(r"[,，]", dialect_words_text or "") if w.strip()]
        marked_text = mark_words_in_text(asr_text_, dialect_words)
        return marked_text

        # except Exception as e:
        #     print(f"处理文件 {os.path.basename(audio_path)} 时发生模型推理错误: {e}")
        return transcription