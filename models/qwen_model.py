# your_project_folder/models/qwen_model.py

import os
import torch
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from .base_model import MultimodalModel # 从同一目录下的 base_model 导入基类

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
        使用 Qwen 模型处理单个音频文件，并返回标记了方言词的文本。

        Args:
            audio_path (str): 音频文件的绝对路径。
            transcription (str): 音频对应的纯文本转写。

        Returns:
            str: 模型生成的、用 <> 标记了方言词的文本。
        """
        try:
            if not os.path.exists(audio_path):
                print(f"错误: 音频文件未找到 at {audio_path}")
                return f"[错误: 音频文件未找到] {transcription}"

            # 1. 构建模型的对话输入 (使用 few-shot 示例)
            conversation = [
                {'role': 'system', 'content': '听以下音频，结合已有的转写文字，找出其中方言特有的词汇，如输入“你三不孜儿地看哈短信息”，只需输出“三不孜儿地”，不需要输出其他内容\n'},
                # {"role": "user", "content": [
                #     # {"type": "audio", "audio_url": "./example1.wav"}, # 示例，不会真的加载
                #     {"type": "text", "text":"你三不孜儿地看哈短信息"},
                # ]},
                # {"role": "assistant", "content": "你（三不孜儿地）看哈短信息"},
                # --- 这是我们要处理的实际数据 ---
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": transcription}, 
                ]},
            ]

            # 2. 准备输入
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audio_data, _ = librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)
            inputs = self.processor(text=text, audios=[audio_data], return_tensors="pt", padding=True)
            inputs = inputs.to(self.device)

            # 3. 模型推理
            generate_ids = self.model.generate(**inputs, max_length=2048)
            generate_ids = generate_ids[:, inputs.input_ids.size(1):]
            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            # 4. 后处理：将模型输出的 `()` 转换为 `<>`
            response = response.replace('(', '<').replace(')', '>')
            
            return response

        except Exception as e:
            print(f"处理文件 {os.path.basename(audio_path)} 时发生模型推理错误: {e}")
            return transcription # 推理失败，返回原始纯文本作为兜底