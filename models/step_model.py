import os
import sys
import re
from .base_model import MultimodalModel
import config

# Import utils first before modifying sys.path
from my_utils.text_processing import mark_words_in_text

# Store original sys.path
original_sys_path = sys.path.copy()

# Add the Step-Audio2 directory to the path
stepaudio_root_path = config.MODEL_CONFIGS["StepAudioModel"]["stepaudio_root_path"]
sys.path.append(stepaudio_root_path)

try:
    from stepaudio2 import StepAudio2
finally:
    # Restore original sys.path to avoid affecting other imports
    sys.path = original_sys_path

class StepAudioModel(MultimodalModel):
    """
    Step-Audio-2 模型的具体实现。
    使用 stepaudio2 库进行推理。
    """

    def __init__(self, model_path: str, processor_path: str, device: str, **kwargs):
        """
        加载 Step-Audio-2 模型。

        Args:
            model_path (str): Step-Audio-2 模型的名称 (例如 'Step-Audio-2-mini')。
            processor_path (str): 此模型未使用，但为保持接口一致性而保留。
            device (str): 此模型未使用，stepaudio2 内部处理设备。
            **kwargs: 用于接收额外的配置参数。
        """
        # 构建完整的模型路径
        full_model_path = os.path.join(stepaudio_root_path, model_path)
        print("="*50)
        print(f"正在加载 Step-Audio-2 模型: '{full_model_path}'...")
        
        try:
            # StepAudio2 库在其内部处理设备分配
            self.model = StepAudio2(full_model_path)
            # 从 kwargs 获取采样参数，如果没有提供则使用默认值
            self.max_new_tokens = kwargs.get("max_new_tokens", 256)
            self.temperature = kwargs.get("temperature", 0.1)
            self.do_sample = kwargs.get("do_sample", True)
            print("Step-Audio-2 模型加载成功！")
            print(f"已加载以下参数:")
            print(f"  - max_new_tokens: {self.max_new_tokens}")
            print(f"  - temperature: {self.temperature}")
            print(f"  - do_sample: {self.do_sample}")

        except Exception as e:
            print(f"错误: 加载 Step-Audio-2 模型失败。请检查路径和依赖。")
            raise e
        print("="*50)

    def process(self, audio_path: str, transcription: str) -> str:
        """
        两阶段流程：
        1) 先用 Step-Audio-2 做 ASR，得到识别文本 asr_text；
        2) 再用 Step-Audio-2 基于 asr_text 让其输出用逗号分隔的方言特有词汇；
        3) 使用 mark_words_in_text 在 asr_text 中标记这些词汇，返回标记后的文本。
        """
        if not os.path.exists(audio_path):
            print(f"错误: 音频文件未找到 at {audio_path}")
            return f"[错误: 音频文件未找到] {transcription}"

        # --- 阶段一：ASR 转写 ---
        asr_messages = [
            {"role": "system", "content": "请记录下你所听到的语音内容。"},
            {"role": "human", "content": [{"type": "audio", "audio": audio_path}]},
            {"role": "assistant", "content": None}
        ]
        tokens, asr_text, _ = self.model(
            asr_messages, 
            max_new_tokens=self.max_new_tokens
        )

        asr_text_ = asr_text.split(">")[-1]
        
        if not asr_text:
            asr_text = transcription or ""

        # --- 阶段二：提取方言特有词汇（逗号分隔）---
        extract_messages = [
            {"role": "system", "content": """对于方言音频以及给定的转写成的文字，找出其中所有的方言特有表达词汇，并用逗号隔开，不用输出其他内容，注意有的方言表达是没有汉字对应的拟声词
案例输入：
你三不孜儿地看下停电短信息，是不是门子跳了
案例输出：
三不孜儿地，门子

输入："""},
            {"role": "human", "message_type": "text", "content": asr_text_},
            {"role": "human", "content": [{"type": "audio", "audio": audio_path}]},
            {"role": "assistant", "content": None}
        ]
        
        tokens, dialect_words_text, _ = self.model(
            extract_messages, 
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample
        )

        print("\n\n\n\nOriginal OUTPUT: \n", asr_text, '\n', asr_text_, '\n', dialect_words_text)

        # --- 阶段三：在 ASR 文本中标记方言词 ---
        dialect_words = [w.strip() for w in re.split(r"[,，]", dialect_words_text or "") if w.strip()]
        marked_text = mark_words_in_text(asr_text_, dialect_words)
        return marked_text

    def answer(self, audio_path: str, question: str, options: list, dialect_explanations: str = None) -> str:
        """
        回答问题流程：
        1) 将问题、选项和上下文结合，让模型生成答案。
        """
        if not os.path.exists(audio_path):
            print(f"错误: 音频文件未找到 at {audio_path}")
            return "E"  # 返回错误标记

        # 构建选项文本
        options_text = "\n".join([option for option in options])
        
        # 构建提示词
        prompt = f"""请根据提供的音频和文本回答问题。

问题: {question}
选项:
{options_text}
"""
        # 如果 dialect_explanations 不为空，则加入到 prompt 中
        if dialect_explanations:
            prompt += f"\n方言解释: {dialect_explanations}\n"

        prompt += "请只输出答案的字母（例如：A），不要输出其他内容。"

        # 构建消息
        messages = [
            {"role": "system", "content": prompt},
            {"role": "human", "content": [{"type": "audio", "audio": audio_path}]},
            {"role": "assistant", "content": None}
        ]
        
        # 生成答案
        tokens, answer_text, _ = self.model(
            messages, 
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample
        )
        
        # 从回答中提取答案字母
        if answer_text:
            # 查找第一个大写字母
            match = re.search(r'[A-D]', answer_text)
            if match:
                return match.group()
        
        # 如果没有找到有效答案，返回错误标记
        return "E"