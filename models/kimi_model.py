import os
import sys
import re
from .base_model import MultimodalModel
from utils.text_processing import mark_words_in_text
kimi_root_path = "/mnt/sda/20250403来自HDD的备份/YuYinDuoMoTai/Kimi-Audio"
sys.path.append(kimi_root_path)
from kimia_infer.api.kimia import KimiAudio

class KimiAudioModel(MultimodalModel):
    """
    Kimi-Audio-7B-Instruct 模型的具体实现。
    使用 kimia_infer 库进行推理。
    """

    def __init__(self, model_path: str, processor_path: str, device: str, **kwargs):
        """
        加载 Kimi-Audio 模型。

        Args:
            model_path (str): Kimi-Audio 模型的路径。
            processor_path (str): 此模型未使用，但为保持接口一致性而保留。
            device (str): 此模型未使用，kimia_infer 内部处理设备。
            **kwargs: 用于接收额外的配置，如此处的 sampling_params。
        """
        model_path = f"{kimi_root_path}/{model_path}"
        print("="*50)
        print(f"正在从 '{model_path}' 加载 Kimi-Audio 模型...")
        
        try:
            # KimiAudio 库在其内部处理设备分配
            self.model = KimiAudio(
                model_path=model_path,
                load_detokenizer=False, # 根据您的示例代码设置
            )
            # 从 kwargs 获取采样参数，如果没有提供则使用空字典
            self.sampling_params = kwargs.get("sampling_params", {})
            print("Kimi-Audio 模型加载成功！")
            if self.sampling_params:
                print("已加载以下采样参数:")
                for key, value in self.sampling_params.items():
                    print(f"  - {key}: {value}")

        except Exception as e:
            print(f"错误: 加载 Kimi-Audio 模型失败。请检查路径和依赖。")
            raise e
        print("="*50)

    def process(self, audio_path: str, transcription: str) -> str:
        """
        两阶段流程：
        1) 先用 Kimi-Audio 做 ASR，得到识别文本 asr_text；
        2) 再用 Kimi-Audio 基于 asr_text 让其输出用逗号分隔的方言特有词汇；
        3) 使用 mark_words_in_text 在 asr_text 中标记这些词汇，返回标记后的文本。
        """
        # try:
        if not os.path.exists(audio_path):
            print(f"错误: 音频文件未找到 at {audio_path}")
            return f"[错误: 音频文件未找到] {transcription}"

        # --- 阶段一：ASR 转写 ---
        asr_messages = [
            {"role": "user", "message_type": "text", "content": "请将接下来提供的音频内容转写为中文，不要添加任何额外说明。"},
            {"role": "user", "message_type": "audio", "content": audio_path},
        ]
        _wav, asr_text = self.model.generate(
            asr_messages,
            **self.sampling_params,
            output_type="text"
        )
        if not asr_text:
            asr_text = transcription or ""

        # --- 阶段二：提取方言特有词汇（逗号分隔）---
        extract_messages = [
            {"role": "user", "message_type": "text", "content": """对于方言音频以及给定的转写成的文字，找出其中所有的方言特有表达词汇，并用逗号隔开，不用输出其他内容，注意有的方言表达是没有汉字对应的拟声词
案例输入：
你三不孜儿地看下停电短信息，是不是门子跳了
案例输出：
三不孜儿地，门子

输入："""},
            {"role": "user", "message_type": "text", "content": asr_text},
            {"role": "user", "message_type": "audio", "content": audio_path},
            
        ]
        # import pdb; pdb.set_trace()
        _wav, dialect_words_text = self.model.generate(
            extract_messages,
            **self.sampling_params,
            output_type="text"
        )

        # --- 阶段三：在 ASR 文本中标记方言词 ---
        dialect_words = [w.strip() for w in re.split(r"[,，]", dialect_words_text or "") if w.strip()]
        marked_text = mark_words_in_text(asr_text, dialect_words)
        return marked_text

        # except Exception as e:
        #     print(f"处理文件 {os.path.basename(audio_path)} 时发生 Kimi-Audio 模型推理错误: {e}")
        #     return transcription