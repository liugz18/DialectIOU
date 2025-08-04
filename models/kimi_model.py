import os
import sys
from .base_model import MultimodalModel
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
        使用 Kimi-Audio 模型处理单个音频文件。

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

            # 1. 构建 KimiAudio 格式的 messages
            # 我们请求模型使用 () 标记，以便后续统一转换为 <>
            messages = [
                {'role': 'user', "message_type": "text", 'content': """对于方言音频以及给定的转写成的文字，找出其中所有的方言特有表达词汇，并用逗号隔开，不用输出其他内容，注意有的方言表达是没有汉字对应的拟声词
案例输入：
你三不孜儿地看下停电短信息，是不是门子跳了
案例输出：
三不孜儿地，门子

输入："""},
                {"role": "user", "message_type": "audio", "content": audio_path },
                {"role": "user", "message_type": "text", "content": transcription},
            ]

            # 2. 模型推理
            # model.generate 返回 (wav, text) 元组，我们只需要文本部分
            _wav, response_text = self.model.generate(
                messages, 
                **self.sampling_params, 
                output_type="text"
            )

            # 3. 后处理：将模型输出的 `()` 转换为 `<>`
            if response_text:
                response_text = response_text.replace('(', '<').replace(')', '>')
            
            return response_text

        except Exception as e:
            print(f"处理文件 {os.path.basename(audio_path)} 时发生 Kimi-Audio 模型推理错误: {e}")
            return transcription # 推理失败，返回原始纯文本作为兜底