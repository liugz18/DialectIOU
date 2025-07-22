# your_project_folder/models/base_model.py

from abc import ABC, abstractmethod

class MultimodalModel(ABC):
    """
    多模态模型接口的抽象基类。
    所有具体的模型实现都应继承此类并实现其方法。
    """
    
    @abstractmethod
    def __init__(self, model_path: str, processor_path: str, device: str, **kwargs):
        """
        初始化模型，加载处理器和模型到指定设备。
        
        Args:
            model_path (str): 模型权重文件的路径。
            processor_path (str): 处理器/分词器的路径。
            device (str): 运行设备的名称 (例如 "cuda:0", "cpu")。
        """
        pass

    @abstractmethod
    def process(self, audio_path: str, transcription: str) -> str:
        """
        处理单个音频和文本对，返回模型的预测结果。

        Args:
            audio_path (str): 音频文件的绝对路径。
            transcription (str): 音频对应的纯文本转写。

        Returns:
            str: 模型生成的、带有标记的预测文本。
        """
        pass