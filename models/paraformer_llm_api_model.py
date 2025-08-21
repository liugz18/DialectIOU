# your_project_folder/models/paraformer_llm_api_model.py
import re
import os
import requests
import json
import getpass
import librosa
import numpy as np

# Local imports from our project structure
from .base_model import MultimodalModel

# --- Module-level cache for the API key ---
_API_KEY_CACHE = None

def _get_api_key(api_key_name: str):
    """
    Securely prompts the user for an API key once and caches it for the session.
    """
    global _API_KEY_CACHE
    if _API_KEY_CACHE is None:
        try:
            print("-" * 60)
            print(f"需要 API Key 来访问远程 LLM 服务 ({api_key_name})")
            _API_KEY_CACHE = getpass.getpass("请输入您的 API Key: ")
            print("API Key 已接收。")
            print("-" * 60)
        except Exception as e:
            print(f"无法读取 API Key: {e}")
            _API_KEY_CACHE = None
    return _API_KEY_CACHE


class ParaformerLlmApiModel(MultimodalModel):
    """
    一个两阶段模型：
    1. 使用本地 funasr 模型进行语音识别。
    2. 调用远程 LLM API 对识别出的文本进行方言区间检测。
    """
    def __init__(self, device: str, **kwargs):
        """
        初始化 funasr 模型和 LLM API 的配置。
        
        Args:
            device (str): 运行 funasr 模型的设备。
            **kwargs: 包含所有模型配置的字典。
        """
        print("="*50)
        print("正在初始化 ParaformerLlmApiModel...")
        
        # --- 1. 初始化 funasr ---
        self.paraformer_model = None
        funasr_model_path = kwargs.get("model_path")
        if not funasr_model_path:
            raise ValueError("配置错误: ParaformerLlmApiModel 需要 'funasr_model_path'。")
            
        try:
            from funasr import AutoModel
            self.paraformer_model = AutoModel(model=funasr_model_path, device=device)
            print(f"funasr 模型 '{funasr_model_path}' 加载成功！")
        except ImportError:
            print("错误: 未找到 'funasr' 库。请先执行 'pip install funasr'。")
            raise
        except Exception as e:
            print(f"错误: funasr 模型加载失败: {e}")
            raise

        # --- 2. 初始化 LLM API 配置 ---
        self.llm_api_url = kwargs.get("llm_api_url")
        self.llm_model_name = kwargs.get("llm_model_name")
        self.llm_prompt_template = \
"""对于方言音频以及给定的转写成的文字，找出其中所有的方言特有表达词汇，并用逗号隔开，不用输出其他内容，注意有的方言表达是没有汉字对应的拟声词
案例输入：
你三不孜儿地看下停电短信息，是不是门子跳了
案例输出：
三不孜儿地，门子

输入：
{transcription}

输出：
"""
        self.llm_input_source = kwargs.get("llm_input_source", "gt").lower() # 默认使用 'gt'
        
        if not all([self.llm_api_url, self.llm_model_name, self.llm_prompt_template]):
            raise ValueError("配置错误: LLM API 需要 'llm_api_url', 'llm_model_name', 和 'llm_prompt_template'。")
        
        print(f"LLM API 配置加载成功 (模型: {self.llm_model_name}, 输入源: {self.llm_input_source.upper()})")
        
        # --- 3. 获取并缓存 API Key ---
        self.api_key = _get_api_key(self.llm_model_name)
        if not self.api_key:
            raise ValueError("未能获取 API Key，无法继续。")

        print("ParaformerLlmApiModel 初始化完成！")
        print("="*50)


    def _run_paraformer(self, audio_path: str) -> str:
        """使用 funasr 对单个音频文件进行转写。"""
        if not self.paraformer_model:
            return "[paraformer失败: 模型未加载]"
        try:
            # 使用 librosa 加载音频，与之前模型保持一致
            # audio_data, sample_rate = librosa.load(audio_path, sr=16000) # funasr 通常使用 16kHz
            
            # funasr 的 generate 方法需要的是 numpy 数组
            res = self.paraformer_model.generate(input=audio_path, batch_size_s=300,
                                 output_dir="./output")
            
            # 提取文本结果
            paraformer_text = res[0].get("text", "").replace(" ", "")
            return paraformer_text if paraformer_text else "[paraformer失败: 未返回文本]"

        except Exception as e:
            print(f"funasr 在处理 {os.path.basename(audio_path)} 时发生错误: {e}")
            return f"[paraformer失败: {e}]"


    def _call_llm_api(self, text_to_process: str) -> str:
        """调用远程 LLM API 进行区间检测。"""
        prompt = self.llm_prompt_template.format(transcription=text_to_process)
        payload = {
            "model": self.llm_model_name,
            "messages": [{"role": "user", "content": prompt}]
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.llm_api_url, json=payload, headers=headers, timeout=300)
            response.raise_for_status() # 如果状态码不是 2xx，则抛出异常
            result = response.json()

            if result.get("choices") and result["choices"][0].get("message"):
                fused_text = result["choices"][0]["message"].get("content", "").replace('\n',"")
                return fused_text if fused_text else "[LLM返回空内容]"
            else:
                return f"[LLM返回格式错误: {response.text}]"
        except requests.exceptions.RequestException as e:
            return f"[LLM请求失败: {e}]"
        except Exception as e:
            return f"[LLM未知错误: {e}]"

    def process(self, audio_path: str, transcription: str) -> str:
        """
        执行完整的 paraformer -> LLM 流水线。
        """
        # --- 决定 LLM 的输入文本 ---
        if self.llm_input_source == 'paraformer':
            # 如果选择 paraformer 作为输入源，先运行 paraformer
            print(f"  -> 正在运行 funasr...")
            text_for_llm = self._run_paraformer(audio_path)
            print(f"  -> funasr 结果: {text_for_llm}")
        else:
            # 否则，直接使用 Ground Truth 文本
            text_for_llm = transcription

        # --- 调用 LLM API ---
        print(f"  -> 正在调用 LLM API...")
        final_text = self._call_llm_api(text_for_llm)

        # --- 后处理，提取方言词汇并用【】标记在原文本中 ---
        if final_text:
            # 1. 用正则提取出final_text中所有被"," 或"，"分隔开的词
            from utils.text_processing import mark_words_in_text
            
            dialect_words = [word.strip() for word in re.split(r'[,，]', final_text) if word.strip()]
            final_text = mark_words_in_text(text_for_llm, dialect_words)
            
        return final_text