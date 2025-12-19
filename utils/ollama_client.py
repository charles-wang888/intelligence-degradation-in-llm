"""
Ollama API客户端
用于调用Qwen系列模型
"""

import requests
import json
import time
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class OllamaClient:
    """Ollama API客户端类"""
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300):
        """
        初始化Ollama客户端
        
        Args:
            base_url: Ollama服务地址
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.api_url = f"{self.base_url}/api/generate"
        
    def generate(self, 
                 model: str,
                 prompt: str,
                 temperature: float = 0.0,
                 max_tokens: int = 512,
                 stream: bool = False) -> Dict:
        """
        生成文本
        
        Args:
            model: 模型名称（如 "qwen2:1.5b"）
            prompt: 输入提示词
            temperature: 温度参数
            max_tokens: 最大生成token数
            stream: 是否流式输出
            
        Returns:
            包含生成结果的字典
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": stream
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                # 处理流式输出
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if 'response' in chunk:
                            full_response += chunk['response']
                        if chunk.get('done', False):
                            break
                return {"response": full_response, "done": True}
            else:
                return response.json()
                
        except requests.exceptions.Timeout:
            logger.error(f"请求超时: model={model}, timeout={self.timeout}s")
            return {"error": "timeout", "response": ""}
        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {e}")
            return {"error": str(e), "response": ""}
    
    def check_model_available(self, model: str) -> bool:
        """
        检查模型是否可用
        
        Args:
            model: 模型名称
            
        Returns:
            模型是否可用
        """
        try:
            # 尝试发送一个简单的请求
            test_response = self.generate(
                model=model,
                prompt="test",
                max_tokens=1
            )
            return "error" not in test_response
        except Exception as e:
            logger.error(f"检查模型失败: {e}")
            return False
    
    def get_model_info(self, model: str) -> Optional[Dict]:
        """
        获取模型信息
        
        Args:
            model: 模型名称
            
        Returns:
            模型信息字典
        """
        try:
            list_url = f"{self.base_url}/api/tags"
            response = requests.get(list_url, timeout=10)
            response.raise_for_status()
            
            models = response.json().get("models", [])
            for m in models:
                if m.get("name") == model:
                    return m
            return None
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return None
    
    def get_model_context_length(self, model: str) -> Optional[int]:
        """
        尝试从Ollama API获取模型的最大上下文长度
        
        Args:
            model: 模型名称
            
        Returns:
            最大上下文长度（token数），如果无法获取则返回None
        """
        try:
            # 尝试使用show API获取模型详细信息
            show_url = f"{self.base_url}/api/show"
            response = requests.post(
                show_url,
                json={"name": model},
                timeout=10
            )
            response.raise_for_status()
            
            model_info = response.json()
            # 尝试从不同可能的字段中获取上下文长度
            # Ollama的show API可能返回不同的字段名
            context_length = (
                model_info.get("modelfile", {}).get("parameter", {}).get("num_ctx") or
                model_info.get("details", {}).get("context_length") or
                model_info.get("context_length") or
                model_info.get("num_ctx")
            )
            
            if context_length:
                return int(context_length)
            
            return None
        except Exception as e:
            logger.debug(f"无法从API获取模型上下文长度: {e}")
            return None


def test_ollama_connection(base_url: str = "http://localhost:11434") -> bool:
    """
    测试Ollama连接
    
    Args:
        base_url: Ollama服务地址
        
    Returns:
        连接是否成功
    """
    try:
        client = OllamaClient(base_url=base_url)
        # 尝试列出模型
        list_url = f"{base_url}/api/tags"
        response = requests.get(list_url, timeout=5)
        response.raise_for_status()
        logger.info("Ollama连接成功")
        return True
    except Exception as e:
        logger.error(f"Ollama连接失败: {e}")
        logger.error("请确保Ollama服务正在运行: ollama serve")
        return False


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    if test_ollama_connection():
        client = OllamaClient()
        # 测试生成
        result = client.generate(
            model="qwen2:1.5b",
            prompt="你好，请介绍一下你自己。",
            temperature=0.0,
            max_tokens=100
        )
        print("生成结果:", result.get("response", ""))

