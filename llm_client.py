from openai import OpenAI

# API配置
API_SECRET_KEY = "your_api_secret_key"
BASE_URL = "https://api.zhizengzeng.com/v1/"

class LLMClient:
    def __init__(self, api_key: str = API_SECRET_KEY, base_url: str = BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        
        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def chat_completion(self, messages, model="gpt-3.5-turbo", temperature=0.7, **kwargs):
        return self.client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            **kwargs
        )




