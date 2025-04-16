from openai import OpenAI

# API配置
ZHIZENGZENG_API_KEY = "your-zhizengzeng-api-key"
ZHIZENGZENG_BASE_URL = "https://api.zhizengzeng.com/v1/"
DEEPSEEK_API_KEY = "your-deepseek-api-key"  
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

class LLMClient:
    def __init__(self, 
                 api_key: str = API_SECRET_KEY, 
                 base_url: str = BASE_URL,
                 provider: str = "deepseek"):  # 添加 provider 参数
        self.provider = provider
        
        # 根据 provider 初始化不同的客户端
        if provider == "deepseek":
            self.client = OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL
            )
        elif provider == "zhizengzeng":
            self.client = OpenAI(
                api_key=self.ZHIZENGZENG_API_KEY,
                base_url=self.ZHIZENGZENG_BASE_URL
            )
    
    def chat_completion(self, messages, model="deepseek-chat", temperature=0.7, **kwargs):
            
        return self.client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            **kwargs
        )




