from openai import OpenAI

# API配置
ZHIZENGZENG_API_KEY = "sk-zk2401d2ebeef4b3db1a034d699056eff3367f933b8c6170"
ZHIZENGZENG_BASE_URL = "https://api.zhizengzeng.com/v1/"
DEEPSEEK_API_KEY = "sk-66952492e6464a3da6f79d04d62ed89e"  
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

class LLMClient:
    def __init__(self, 
                 api_key: str = None, 
                 base_url: str = None,
                 provider: str = "zhizengzeng"):  # 添加 provider 参数
        self.provider = provider
        
        # 根据 provider 初始化不同的客户端
        if provider == "deepseek":
            self.api_key = DEEPSEEK_API_KEY
            self.base_url = DEEPSEEK_BASE_URL
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )


        elif provider == "zhizengzeng":
            self.api_key = ZHIZENGZENG_API_KEY
            self.base_url = ZHIZENGZENG_BASE_URL
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

    
    def chat_completion(self, messages, model="deepseek-chat", temperature=0.7, **kwargs):
            
        return self.client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            **kwargs
        )




