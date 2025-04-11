from llm_client import LLMClient

def analyze(content):
    """
    分析哈佛语和汉语对照内容，提取哈佛语词汇并进行分类
    """
    # 初始化LLM客户端
    client = LLMClient()
    
    # 构建prompt
    prompt = f"""以下是哈佛语与中文的对照例句：
                {content}

                任务：
                - 列出所有哈佛语词汇（如qu, nt, jip等），并为每个词汇标注其可能的类别：
                - 主语（如"我/你/他/我们/你们/他们"）
                - 动词（如"睡/看见/知道/打"）
                - 时态标记（如过去时、现在时）
                - 疑问标记（如"吗"）
                - 其他（无法分类的词汇）

                输出格式：
                {{词汇: 类别}}"""
                    
    messages = [
        {"role": "system", "content": "你是一个语言学专家，擅长分析不同语言的语法结构和词汇分类。"},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat_completion(messages=messages, temperature=0.2)
    result = response.choices[0].message.content

    return {"raw_result": result}

