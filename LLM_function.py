from llm_client import LLMClient

def analyze_0(content):
    """
    分析哈坤语和汉语对照内容，提取哈坤语词汇并进行分类
    """
    # 初始化LLM客户端
    client = LLMClient()
    
    # 构建prompt
    prompt = f"""以下是哈坤语与中文的对照例句：
                {content}

                任务：
                - 列出所有哈坤语词汇，并为每个词汇标注其可能的类别：
                - 主语（如"我/你/他/我们/你们/他们"）
                - 动词（如"睡/看见/知道/打"）
                - 时态标记（如过去时、现在时）
                - 疑问标记（如"吗"）
                - 其他（无法分类的词汇）

                输出格式：
                {{哈坤词汇: 类别（中文词汇）}}"""
                    
    messages = [
        {"role": "system", "content": "你是一个语言学专家，擅长分析不同语言的语法结构和词汇分类。"},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat_completion(messages=messages, temperature=0.2)
    result = response.choices[0].message.content

    return {"raw_result": result}

def analyze_1(content):
    """
    根据哈坤语词汇分类，给出合并同类项后的结构化规则
    """
    # 初始化LLM客户端
    client = LLMClient()
    
    # 构建prompt
    prompt = f"""以下是哈坤语词汇分类：
                    {content}

                    任务：
                    - 规则泛化（合并同类项），将相同属性的词汇进行合并。
                    - 生成结构化映射表（JSON格式），基于泛化规则，生成包含以下键的JSON：  
                    - `subjects`: 主语（区分时态）  
                    - `verbs`: 动词  
                    - `tenses`: 时态标记  
                    - `question_marker`: 疑问标记  
                    - 其他必要类别（如连接词）
        

                    输出格式：
                    只输出最终生成的json格式，例如：
                    {{
                        "键": {{
                            "哈坤语": "汉语"
                        }},
                    }}
                    """
                    
    messages = [
        {"role": "system", "content": "你是一个语言学专家，擅长分析不同语言的语法结构和词汇分类。"},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat_completion(messages=messages, temperature=0.2)
    result = response.choices[0].message.content

    return {"raw_result": result}

def encode(sentence,rules):
    """
    将输入的哈坤语转换为token形式并给出对应token的属性
    """
    # 初始化LLM客户端
    client = LLMClient()
    
    # 构建prompt
    prompt = f"""请将以下哈坤语句子分解为语法单元，并标记每个成分的角色（主语、动词、时态、疑问标记等）：  
                **输入句子**： {sentence}
                **已知规则**： {rules} 
                **输出格式**：  
                - Tokens: [列表形式的分词结果]  
                - Roles: [与Token对应的属性列表]"""
                    
    messages = [
        {"role": "system", "content": "你是一个语言学专家，擅长分析不同语言的语法结构和词汇分类。"},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat_completion(messages=messages, temperature=0.2)
    result = response.choices[0].message.content

    return {"raw_result": result}

def decode(tokens,rules):
    """
    将输入的哈坤语token进行同态映射到中文，并按中文语法重组为通顺的句子
    """
    # 初始化LLM客户端
    client = LLMClient()
    
    # 构建prompt
    prompt = f"""请将以下哈坤语token同态映射到中文，并按中文语法重组为通顺的句子： 
                **原始token**： {tokens}
                **已知规则**： {rules} 
                只输出最终的汉语句子。
                """
                    
    messages = [
        {"role": "system", "content": "你是一个语言学专家，擅长分析不同语言的语法结构和词汇分类。"},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat_completion(messages=messages, temperature=0.2)
    result = response.choices[0].message.content

    return {"raw_result": result}