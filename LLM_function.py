from llm_client import LLMClient
from knowledge_base import KnowledgeBase
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class RAG:
    def __init__(self):
        self.kb = KnowledgeBase()
        # 使用已配置的API key
        llm_client = LLMClient()
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=llm_client.api_key,
            base_url=llm_client.base_url
        )
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # 构建知识库向量存储
        knowledge_texts = []
        # 添加语法规则
        knowledge_texts.extend(self.kb.get_grammar_rules())
        
        # 添加词汇分类
        for category in ['subjects', 'verbs', 'tense_markers', 'question_markers', 'connectors']:
            vocab = self.kb.get_vocabulary_by_category(category)
            for item in vocab:
                text = f"{item['token']}: {item['meaning']} ({category}, {item['notes']})"
                knowledge_texts.append(text)
        
        # 添加分析示例
        for example in self.kb.get_analysis_examples():
            text = f"例句：{example['sentence']}\n翻译：{example['translation']}\n分析：{example['analysis']}"
            knowledge_texts.append(text)
            
        # 创建文档chunks
        docs = self.text_splitter.create_documents(knowledge_texts)
        
        # 构建向量存储
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
    
    def get_relevant_context(self, query, k=3):
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])

def extract(content):
    """
    分析哈坤语和汉语对照内容，提取哈坤语词汇并进行分类
    """
    # 初始化LLM客户端和RAG
    client = LLMClient()
    rag = RAG()
    
    # 获取相关上下文
    relevant_context = rag.get_relevant_context(content)

    prompt = f"""以下是哈坤语与中文的对照例句：
                {content}

                基于以下相关的哈坤语知识：
                {relevant_context}
                
                任务：
                - 列出所有哈坤语词汇，并为每个词汇标注其可能的类别：
                - 主语（如"我/你/他/我们/你们/他们"）
                - 动词（如"睡/看见/知道/打"）
                - 时态标记（如过去时、现在时）
                - 疑问标记（如"吗"）
                - 其他（无法分类的词汇）

                输出格式：
                哈坤词汇: 类别（中文词汇）"""
                    
    messages = [
        {"role": "system", "content": "你是一位专精于语言分析的语言学家，擅长识别和分类语言中的词汇单元。你需要仔细分析每个词汇的语法功能和语义角色。"},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat_completion(messages=messages, temperature=0.2)
    result = response.choices[0].message.content

    return {"raw_result": result}

def generate_rules(content):
    """
    根据哈坤语词汇分类，给出合并同类项后的结构化规则
    """
    # 初始化LLM客户端
    client = LLMClient()
    
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
        {"role": "system", "content": "你是一位语言规则专家，擅长归纳和总结语言规律，能够将具体语言现象抽象为通用规则。你需要系统地组织和结构化语言规则。"},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat_completion(messages=messages, temperature=0.2)
    result = response.choices[0].message.content

    return {"raw_result": result}

def encode(sentence, rules):
    """
    将输入的哈坤语转换为token形式并给出对应token的属性
    """
    # 初始化LLM客户端
    client = LLMClient()

    prompt = f"""请将以下哈坤语句子分解为语法单元，并标记每个成分的角色（主语、动词、时态、疑问标记等）：  
                **输入句子**： {sentence}
                **已知规则**： {rules}
                **输出格式**：  
                - Tokens: [列表形式的分词结果]  
                - Roles: [与Token对应的属性列表]
                无需给出分析过程与其他多余部分"""
                    
    messages = [
        {"role": "system", "content": "你是一位计算语言学专家，专注于语言单元的分析和标注。你需要准确识别每个语言成分的语法角色，并进行规范化的标记。"},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat_completion(messages=messages, temperature=0.2)
    result = response.choices[0].message.content

    return {"raw_result": result}

def decode(tokens, rules):
    """
    将输入的哈坤语token进行同态映射到中文，并按中文语法重组为通顺的句子
    """
    # 初始化LLM客户端
    client = LLMClient()
    

    prompt = f"""请将以下哈坤语token同态映射到中文，并按中文语法重组为通顺的句子： 
                **原始token**： {tokens}
                **已知规则**： {rules}
                只输出最终的汉语句子。
                """
                    
    messages = [
        {"role": "system", "content": "你是一位语言转换专家，精通语言间的映射转换。你需要在保持原文语义的基础上，生成符合目标语言语法规范的表达。"},
        {"role": "user", "content": prompt}
    ]

    response = client.chat_completion(messages=messages, temperature=0.2)
    result = response.choices[0].message.content

    return {"raw_result": result}

def verify(secret_text, known_text, rules, origin_rules):
    """
    验证哈坤语到汉语的翻译是否正确，并提供详细的推理过程和下一步行动建议
    """
    client = LLMClient()
    
    result_template = '''
    {   
        "is_true": "true/false",
        "action": "output/extract/generate_rules/encode/decode",
        "analysis": {
            "vocabulary_check": "词汇映射分析",
            "grammar_check": "语法结构分析",
            "semantic_check": "语义完整性分析"
        },
        "reasoning": "详细的推理过程",
        "suggestions": "优化建议",
        "input_for_retry": {
            "content": "如果需要重新推理，这里是需要重点关注的内容，以文本格式给出即可"
        }
    }
    '''
       
    prompt = f"""请验证以下哈坤语翻译是否正确，并给出下一步行动建议：
                原始的规则：{origin_rules}
                经过推理后提取的语法规则：{rules}
                需要翻译的哈坤语原文：{secret_text}
                经过推理后的汉语译文：{known_text}

                翻译正确的标准：

                1. 语法结构基本对应：
                - 主要句子成分的顺序正确
                - 允许有轻微的语序调整以符合中文表达习惯

                2. 语义表达基本准确：
                - 核心语义传达正确
                - 允许有细微的表达差异，但不影响整体理解
                - 疑问语气表达清晰
                
                3. 对于规则中没有出现的词汇，推测合理即可，无需重新推理。

                行动建议：
                如果根据现有的规则翻译正确：
                - 即可返回is_true为"true"
                - 输出 "output" 作为 action
                如果翻译不正确，根据推理结果给出下一步行动建议：
                - 如果需要重新提取词汇（extract）：
                  在input_for_retry中详细说明哪些词汇需要重新分析，以及可能的词性或语法功能
                
                - 如果需要重新生成规则（generate_rules）：
                  在input_for_retry中指出当前规则的具体问题，如某类词汇的映射规则缺失或不准确
                
                - 如果需要重新进行编码（encode）：
                  在input_for_retry中说明token分解或角色标注的具体问题
                
                - 如果需要重新进行解码（decode）：
                  在input_for_retry中指出当前翻译中的具体语序或语义问题

                对于每种情况，input_for_retry都应该：
                1. 明确指出当前步骤的具体问题
                2. 提供可能的解决方向
                3. 给出具体的例子说明问题
                4. 避免模糊不清的表述

                输出格式：{result_template}"""
    
    messages = [
        {"role": "system", "content": "你是一位语言质量评估专家，擅长分析翻译的准确性和完整性。你需要从词汇、语法和语义三个层面严格评估翻译质量，并提供具体的改进建议。"},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat_completion(messages=messages, temperature=0.2)
    result = response.choices[0].message.content
    
    return result


def adjust(action, retry_content, original_input=None, original_output=None):
    """
    根据verify的结果和上一轮的输入输出，生成优化后的结果
    Args:
        action: verify返回的action类型
        retry_content: verify中input_for_retry的内容
        original_input: 上一轮该步骤的输入内容
        original_output: 上一轮该步骤的输出内容
    """
    client = LLMClient()
    
    prompts = {
        "extract": f"""基于上一轮的分析结果，请重新分析哈坤语词汇：
                    
                    原始规则：
                    {original_input}

                    上一轮分析结果：
                    {original_output}

                    需要特别关注的问题：
                    {retry_content}

                    请对上述分析结果进行调整，重点关注提到的问题。
                    保持合理的分类结果，调整存在问题的部分。
                    
                    输出格式：
                    哈坤词汇: 类别（中文词汇）""",
                    
        "generate_rules": f"""基于上一轮生成的规则，进行优化调整：

                    原始规则：
                    {original_input}

                    上一轮规则：
                    {original_output}

                    需要调整的问题：
                    {retry_content}

                    请对规则进行针对性修改，确保解决提到的问题。
                    
                    输出格式：
                    只输出JSON格式的规则映射表。""",
                    
        "encode": f"""基于上一轮的编码结果，进行优化：

                    上一轮输入：{original_input}
                    上一轮结果：{original_output}

                    需要调整的问题：
                    {retry_content}

                    请对token分解和角色标注进行调整，重点解决提到的问题。
                    
                    输出格式：
                    - Tokens: [列表形式的分词结果]
                    - Roles: [与Token对应的属性列表]""",
                    
        "decode": f"""基于上一轮的翻译结果，进行优化：

                    上一轮输入：{original_input}
                    上一轮翻译：{original_output}

                    需要调整的问题：
                    {retry_content}

                    请对翻译进行针对性修改，确保解决语序或语义问题。
                    
                    只输出最终的汉语句子。"""
    }
    
    if action not in prompts:
        return {"error": "未知的action类型"}
        
    messages = [
        {"role": "system", "content": "你是一位语言分析专家，需要根据具体反馈优化已有的分析结果。请保持合理的部分，针对性地调整存在问题的部分。"},
        {"role": "user", "content": prompts[action]}
    ]
    
    response = client.chat_completion(messages=messages, temperature=0.2)
    result = response.choices[0].message.content
    
    return {"raw_result": result}

