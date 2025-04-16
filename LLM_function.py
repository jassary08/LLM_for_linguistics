from llm_client import LLMClient
from knowledge_base import KnowledgeBase
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
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
    print(relevant_context)
    # 构建prompt
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
        {"role": "system", "content": "你是一个语言学专家，擅长分析不同语言的语法结构和词汇分类。"},
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

def verify(hakun_text, chinese_text, rules):
    """
    验证哈坤语到汉语的翻译是否正确，并提供详细的推理过程
    """
    client = LLMClient()
    
    # 使用 json 字符串作为模板
    result_template = '''
    {
        "is_correct": true/false,
        "analysis": {
            "vocabulary_check": "词汇映射分析",
            "grammar_check": "语法结构分析",
            "semantic_check": "语义完整性分析"
        },
        "reasoning": "详细的推理过程",
        "suggestions": "如果有错误，给出修正建议"
    }
    '''
       
    prompt = f"""请验证以下哈坤语翻译是否正确：
                哈坤语原文：{hakun_text}
                汉语译文：{chinese_text}
                语法规则：{rules}

                请按以下步骤进行分析：
                1. 词汇映射验证：
                - 检查每个哈坤语词汇是否正确对应到汉语
                - 标注任何不匹配或可疑的映射

                2. 语法结构验证：
                - 分析哈坤语的语法结构
                - 验证汉语译文是否保持了相同的语法关系

                3. 语义完整性：
                - 确认译文是否完整传达了原文的所有信息
                - 检查是否有遗漏或添加的信息

                4. 推理过程：
                - 详细说明如何得出验证结论
                - 引用相关的语法规则支持你的判断

                输出格式：{result_template}"""
    
    messages = [
        {"role": "system", "content": "你是一个严谨的语言学专家，擅长分析和验证语言翻译的准确性。"},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat_completion(messages=messages, temperature=0.2)
    result = response.choices[0].message.content
    
    return {"verification_result": result}