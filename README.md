# LLM for linguistics

## 项目背景

本项目旨在利用大型语言模型(LLM)技术解决国际语言学奥林匹克问题。政治课本曾提到过，文字是语言的符号，我们的目标是寻找两个符号集合之间的映射关系。我们希望通过LLM技术，引导llm模型自动生成哈坤语的语法规则，以解决国际语言学奥林匹克竞赛中的语法问题。同时借鉴密码学中同态映射的概念，使得llm能更好的理解和处理文本信息。

## 项目功能

本项目主要实现以下功能：

1. **词汇分析与分类**：自动识别句子中的词汇，并将其分类为各种语法成分（主语、动词、时态标记等）。

2. **语法规则提取与结构化**：基于词汇分类结果，自动生成哈坤语的结构化语法规则，以JSON格式呈现。

3. **编码与解码**：借鉴密码学中同态映射的概念，将句子转换为结构化的token形式，实现不同符号集合间的映射。

4. **知识增强检索(RAG)**：利用向量数据库技术，实现对哈坤语知识库的语义检索，为LLM提供更准确的上下文信息。

5. **自动验证与优化**：通过验证模块自动检查翻译结果，并根据反馈进行针对性优化。

## 技术架构

项目基于以下技术：

- **LLM接口**：通过`llm_client.py`封装对大型语言模型API的调用，支持多个LLM服务商
- **语言分析模块**：`LLM_function.py`中实现了核心的语言分析功能
- **推理链模块**：`reasoning.py`实现了完整的语言分析推理链
- **知识库模块**：`knowledge_base.py`存储哈坤语的语法规则和词汇信息
- **向量检索**：使用FAISS实现高效的语义相似度搜索

### 核心流程

1. **知识准备**：
   - 初始化知识库
   - 构建向量存储
   - 配置LLM服务

2. **分析流程**：
   - 词汇提取与分类
   - 规则生成与结构化
   - Token编码与解码
   - 结果验证与优化

## 使用方法

### 环境准备

1. 确保已安装Python 3.6+
2. 安装依赖：
```bash
pip install openai langchain langchain-openai faiss-cpu
```

### 配置说明

在 llm_client.py 中配置您的API密钥：

```python
ZHIZENGZENG_API_KEY = "your-api-key"
DEEPSEEK_API_KEY = "your-api-key"
OPENAI_API_KEY = "your-api-key"
QWEN_API_KEY = "your-api-key"
 ```

### 运行示例

```bash
python main.py
```
## 项目特点
1. 多级验证机制 ：
   
   - 词汇映射验证
   - 语法结构验证
   - 语义完整性验证
2. 智能重试机制 ：
   
   - 根据验证结果自动定位问题
   - 针对性优化失败步骤
   - 保留有效分析结果
3. 知识增强 ：
   
   - 语法规则库
   - 词汇分类库
   - 示例分析库

## 应用场景
- 国际语言学奥林匹克竞赛(IOLC)题目解析
- 语言学研究中的语法规则提取
- 跨语言文本分析与处理
- 语言教学辅助工具

## 未来展望
- 支持更多种语言学竞赛题型
- 优化RAG检索效果，提高知识利用率
- 扩展知识库覆盖范围，支持更多语言特征
- 添加可视化界面，提升用户体验
- 实现批量处理功能
- 优化重试策略，提高成功率