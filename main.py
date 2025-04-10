from LLM_function import analyze_0, analyze_1, encode, decode
import json

def main():
    # 示例哈坤语和汉语对照内容
    harvard_chinese_content = """
    1. ŋa ka kɤ ne 我走吗?
    2. nɤ ʒip tuʔ ne 你（过去时）睡吗?
    3. ŋabə ati lapkʰi tɤʔ ne 我（过去时）看见他吗?
    4. nirum kəmə nuʔrum cʰam ki ne 我们知道你们吗?
    5. nɤbə ŋa lapkʰi rɤ ne 你看见我吗?
    6. tarum kəmə nɤ lan tʰu ne 他们（过去时）打你吗?
    7. nuʔrum kəmə ati lapkʰi kan ne 你们看见他吗?
    8. nɤbə ati cʰam tuʔ ne 你（过去时）知道他吗?
    9. tarum kəmə nirum lapkʰi ri ne 他们看见我们吗?
    10. ati kəmə ŋa lapkʰi tʰɤ ne 他（过去时）看见我吗?
    """
    
    print("=== 测试哈坤语分析流程 ===")
    print("\n1. 原始哈坤语和汉语对照内容:")
    print(harvard_chinese_content)
    
    # 步骤1: 分析哈坤语词汇并进行分类
    print("\n2. 分析哈坤语词汇并进行分类:")
    analysis_result = analyze_0(harvard_chinese_content)
    print(analysis_result["raw_result"])
    
    # 步骤2: 根据词汇分类，给出结构化规则
    print("\n3. 根据词汇分类，给出结构化规则:")
    rules_result = analyze_1(analysis_result["raw_result"])
    print(rules_result["raw_result"])

    rules_json = rules_result["raw_result"]
    
    # 步骤3: 测试encode函数 - 将哈坤语句子转换为token
    test_sentence = "nɤ ʒip ku ne"
    print(f"\n4. 测试encode函数 - 将哈坤语句子 '{test_sentence}' 转换为token:")
    encode_result = encode(test_sentence, rules_json)
    print(encode_result["raw_result"])
    
    # 步骤4: 测试decode函数 - 将token转换为中文句子
    # 从encode结果中提取tokens
    tokens_text = encode_result["raw_result"]
    print("\n5. 测试decode函数 - 将token转换为中文句子:")
    decode_result = decode(tokens_text, rules_json)
    print(decode_result["raw_result"])

if __name__ == "__main__":
    main()