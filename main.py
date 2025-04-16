from LLM_function import extract, generate_rules, encode, decode, verify
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

    print("\n1. 原始哈坤语和汉语对照内容:")
    print(harvard_chinese_content)
    
    # 步骤1: 分析哈坤语词汇并进行分类
    print("\n2. 分析哈坤语词汇并进行分类:")
    analysis_result = extract(harvard_chinese_content)
    print(analysis_result["raw_result"])
    
    # 步骤2: 根据词汇分类，给出结构化规则
    print("\n3. 根据词汇分类，给出结构化规则:")
    rules_result = generate_rules(analysis_result["raw_result"])
    print(rules_result["raw_result"])

    rules_json = rules_result["raw_result"]
    
    # 步骤3: 测试encode函数 - 将哈坤语句子转换为token
    test_sentence = "nɤ ʒip ku ne"
    print(f"\n4. 测试encode函数 - 将哈坤语句子 '{test_sentence}' 转换为token:")
    encode_result = encode(test_sentence, rules_json)
    print(encode_result["raw_result"])
    
    # 步骤4: 测试decode函数 - 将token转换为中文句子
    tokens_text = encode_result["raw_result"]
    print("\n5. 测试decode函数 - 将token转换为中文句子:")
    decode_result = decode(tokens_text, rules_json)
    print(decode_result["raw_result"])
    
    # 步骤5: 验证翻译结果
    print("\n6. 验证翻译结果:")
    verification_result = verify(
        hakun_text=test_sentence,
        chinese_text=decode_result["raw_result"],
        rules=rules_json
    )
    print("验证结果:")
    print(json.dumps(verification_result["verification_result"], 
                    ensure_ascii=False, 
                    indent=2))

if __name__ == "__main__":
    main()