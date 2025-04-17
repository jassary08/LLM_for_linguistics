from reasoning import *
import json

def main():
    # 哈坤语和汉语对照内容
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

    test_sentence = "nɤ ʒip ku ne"

    print("\n=== 哈坤语翻译推理链 ===")
    print(f"待翻译句子: {test_sentence}")
    
    # 进行推理过程
    result = ReasoningChain(harvard_chinese_content, test_sentence, max_retries=3)
    
    if result["success"]:
        print("\n=== 成功 ===")
        print(f"最终译文: {result['translation']}")
        print("\n处理过程:")
        for step, content in result["process_history"].items():
            print(f"\n{step}步骤结果:")
            if isinstance(content, dict):
                print(json.dumps(content, ensure_ascii=False, indent=2))
            else:
                print(content)
    else:
        print("\n=== 失败 ===")
        print(f"失败步骤: {result['failed_step']}")
        print(f"错误信息: {result['error']}")
        print("\n已完成的步骤:")
        for step, content in result["process_history"].items():
            print(f"\n{step}:")
            if isinstance(content, dict):
                print(json.dumps(content, ensure_ascii=False, indent=2))
            else:
                print(content)

if __name__ == "__main__":
    main()