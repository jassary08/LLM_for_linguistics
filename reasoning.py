from llm_function import extract, generate_rules, encode, decode, verify
import json

def ReasoningChain(origin_rules,secret_text, max_retries=3):
    retry_count = 0
    current_step = "extract"
    result_history = {}
    retry_key = None
    
    while retry_count < max_retries:
        if current_step == "extract":
            # 步骤1: 分析哈坤语词汇并进行分类
            analysis_result = extract(origin_rules,retry_key)
            result_history["extract"] = analysis_result
            current_step = "generate_rules"
            
        elif current_step == "generate_rules":
            # 步骤2: 根据词汇分类，生成结构化规则
            rules_result = generate_rules(result_history["extract"]["raw_result"], retry_key)
            result_history["rules"] = rules_result
            current_step = "encode"
            
        elif current_step == "encode":
            # 步骤3: 将哈坤语句子转换为token
            encode_result = encode(secret_text, result_history["rules"]["raw_result"], retry_key)
            result_history["encode"] = encode_result
            current_step = "decode"
            
        elif current_step == "decode":
            # 步骤4: 将token转换为中文句子
            decode_result = decode(
                result_history["encode"]["raw_result"],
                result_history["rules"]["raw_result"],
                retry_key
            )
            result_history["decode"] = decode_result
            current_step = "verify"
            
        elif current_step == "verify":
            # 步骤5: 验证翻译结果
            verification_result = verify(
                secret_text=secret_text,
                known_text=result_history["decode"]["raw_result"],
                rules=result_history["rules"]["raw_result"],
                origin_rules=origin_rules,
            )
            print(verification_result)
            
            if verification_result.startswith("```json"):
                verification_result = verification_result[7:]  # 移除开头的```json
            if verification_result.endswith("```"):
                verification_result = verification_result[:-3]  # 移除结尾的```
                

            verify_data = json.loads(verification_result)
            result_history["verify"] = verify_data
            
            # 根据验证结果决定下一步操作
            if verify_data["is_true"] == "true":
                return {
                    "success": True,
                    "translation": result_history["decode"]["raw_result"],
                    "process_history": result_history
                }
            else:
                # 如果需要重试，更新当前步骤和输入
                current_step = verify_data["action"]
                if "input_for_retry" in verify_data:
                    retry_key = verify_data["input_for_retry"]["content"]
                retry_count += 1
    
    return {
        "success": False,
        "error": "达到最大重试次数",
        "failed_step": current_step,
        "process_history": result_history
    }