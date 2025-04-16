# 哈坤语知识库

class KnowledgeBase:
    def __init__(self):
        # 初始化哈坤语知识库
        self.knowledge = {
            # 语法规则
            "grammar_rules": [
                "哈坤语疑问句通常以'ne'结尾",
                "哈坤语中主语通常位于句首",
                "哈坤语中动词通常位于主语之后",
                "哈坤语中时态标记通常附加在动词之后",
                "哈坤语中'kəmə'通常用于连接主语和宾语"
            ],
            
            # 词汇分类知识
            "vocabulary_categories": {
                "subjects": [
                    {"token": "ŋa", "meaning": "我", "notes": "第一人称单数"},
                    {"token": "nɤ", "meaning": "你", "notes": "第二人称单数"},
                    {"token": "ati", "meaning": "他", "notes": "第三人称单数"},
                    {"token": "nirum", "meaning": "我们", "notes": "第一人称复数"},
                    {"token": "nuʔrum", "meaning": "你们", "notes": "第二人称复数"},
                    {"token": "tarum", "meaning": "他们", "notes": "第三人称复数"}
                ],
                "verbs": [
                    {"token": "ka", "meaning": "走", "notes": "移动动词"},
                    {"token": "ʒip", "meaning": "睡", "notes": "状态动词"},
                    {"token": "lapkʰi", "meaning": "看见", "notes": "感知动词"},
                    {"token": "cʰam", "meaning": "知道", "notes": "认知动词"},
                    {"token": "lan", "meaning": "打", "notes": "动作动词"}
                ],
                "tense_markers": [
                    {"token": "tuʔ", "meaning": "过去时", "notes": "用于表示过去发生的动作"},
                    {"token": "tɤʔ", "meaning": "过去时", "notes": "与第一人称单数搭配"},
                    {"token": "tʰu", "meaning": "过去时", "notes": "与第三人称复数搭配"},
                    {"token": "ku", "meaning": "现在时", "notes": "用于表示当前发生的动作"},
                    {"token": "kɤ", "meaning": "现在时", "notes": "与第一人称单数搭配"},
                    {"token": "ki", "meaning": "现在时", "notes": "与第一人称复数搭配"},
                    {"token": "kan", "meaning": "现在时", "notes": "与第二人称复数搭配"},
                    {"token": "ri", "meaning": "现在时", "notes": "与第三人称复数搭配"},
                    {"token": "rɤ", "meaning": "现在时", "notes": "与第二人称单数搭配"},
                    {"token": "tʰɤ", "meaning": "过去时", "notes": "与第三人称单数搭配"}
                ],
                "question_markers": [
                    {"token": "ne", "meaning": "吗", "notes": "疑问标记，通常位于句尾"}
                ],
                "connectors": [
                    {"token": "kəmə", "meaning": "[连接词]", "notes": "用于连接主语和宾语"},
                    {"token": "bə", "meaning": "[连接词]", "notes": "用于连接主语和谓语"}
                ]
            },
            
            # 语言学分析示例
            "analysis_examples": [
                {
                    "sentence": "ŋa ka kɤ ne",
                    "translation": "我走吗?",
                    "analysis": "ŋa(我,主语) + ka(走,动词) + kɤ(现在时,时态标记) + ne(吗,疑问标记)"
                },
                {
                    "sentence": "nɤ ʒip tuʔ ne",
                    "translation": "你（过去时）睡吗?",
                    "analysis": "nɤ(你,主语) + ʒip(睡,动词) + tuʔ(过去时,时态标记) + ne(吗,疑问标记)"
                },
                {
                    "sentence": "tarum kəmə nɤ lan tʰu ne",
                    "translation": "他们（过去时）打你吗?",
                    "analysis": "tarum(他们,主语) + kəmə(连接词) + nɤ(你,宾语) + lan(打,动词) + tʰu(过去时,时态标记) + ne(吗,疑问标记)"
                }
            ]
        }
    
    def get_grammar_rules(self):
        return self.knowledge["grammar_rules"]
    
    def get_vocabulary_by_category(self, category):
        if category in self.knowledge["vocabulary_categories"]:
            return self.knowledge["vocabulary_categories"][category]
        return []
    
    def get_analysis_examples(self):
        return self.knowledge["analysis_examples"]
    
    