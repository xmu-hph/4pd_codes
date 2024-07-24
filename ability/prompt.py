class ZhPrompt:
    choice_prompt_format_v1 = """你正在进行一场{language}考试。下面是一道单选题，该问题只有一个正确答案。

{question}

请你仔细地分析每个选项内容的意思，确定每个选项的内容与题目要求的匹配程度，分析过程保留在草稿中，匹配度最高的就是正确的选项。最后请你指出正确的选项，不用说明选项的内容，直接返回正确的选项即可，例如正确的选项是：A。"""
    
    writing_prompt_format_v1 = """你是一位母语为{language}的高级知识分子。请你用{language}回答下面的问题。

{question}

你的回答必须满足以下条件:
1. 不超过100字
2. 关键词中心思想明确
3. 没有语法错误和逻辑错误
4. 段落间应有衔接
你必须完全用{language}作答。"""


class EnPrompt:
    choice_prompt_format_v1 = """You are in a language test of {language}. Please find the only correct option in the question.

{question}

The correct option must be both grammatically and logically correct. You must only return the letter and the original content of the correct option, without any extra explanation."""

    writing_prompt_format_v1 = """You are in a language test of {language}. Please write down your answer to the question.

{question}

You must write your answer completely in {language}, like a native speaker."""