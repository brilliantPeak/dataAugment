
def generate_few_shot_prompts(intents, examples):
    """
    生成多意图场景下的few-shot文本增强提示词

    参数:
        intents: list[str] - 意图列表
        examples: dict[str: list[str]] - 各意图对应的示例文本

    返回:
        list[str] - 生成的提示词列表
    """
    prompts = []
    for intent in intents:
        for example in examples.get(intent, []):
            prompt = f"""
            意图: {intent}
            示例: {example}
            增强要求: 保持原意图不变，生成3个语义相似但表达不同的文本
            
            增强结果:
            1. 
            2. 
            3.
            """
            prompts.append(prompt)
    return prompts