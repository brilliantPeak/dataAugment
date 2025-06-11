# -*- coding: utf-8 -*-
import json
import re

def create_new_dialogues(original_text, num_new_dialogues=3):
    """
    从单行原始对话文本中创建新的多轮对话
    返回一个列表，包含num_new_dialogues个新的完整对话
    """
    # 提取原始User提问
    # 使用正则表达式匹配所有"User:"后的内容，直到遇到"Assistant:"
    pattern = r'User: (.*?)(?=Assistant:|$)'
    original_user_questions = re.findall(pattern, original_text)

    # 为每个User提问生成增强版本
    all_augmented_questions=[
    # 原始提问1: "狗狗肚子咕咕叫还吐白色泡沫"
    ["请问狗狗肚子咕咕叫还吐白色泡沫", "狗狗肚子咕咕叫还吐白色泡沫，可以详细说明一下吗？"],

    # 原始提问2: "当然可以。"
    ["请问当然可以。", "当然可以。，可以详细说明一下吗？"],

    # 原始提问3: "它是一只金毛。"
    ["请问它是一只金毛。", "它是一只金毛。，可以详细说明一下吗？"],

    # 原始提问4: "三岁。"
    ["请问三岁。", "三岁。，可以详细说明一下吗？"],

    # 原始提问5: "是公狗。"
    ["请问是公狗。", "是公狗。，可以详细说明一下吗？"],

    # 原始提问6: "没有绝育。"
    ["请问没有绝育。", "没有绝育。，可以详细说明一下吗？"],

    # 原始提问7: "所有的常规疫苗都按时注射了。"
    ["请问所有的常规疫苗都按时注射了。", "所有的常规疫苗都按时注射了。，可以详细说明一下吗？"],

    # 原始提问8: "驱虫按时进行，上个月刚做过。"
    ["请问驱虫按时进行，上个月刚做过。", "驱虫按时进行，上个月刚做过。，可以详细说明一下吗？"],

    # 原始提问9: "最近没有太大变化，饮食还是和以前一样。"
    ["请问最近没有太大变化，饮食还是和以前一样。", "最近没有太大变化，饮食还是和以前一样。，可以详细说明一下吗？"],

    # 原始提问10: "没有什么改变，还是住在原来的环境里。"
    ["请问没有什么改变，还是住在原来的环境里。", "没有什么改变，还是住在原来的环境里。，可以详细说明一下吗？"],

    # 原始提问11: "食欲有点减退，经常不怎么吃东西。"
    ["请问食欲有点减退，经常不怎么吃东西。", "食欲有点减退，经常不怎么吃东西。，可以详细说明一下吗？"],

    # 原始提问12: "有点不太活泼，经常安静地躺着。"
    ["请问有点不太活泼，经常安静地躺着。", "有点不太活泼，经常安静地躺着。，可以详细说明一下吗？"],

    # 原始提问13: "好像喝水比平时多了一些。"
    ["请问好像喝水比平时多了一些。", "好像喝水比平时多了一些。，可以详细说明一下吗？"],

    # 原始提问14: "有点腹泻，便便比较稀。"
    ["请问有点腹泻，便便比较稀。", "有点腹泻，便便比较稀。，可以详细说明一下吗？"],

    # 原始提问15: "昨天测量过，好像有点发热。"
    ["请问昨天测量过，好像有点发热。", "昨天测量过，好像有点发热。，可以详细说明一下吗？"],

    # 原始提问16: "皮肤弹性还可以，好像没有脱水的现象。"
    ["请问皮肤弹性还可以，好像没有脱水的现象。", "皮肤弹性还可以，好像没有脱水的现象。，可以详细说明一下吗？"]
]
    # 生成新的对话
    new_dialogues = []
    for i in range(num_new_dialogues):
        # 构建新的对话文本
        new_dialogue = original_text
        for j in range(len(original_user_questions)):
            # 选择第i个增强版本（循环使用）
            augmentation_index = i
            augmented_question = all_augmented_questions[j][augmentation_index]

            # 替换User提问
            # 使用正则表达式替换第j个User提问
            pattern = rf'(User: {original_user_questions[j]})(.*?)(Assistant: )'
            replacement = rf'\1{augmented_question}\3'
            new_dialogue = re.sub(pattern, replacement, new_dialogue, count=1)

        new_dialogues.append(new_dialogue)

    return new_dialogues

def extract_user_questions_from_single_line(text):
    """
    从单行文本中提取所有User提问（通过"User:"和"Assistant:"标签分隔）
    通过User和Assistant交互拼接组装对话
    """
    # 使用正则表达式匹配所有"User:"后的内容，直到遇到"Assistant:"
    pattern = r'User: (.*?)(?=Assistant:|$)'
    user_questions = re.findall(pattern, text)

    assistant_pattern = r'Assistant: (.*?)(?=User:|$)'
    assistant_questions = re.findall(assistant_pattern, text)

    if len(assistant_questions)<len(user_questions):
        last_assistant=text.split("Assistant:")[-1]
        assistant_questions.append(last_assistant)

    re_content=""
    for user,assistant in  zip(user_questions,assistant_questions):
        re_content +=f"User:{user} \nAssistant:{assistant}\n"

    return re_content


import  pandas as pd
if __name__ == '__main__':
    df = pd.read_csv("data/enhance_intention_20250604.csv",encoding='gbk')
    transList=[]
    for index, row in df.iterrows():
        print(f"index:{index}  row:{row} ")

        transcontent=extract_user_questions_from_single_line(row["representative"])

        diff_rep = {
            'cluster': row["cluster"],
            'content': row["representative"],
            'transcontent':transcontent,
            'similarity': row["similarity"],
            'type': row["type"]
        }
        transList.append(diff_rep)
    df=pd.DataFrame(transList)
    df.to_csv("data/enhance_intention_20250606.csv",mode='a', index=False,header=False,encoding='utf-8-sig')


