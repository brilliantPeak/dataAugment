# -*- coding: utf-8 -*-

import json
import logging
import time
import re
import  csv
import pandas as pd
import requests
import ast
from  user_prompt_template import  intention_prompt_template,noise_prompt_template
#  pip install 包名 -i https://pypi.tuna.tsinghua.edu.cn/simple
def construct_prompt(his: str, inputx: str) -> str:
    his_list= list(json.loads(his))
    his_list.append({"role" :"user" ,"content" :inputx})
    prompt = {"model": "deepseek-v1",
              "temperature": 0.2,   #temperature 的参数值越小，模型就会返回越确定的一个结果
              "max_length": 10000,
              "chat_template_kwargs": {"enable_thinking": False}
              }
    msgs = [{"role": "system",
             "content": """你是一名经验丰富的兽医，在犬猫的诊断、治疗和预防等方面有着多年临床经验。熟悉各类犬猫常见疾病，
             包括传染病、寄生虫病、内科疾病、外科疾病等，能够为犬猫主人提供专业的医疗建议和指导。当前为多轮对话数据增强背景
             ，请根据你的职业经验从用户提问意图多样化，模糊口语化，错别字&噪声提问三个角度对用户侧的数据进行增强。"""},
            {
                "role": "user", "content": json.dumps(his_list, ensure_ascii=False)
            }]

    prompt['messages'] = msgs
    return json.dumps(prompt, ensure_ascii=False)

def build_request(prompt):
    url = "http://172.31.142.158:38080/v1/chat-messages"
    payload = {
        "inputs": {},
        "query": prompt,
        "response_mode": "blocking",
        "conversation_id": "",
        "user": "liuwei",
        "files": []
    }

    payload=json.dumps(payload,ensure_ascii=False)

    headers = {
        'Authorization': 'Bearer app-saf8TcQnDsKpQu03WiUMBq3U',
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response

def extract_user_questions_from_single_line(text):
    """
    从单行文本中提取所有User提问（通过"User:"和"Assistant:"标签分隔）
    """
    # 使用正则表达式匹配所有"User:"后的内容，直到遇到"Assistant:"
    pattern = r'User: (.*?)(?=Assistant:|$)'
    user_questions = re.findall(pattern, text)
    return user_questions

def create_new_dialogues(row_text, num_new_dialogues=3):
    """
    从单行原始对话文本中创建新的多轮对话
    返回一个列表，包含num_new_dialogues个新的完整对话
    20250611
    """
    # 提取原始User提问
    original_text=row_text["representative"]
    original_user_questions = extract_user_questions_from_single_line(original_text)

    # 为每个User提问生成增强版本

    format_prompt = noise_prompt_template % original_user_questions

    history = []
    his_json = json.dumps(history, ensure_ascii=False)
    prompt = construct_prompt(his_json, format_prompt)

    response = build_request(prompt)
    res_answer=json.loads(response.text)["answer"]

    cleaned_str = res_answer.strip("```json").strip("```")  if res_answer.startswith("```json") and res_answer.endswith("```") else res_answer
    try:
        all_augmented_questions=json.loads(cleaned_str)
    except Exception as e:
        print(e)
        print(f"cleaned_str:{cleaned_str}")

    if not isinstance(all_augmented_questions, list):
        ast.literal_eval(all_augmented_questions)

    # 生成新的对话
    new_dialogues = [row_text]
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
            replacement = rf'User: {augmented_question}\3'
            new_dialogue = re.sub(pattern, replacement, new_dialogue, count=1)
        append_text=row_text.copy()
        append_text["representative"]=new_dialogue
        new_dialogues.append(append_text)
    return new_dialogues

if __name__ == '__main__':
    df = pd.read_csv("data/outYinkaClean20250603.csv")
    for index, row in df.iterrows():
        print(f"index:{index}  row:{row} ")
        new_dialogues=create_new_dialogues(dict(row))
        df=pd.DataFrame(new_dialogues)
        df.to_csv("data/enhance_noise_20250604.csv",mode='a', index=False,header=False,encoding='utf-8-sig')





