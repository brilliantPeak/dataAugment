# -*- coding: utf-8 -*-

import json
import logging
import time

import requests
from  prompt_template import intention_prompt_template

def construct_prompt(his: str, inputx: str) -> str:
    his_list= list(json.loads(his))
    his_list.append({"role" :"user" ,"content" :inputx})
    prompt = {"model": "deepseek-v1",
              "temperature": 0.2,
              "max_length": 10000,
              "chat_template_kwargs": {"enable_thinking": False}
              }
    msgs = [{"role": "system",
             "content": """你是一名经验丰富的兽医，在犬猫的诊断、治疗和预防等方面有着多年临床经验。熟悉各类犬猫常见疾病，
             包括传染病、寄生虫病、内科疾病、外科疾病等，能够为犬猫主人提供专业的医疗建议和指导。当前为多轮对话数据增强背景
             ，请根据你的职业经验从用户提问意图多样化，模糊口语化，错别字&噪声提问三个角度对用户侧的数据进行增强。
             注意，给你提供的多轮对话中含有User和Assistant两部分，你只需要对User部分进行增强即可，Assistant部分按照原内容返回。
             请注意，你需要生成五组完整的conversations对话，其中每个conversations对话都包含完整的多轮对话信息"""},
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
    resTxt=json.loads(response.text)
    print(f'Response answer: {resTxt["answer"]}')
    return response

if __name__ == '__main__':

    query="""
    User: 狗狗肚子咕咕叫还吐白色泡沫Assistant: 您好，狗狗肚子咕咕叫并且吐白色泡沫可能有多种原因，例如胃肠炎、胃酸倒流、食物不耐受或寄生虫感染等。不过，仅依据这些简单症状并不能准确找到真正的病因。可以了解更多的宠物信息吗？User: 当然可以。Assistant: 好的，请问您的狗狗是什么品种的呢？User: 它是一只金毛。Assistant: 明白了，那请问您的狗狗现在多大了呢？User: 三岁。Assistant: 请问狗狗的性别是公还是母？User: 是公狗。Assistant: 是否已经绝育了呢？User: 没有绝育。Assistant: 狗狗的疫苗注射情况如何？User: 所有的常规疫苗都按时注射了。Assistant: 狗狗的驱虫情况怎么样？User: 驱虫按时进行，上个月刚做过。Assistant: 最近狗狗的饮食有没有发生什么变化？User: 最近没有太大变化，饮食还是和以前一样。Assistant: 好的，请问狗狗的生活环境有没有什么改变？User: 没有什么改变，还是住在原来的环境里。Assistant: 那么再问下，狗狗在肚子咕咕叫和吐泡沫之外，有没有表现出其他的症状？比如腹泻、便秘、食欲减退或嗜睡？User: 食欲有点减退，经常不怎么吃东西。Assistant: 明白了，请问狗狗的精神状态如何？是否有不活泼或者显得异常疲惫的情况？User: 有点不太活泼，经常安静地躺着。Assistant: 最近狗狗的饮水量有变化吗？比如喝水特别多或者特别少？User: 好像喝水比平时多了一些。Assistant: 狗狗的大小便情况怎样？有没有出现腹泻或便秘的问题？User: 有点腹泻，便便比较稀。Assistant: 请问狗狗的体温您有测量过吗？是否有发烧的现象？User: 昨天测量过，好像有点发热。Assistant: 狗狗的皮肤弹性如何？有没有出现脱水的迹象，比如皮肤拉起后很久才能回弹？User: 皮肤弹性还可以，好像没有脱水的现象。Assistant: 根据您的描述，狗狗的情况可能涉及以下几种疾病。
    诊断小结
    #鉴别诊断
    1. 胃肠炎
    2. 肠道菌群紊乱
    3. 寄生虫感染
    4. 肾功能不全
    以上鉴别诊断只是根据问诊列出概率较大几种疾病，并非全部，请知晓。
    #治疗建议
    - 近期不要给狗狗喂食刺激性或油腻的食物，可以尝试喂食容易消化的处方粮。
    - 注意补充水分，避免脱水。
    - 使用宠物益生菌帮助调节肠道菌群。
    #就医建议
    - 建议尽早带狗狗去兽医处进行详细检查，包括血液化验和粪便检查，以明确具体病因，并制定相应治疗方案。
    #日常护理
    - 继续观察狗狗的食欲、饮水量、大小便情况和精神状态。
    - 确保生活环境清洁，避免交叉感染。
    - 适当减少剧烈运动，让狗狗多休息。
    请根据上述建议调整日常护理，如有进一步问题，随时咨询专业兽医。
    """
    format_prompt = intention_prompt_template %  query

    history = []
    his_json = json.dumps(history, ensure_ascii=False)
    prompt = construct_prompt(his_json, format_prompt)

    print(f"prompt: {prompt}")

    start=time.time()
    response = build_request(prompt)
    end = time.time()
    print(f"耗费时间: {end-start}")
    print(response.text)