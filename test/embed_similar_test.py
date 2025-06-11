# -*- coding: utf-8 -*-


import os
import pandas as pd
import openai
import torch
import numpy as  np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity



bert_MODEL = 'embed_model/chinese_roberta_wwm_ext_pytorch'
Yinka_MODEL="embed_model/Yinka"
GPT_MODEL = 'gpt-3.5-turbo'
N_CLUSTERS = 20
openai.api_key = os.getenv("OPENAI_API_KEY")
batch_size =8


def yinka_embed(dialogues):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(Yinka_MODEL)
    model.to(device)
    vectors = model.encode(dialogues, normalize_embeddings=False)
    cut_vecs = normalize(vectors[:, :])
    torch.cuda.empty_cache()
    sim_matrix = cosine_similarity(cut_vecs)

    # 排除自身相似度（对角线）
    np.fill_diagonal(sim_matrix, np.inf)

    # 找出每个样本与其他成员的最小相似度
    min_sims = np.min(sim_matrix, axis=1)
    return min_sims


if __name__ == '__main__':
    df = pd.read_csv("../data/enhance_noise_test.csv", encoding='gbk')
    embed_txt=[]
    for index, row in df.iterrows():
        print(f"index:{index}  row:{row} ")
        embed_txt.append(row['representative'])

    new_dialogues=yinka_embed(embed_txt)
    df=pd.DataFrame(new_dialogues)