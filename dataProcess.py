# -*- coding: utf-8 -*-
"""
生产级：多轮对话拼接语义向量化 + 聚类 + 去重 + 主问生成流程
适用于完整多轮问诊对话的聚类整理，用于语义一致数据生成标准问、分类标注或相似对话归并。
"""
from collections import defaultdict
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os
import json
import time
import pandas as pd
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
import openai
import torch
import numpy as  np
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ----------- 参数配置 -----------
bert_MODEL = 'embed_model/chinese_roberta_wwm_ext_pytorch'
Yinka_MODEL="embed_model/Yinka"
GPT_MODEL = 'gpt-3.5-turbo'
N_CLUSTERS = 20
openai.api_key = os.getenv("OPENAI_API_KEY")
batch_size =8

class TextClusterRepresentative:
    def __init__(self, clustered_data):
        """初始化函数
        :param clustered_data: 聚类结果列表，每个元素为 {"cluster": int, "content": str, "embedding": np.ndarray}
        :param embeddings: 所有样本的嵌入向量列表/数组
        """

        self.clustered_data = clustered_data
        self.embeddings = np.array([item["embedding"] for item in clustered_data])
        self.clusters = self._group_by_cluster()
        self.dialogues = [item["content"] for item in clustered_data]
    def _group_by_cluster(self):
        """按簇ID分组存储样本索引"""

        clusters = defaultdict(list)
        for idx, item in enumerate(self.clustered_data):
            clusters[item["cluster"]].append(idx)
        return clusters

    def select_representatives(self,similarity_metric='cosine'):
        """选择每个簇的代表样本
        :param similarity_metric: 相似度计算方式，默认为余弦相似度
        :return: 代表样本列表 [{"cluster": int, "content": str, "embedding": np.ndarray}]
        """

        representatives = []
        for cluster_id, member_indices in self.clusters.items():
            if len(member_indices) == 1:
                # 单样本簇直接选取
                representatives.append({'cluster':cluster_id,'content': self.dialogues[member_indices[0]]} )
            else:
                # 多样本簇计算代表样本
                members = [self.embeddings[i] for i in member_indices]
                center = np.mean(members, axis=0)

                if similarity_metric == 'cosine':
                    similarities = cosine_similarity([center], members)[0]
                elif similarity_metric == 'euclidean':
                    # 欧氏距离取反作为相似度
                    distances = np.linalg.norm(center - members, axis=1)
                    similarities = 1 / (1 + distances)  # 转换为相似度分数
                else:
                    raise ValueError(f"不支持的相似度计算方式: {similarity_metric}")

                best_idx = member_indices[np.argmax(similarities)]
                representatives.append({'cluster':cluster_id,'content':self.dialogues[best_idx]})

        return representatives

    def _calculateSimilarities(self, center, members, metric):
        """统一相似度计算接口"""
        if metric == 'cosine':
            # 批量计算余弦相似度 (num_members,)
            sims = cosine_similarity([center], members)[0]
        elif metric == 'euclidean':
            # 批量计算欧氏距离转相似度 (num_members,)
            distances = np.linalg.norm(center - members, axis=1)
            sims = 1 / (1 + distances)  # 转换为相似度分数
        else:
            raise ValueError(f"不支持的相似度计算方式: {metric}")
        return sims


    def select_representatives(self,top_diff, similarity_metric='cosine'):
        """增强版代表选择器
        :param top_diff: 每个簇额外保留的差异最大样本数
        :return: 包含中心代表和差异样本的列表
        """
        representatives = []

        for cluster_id, member_indices in self.clusters.items():
            if len(member_indices) <= 1:
                # 单样本簇直接选取
                representatives.append({
                    'cluster': cluster_id,
                    'content': self.dialogues[member_indices[0]],
                    'similarity':0,
                    'type': 'single'  # 标记样本类型
                })
                continue

            members = self.embeddings[member_indices]
            center = np.mean(members, axis=0)

            # 计算所有成员与中心点的相似度
            sims = self._calculateSimilarities(center, members, similarity_metric)

            # 选择中心代表
            center_idx = member_indices[np.argmax(sims)]
            # print(f"center_idx:{center_idx}")
            center_rep = {
                'cluster': cluster_id,
                'content': self.dialogues[center_idx],
                'similarity': np.max(sims),
                'type': 'center'
            }
            representatives.append(center_rep)

            select_num=int(top_diff*len(member_indices))
            if round(select_num) <= 0:
                continue  # 不需要额外差异样本

            # 计算成员之间的两两相似度矩阵
            sim_matrix = cosine_similarity(members) if similarity_metric == 'cosine' else 1 / (1 + cdist(members, members, 'euclidean'))

            # 排除自身相似度（对角线）
            np.fill_diagonal(sim_matrix, np.inf)

            # 找出每个样本与其他成员的最小相似度
            min_sims = np.min(sim_matrix, axis=1)

            # 选择差异最大的select_num个样本（排除中心样本）
            # diff_indices = np.argsort(min_sims)[::-1][1:select_num + 1]
            diff_indices = np.argsort(min_sims)[:-1][:select_num]   # 跳过中心样本

            # 收集差异最大的样本
            for diff_idx in diff_indices:
                diff_member_idx = member_indices[diff_idx]
                diff_rep = {
                    'cluster': cluster_id,
                    'content': self.dialogues[diff_member_idx],
                    'similarity': min_sims[diff_idx],
                    'type': 'diverse'
                }
                representatives.append(diff_rep)

        return representatives
    def to_dataframe(self):
        """转换为DataFrame格式"""
        return pd.DataFrame([{"cluster": r["cluster"], "representative": r["content"],"similarity":r["similarity"],"type":r["type"]}
                             for r in self.select_representatives(0.2) ])

    def save_to_csv(self, output_path):
        """保存结果到CSV文件"""

        df = self.to_dataframe()
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"代表样本已保存至 {output_path}")



# ----------- 1. 多轮拼接为完整语段 -----------
def load_full_dialogues(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        dialogues = []
        for line in f:
            content = ""
            json_line = json.loads(line)
            for item in json_line['messages']:
                content +=f"{item['role'].capitalize()}: {item['content'].strip()}"
            dialogues.append(content)
    return dialogues


def  elbow(cls_embedding):
    """
    肘部法则图像
    计算不同 k 值下的聚类误差平方和，即簇内样本到簇中心的距离平方和。
    绘制 k 与 WCSS 的关系曲线，选择“肘部”位置（即 WCSS 下降速度明显减缓的点）作为最优 k 值
    :param cls_embedding:
    :return:
    """
    wcss = []
    k_range= range(1, 100)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(cls_embedding)
        wcss.append(kmeans.inertia_)  # inertia_ 是 WCSS

    # 绘制肘部法则图
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, wcss, 'bo-', markersize=8)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()


def silhouette(cls_embedding):
    """
    轮廓系数
    衡量样本与其所在簇的紧密程度以及与其他簇的分离程度。
    取值范围为 [-1, 1]：
    接近 1：样本与簇内其他点紧密，与其他簇分离良好。
    接近 0：样本位于簇边界，可能属于多个簇。
    负值：样本可能被错误分配到其他簇。
    :param cls_embedding:
    :return:
    """
    silhouette_scores = []
    k_range =range(2, 100)  # 轮廓系数至少需要 2 个簇

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(cls_embedding)
        score = silhouette_score(cls_embedding, labels)
        silhouette_scores.append(score)

    # 绘制轮廓系数图
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, silhouette_scores, 'bo-', markersize=8)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for Optimal k')
    plt.grid(True)
    plt.show()


def csv_write(clustered, path):
    df=pd.DataFrame(clustered)
    df[["cluster","content"]].to_csv(path, index=False, encoding='utf-8-sig')


def cluster_call(clu_type, cls_embedding,n_clusters):
    labels =KMedoids(n_clusters=n_clusters, random_state=42).fit_predict(cls_embedding)  \
        if clu_type == 'KMedoids'  else KMeans(n_clusters=n_clusters, random_state=42).fit_predict(cls_embedding)
    return labels

def yinka_embed_and_cluster(dialogues, n_clusters=N_CLUSTERS):
    """
    yinka文本向量化
    :param dialogues: 文本集合
    :param n_clusters: 聚类数
    :return:
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(Yinka_MODEL)
    model.to(device)
    cls_embedding_list=[]
    for i in range(0, len(dialogues), batch_size):
        batch_texts=dialogues[i:i+batch_size]
        vectors = model.encode(batch_texts, normalize_embeddings=False)
        cut_vecs = normalize(vectors[:, :])
        cls_embedding_list.append(cut_vecs)
        torch.cuda.empty_cache()
    cls_embedding = np.concatenate(cls_embedding_list, axis=0)

    labels=cluster_call(KMeans,cls_embedding,n_clusters)
    clustered=[{"cluster": int(label), "content":d,"embedding":embedding,}   for d, label,embedding in zip(dialogues, labels,cls_embedding)]
    csv_write(clustered,"data/outYinka20250603.csv")
    return clustered


def bert_embed_and_cluster(dialogues, n_clusters=N_CLUSTERS):
    """
    bert文本向量化
    :param dialogues: 文本集合
    :param n_clusters: 聚类数
    :return:
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained(bert_MODEL)
    model = BertModel.from_pretrained(bert_MODEL)
    print(f"model_config:{model.config}")

    model.to(device)
    cls_embedding_list = []
    for i in range(0, len(dialogues), batch_size):
        batch_texts=dialogues[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs.to(device)
        # print(f"input_ids:{inputs['input_ids'].shape}")
        outputs = model(**inputs)
        batch_cls_embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        cut_vecs = normalize(batch_cls_embedding[:, :])

        cls_embedding_list.append(cut_vecs)
        torch.cuda.empty_cache()
    cls_embedding = np.concatenate(cls_embedding_list, axis=0)

    labels=cluster_call(KMeans,cls_embedding,n_clusters)
    clustered = [{"cluster": int(label), "content": d, "embedding": embedding, } for d, label, embedding in  zip(dialogues, labels, cls_embedding)]

    csv_write(clustered,"data/outbert.csv")
    return clustered


def  count_figure_show(cluster_counts):
    # 绘制柱状图
    cluster_counts.plot(kind="bar", figsize=(18, 6))
    plt.title("Cluster Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.xticks(rotation=0)  # 不旋转 x 轴标签
    plt.show()



# --------------------- 4. 主流程 ------------------------
def run_pipeline(input_path, output_path):
    full_dialogues = load_full_dialogues(input_path)
    clustered=yinka_embed_and_cluster(full_dialogues)

    selector=TextClusterRepresentative(clustered)
    selector.save_to_csv(output_path)

    # -----------分组打印 -----------
    # df=pd.read_csv("data/outbert.csv")
    # cluster_size = df.groupby("cluster").size()
    # count_figure_show(cluster_size)

    # -----------循环写入 -----------
    # with open(output_jsonl, 'w', encoding='utf-8') as f:
    #     for item in results:
    #         f.write(json.dumps(item, ensure_ascii=False) + '\n')
    # print(f"已完成处理并输出至: {output_jsonl}")

# ----------- 入口 -----------
if __name__ == '__main__':
    run_pipeline("data/samples.jsonl", "data/outbertClean20250603.csv")
