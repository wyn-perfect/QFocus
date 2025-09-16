
import os
import random
from typing import List
import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from util import load_json, dump_json
import argparse

encoder = SentenceTransformer('BAAI/bge-base-zh-v1.5')


def card(l):
    encoded_l = encoder.encode(list(l))
    cosine_sim = cosine_similarity(encoded_l)
    soft_count = 1 / cosine_sim.sum(axis=1)

    return soft_count.sum()


def heading_soft_recall(golden_opinions: List[str], predicted_opinions: List[str]):
    """
    Given golden headings and predicted headings, compute soft recall.
        -  golden_headings: list of strings
        -  predicted_headings: list of strings

    Ref: https://www.sciencedirect.com/science/article/pii/S0167865523000296
    """

    g = set(golden_opinions)
    p = set(predicted_opinions)
    if len(p) == 0:
        return 0
    card_g = card(g)
    card_p = card(p)
    card_intersection = card_g + card_p - card(g.union(p))
    return card_intersection / card_g



def extract_core_points(frame_path):
    try:
        frame_data = load_json(frame_path)
        core_points = []
        for section in frame_data.get("分析部分", []):
            core_points.append(section.get("核心观点", ""))
        return core_points
    except Exception as e:
        print(f"Error in extract_core_points_and_conclusions: {e}")
        return []


def evaluate_opinion_soft(titles_json_path, golden_frame_dir, predicted_frame_dir,start = None,end=None, result_dir=None):   
    
    try:
        # 读取标题列表
        articles = load_json(titles_json_path)
        articles = articles[start:end]
        # 初始化结果列表
        all_recalls = []
    
        # 遍历每个标题
        for article in articles:
            title = article.get("title", "")
            if not title:
                print(f"文章标题为空，跳过处理。")
                continue


            # 构造原文框架文件路径
            golden_frame_path = os.path.join(golden_frame_dir, f"{title}_frame.json")
            if not os.path.exists(golden_frame_path):
                print(f"未找到原文框架文件：{golden_frame_path}")
                continue
            # 提取原文的核心观点和结论
            golden_core_points = extract_core_points(golden_frame_path)
            predicted_frame_path = os.path.join(predicted_frame_dir, f"{title}_frame.json")
            if not os.path.exists(predicted_frame_path):
                print(f"未找到还原文章的框架文件：{predicted_frame_path}")
                continue

            # 提取还原文章的核心观点和结论
            predicted_core_points = extract_core_points(predicted_frame_path)

            # 计算软召回率
            recall = heading_soft_recall(golden_core_points, predicted_core_points)
            #print(f"文章：{title}，软召回率：{recall:.4f}")
            all_recalls.append(recall)
        

        #计算各种平均值
        all_avg = float(np.mean(all_recalls)) if all_recalls else 0
        random100_avg = float(np.mean([np.mean(random.sample(all_recalls, min(100, len(all_recalls)))) for _ in range(5)])) if len(all_recalls) >= 100 else 0
       

            
        # 保存结果到文件
        results_path = os.path.join(result_dir, "opinion_soft_results.json")
        all_results = {
            "全部平均值": all_avg,
            "随机100个平均值": random100_avg,
            
        }
        dump_json(all_results, results_path)
        print(f"评估结果已保存到 {results_path}")

        return all_results


    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return None
    





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate opinion entity recall for articles.")

    parser.add_argument("--titles-json-path", type=str, required=True, help="Path to the JSON file containing titles and articles.")
    parser.add_argument("--golden-frame-dir", type=str, required=True, help="Directory containing the golden frame files.")
    parser.add_argument("--predicted-frame-dir", type=str, required=True, help="Directory containing the predicted frame files.")
    parser.add_argument("--result-dir", type=str, required=True, help="Directory to save the result JSON.")
    parser.add_argument("--start", type=int, help="Start index of articles to process.")
    parser.add_argument("--end", type=int,  help="End index of articles to process.")

    args = parser.parse_args()

    results = evaluate_opinion_soft(args.titles_json_path, args.golden_frame_dir, args.predicted_frame_dir, args.start, args.end, args.result_dir)


