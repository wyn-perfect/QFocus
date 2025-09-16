from openai import OpenAI
import os
import re
import numpy as np  # 导入numpy用于计算均值
import concurrent.futures
import argparse
from util import load_json, write_str, load_str


model = 'XXX'
client = OpenAI(
    api_key='XXX',
    
)


def eval_restore_llm_direct(init_article, restore_article):
    try:
        # 设计简化的prompt
        prompt = f"""
        ### 任务描述：
        你将收到一篇原始文章和一篇基于原始文章的写作思路框架重新构建的还原文章。
        你的任务是对还原文章在以下7个维度上进行评分，并给出一个综合分数：
        1. **内容一致性**：评估还原文章与原文的主题、观点、事实和信息是否一致。
        2. **结构还原度**：评估还原文章是否准确还原了原文的结构，包括子主题及其逻辑关系和组织顺序。
        3. **信息覆盖度**：评估还原文章是否涵盖了原文中的所有主要观点和细节，尤其是关键信息。
        4. **内容准确度**：评估还原文章中关键事实、论据、数据引用等是否准确，确保信息无误。
        5. **语义一致性**：评估还原文章是否准确传达了原文的思想、观点和逻辑，确保语义的连贯性。
        6. **语言流畅性**：评估还原文章的语言是否通顺，句子结构是否符合语法规则，是否存在不自然或不流畅的表达。
        7. **上下文连贯性**：评估还原文章是否能够保持原文中段落和句子之间的逻辑衔接，确保前后段落之间关系清晰。

        每个维度的评分范围为1到5分，综合分数也应介于1到5之间，均可以包含小数，反映还原文章的整体质量。
        对于每个维度评估，请详细说明还原文章在该维度下的表现，请明确指出哪些部分与原文一致，哪些部分有偏差，且描述如何影响评分。注意，反馈要具体详细最后指出哪些内容问题，避免使用模糊的表述。

        请直接按照以下格式输出评分结果：
        - 内容一致性评分：X分
          - **反馈**：
        - 结构还原度评分：X分
          - **反馈**：
        - 信息覆盖度评分：X分
          - **反馈**：
        - 内容准确度评分：X分
          - **反馈**：
        - 语义一致性评分：X分
          - **反馈**：
        - 语言流畅性评分：X分
          - **反馈**：
        - 上下文连贯性评分：X分
          - **反馈**：
        - 综合评分：X分

        ### 原始文章：
        {init_article}

        ### 待评估的还原文章：
        {restore_article}
        
        """
        
        messages = [
            {"role": "system", "content": "你是一个分析文章结构和内容的助手，负责评估还原文章与原文的匹配度并提供评分和反馈。"},
            {"role": "user", "content": prompt},
        ]
        
        
        chat_response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.01,
            top_p=0.8
        )
        
        
        answer = chat_response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error in eval_restore_llm: {e}")
        return None




def eval_restore_llm_direct_1(init_article, restore_article):
    try:
        
        with open("eval_5.prompt", "r", encoding="utf-8") as f:
            criteria = f.read()

        prompt = f"""
        ### 任务描述：
        你将收到一篇原始文章和一篇基于原始文章的写作思路框架重新构建的还原文章。
        请根据以下评估标准对还原文章进行评估:
        {criteria}

        每个维度的评分范围为1到5分，综合分数也应介于1到5之间，均可以包含小数，反映还原文章的整体质量。
        对于每个维度评估，请详细说明还原文章在该维度下的表现，请明确指出哪些部分与原文一致，哪些部分有偏差，且描述如何影响评分。注意，反馈要具体详细最后指出哪些内容问题，避免使用模糊的表述。

        请直接按照以下格式输出评分结果：
        - 内容一致性评分：X分
          - **反馈**：
        - 结构还原度评分：X分
          - **反馈**：
        - 信息覆盖度评分：X分
          - **反馈**：
        - 内容准确度评分：X分
          - **反馈**：
        - 语义一致性评分：X分
          - **反馈**：
        - 综合评分：X分

        ### 原始文章：
        {init_article}

        ### 待评估的还原文章：
        {restore_article}
        
        """
        
        messages = [
            {"role": "system", "content": "你是一个分析文章结构和内容的助手，负责评估还原文章与原文的匹配度并提供评分和反馈。"},
            {"role": "user", "content": prompt},
        ]
        
        
        chat_response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.01,
            top_p=0.8
        )
        
        
        answer = chat_response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error in eval_restore_llm: {e}")
        return None




def extract_scores(text):
    
    pattern = r"(\S+评分)：(\d+)分"
    scores = re.findall(pattern, text)
    
    
    score_dict = {item[0]: int(item[1]) for item in scores}
    
    return score_dict


def calculate_mean(scores_list):
    """
    calculate the mean of all scores in the list of score dictionaries
    """
    if not scores_list:
        return {}
    mean_scores = {}
    keys = scores_list[0].keys()
    for key in keys:
        values = [scores[key] for scores in scores_list if key in scores]
        mean_scores[key] = round(np.mean(values), 2)
    return mean_scores



def process_articles_in_batches(json_path ,input_dir,output_dir,start,end):
    
    
    articles = load_json(json_path)
    articles = articles[start:end]

    for idx, article in enumerate(articles, start=start):  
        title = article.get("title", "")
        if not title:
            print("  - Empty title, skipping.")
            continue
        init_article = article.get("content", "")
        
        article_file_path = os.path.join(input_dir, f"{title}.txt")


        if not os.path.exists(article_file_path):
            
            print(f"  - File not found: {article_file_path}. Skipping.")
            
            
        restore_article = load_str(article_file_path)
        result= eval_restore_llm_direct_1(init_article, restore_article)
        
        write_str(result,os.path.join(output_dir,f"{title}.txt"))




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process articles in batches and evaluate them using LLM.")

    parser.add_argument("--json-path", type=str, required=True, help="Path to the JSON file with articles.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input article files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output results.")
    parser.add_argument("--total-articles", type=int,  help="Total number of articles to process.")
    parser.add_argument("--batch-size", type=int,  help="Number of articles per batch.")
    
    args = parser.parse_args()


    total_articles = args.total_articles
    batch_size = args.batch_size
    num_threads = total_articles // batch_size

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            futures.append(executor.submit(process_articles_in_batches, args.json_path, args.input_dir, args.output_dir, batch_start, batch_end))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")




