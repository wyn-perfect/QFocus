from rouge_chinese import Rouge
import json
import os
import jieba
import random
import argparse
from util import load_json, write_str, load_str,dump_json

def compute_rouge_scores(golden_answer: str, predicted_answer: str):
    
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores between golden and predicted answers.

    Args:
        golden_answer (str): Reference text.
        predicted_answer (str): Generated text.

    Returns:
        dict: ROUGE scores (F1, precision, recall).
    """
    
    words1 = " ".join(jieba.cut(golden_answer))
    words2 = " ".join(jieba.cut(predicted_answer))

    rouge = Rouge()
    scores = rouge.get_scores(words2, words1)[0]  
    result = {
        'rouge1': {
            'fmeasure': scores['rouge-1']['f'],
            'precision': scores['rouge-1']['p'],
            'recall': scores['rouge-1']['r']
        },
        'rouge2': {
            'fmeasure': scores['rouge-2']['f'],
            'precision': scores['rouge-2']['p'],
            'recall': scores['rouge-2']['r']
        },
        'rougeL': {
            'fmeasure': scores['rouge-l']['f'],
            'precision': scores['rouge-l']['p'],
            'recall': scores['rouge-l']['r']
        }
    }
    return result




def process_articles(json_file_path, input_dir, output_dir, start, end):
    """
    Processes articles from a JSON file, calculates ROUGE scores.

    Args:
        json_file_path (str): Path to the JSON file.
        input_dir (str): Directory with text files.
        start (int, optional): Start index.
        end (int, optional): End index.

    Returns:
        None
    """

     
    results = {}  
    rouge_sums = {"rouge1": {"sum": 0, "count": 0}, "rouge2": {"sum": 0, "count": 0}, "rougeL": {"sum": 0, "count": 0}}

    
    articles = load_json(json_file_path)
    if start is not None and end is not None:
        articles = articles[start:end]
    
    for index, article in enumerate(articles):
        title = article.get("title", "")
        golden_answer  = article.get("content", "")

        
        if not title:
            continue
        print(f"正在处理第 {index + 1} 篇文章：'{title}'") 

        if not golden_answer.strip(): 
            continue

        
        article_file_path = os.path.join(input_dir, f"{title}.txt")
        print(article_file_path)
        if not os.path.exists(article_file_path):
            continue

        predicted_answer = load_str(article_file_path)
        if not predicted_answer.strip():
            continue
        results[title] = {}  
                
        scores = compute_rouge_scores(golden_answer, predicted_answer)
        results[title] = scores

        
        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
            rouge_sums[rouge_type]['sum'] += scores[rouge_type]['fmeasure']
            rouge_sums[rouge_type]['count'] += 1

 
    average_rouge = {rouge_type: sums['sum'] / sums['count'] if sums['count'] > 0 else 0 for rouge_type, sums in rouge_sums.items()}
    random_100 = random.sample(list(results.values()), min(100, len(results)))
    average_rouge_random_100 = {rouge_type: sum(score[rouge_type]['fmeasure'] for score in random_100) / len(random_100) for rouge_type in ['rouge1', 'rouge2', 'rougeL']}
    output_file_path = os.path.join(output_dir, "rouge_scores_detail.json")
    dump_json(results, output_file_path)
    all_averages = {
        "average_all": average_rouge,
        "average_random_100": average_rouge_random_100,
    }

    average_output_file_path = os.path.join(output_dir, "all_averages.json")
    dump_json(all_averages, average_output_file_path)
   






# 示例调用
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate Rouge scores for articles.")

    parser.add_argument("--json-path", type=str, required=True, help="Path to the JSON file containing human written articles.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing the predicted article files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the results.")
    parser.add_argument("--start", type=int, required=True, help="Start index of articles to process.")
    parser.add_argument("--end", type=int, required=True, help="End index of articles to process.")

    args = parser.parse_args()

    process_articles(args.json_path, args.input_dir, args.output_dir, args.start, args.end)
