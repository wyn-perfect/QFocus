
import re
import json
import os
from LAC import LAC
from typing import List, Optional
import numpy as np
import random
from argparse import ArgumentParser

lac = LAC(mode='lac')

# -----------------------------
# I/O utilities
# -----------------------------
def dump_json(obj, file_name, encoding="utf-8"):
    """Save a Python object as JSON to the given path."""
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w', encoding=encoding) as fw:
        json.dump(obj, fw, ensure_ascii=False,indent=4)

def load_json(file_path, n=None):
    """Load a JSON file; if n is provided, return only the first n items."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if n is not None:
            data = data[:n]
        return data
    except Exception as e:
        print(f"Error: {e}")
        return []



def load_str(path):
    """Load a text file as a single string."""
    with open(path, 'r') as f:
        return '\n'.join(f.readlines())


# -----------------------------
# NER with LAC
# -----------------------------

def extract_entities_lac(text: str) -> List[str]:
    """
    Extract named entities from Chinese text with LAC.
    Keeps basic entity tags: ORG, PER, LOC, TIME.
    """
    lac_result = lac.run(text)
    entities = [word for word, tag in zip(lac_result[0], lac_result[1]) if tag in ['ORG', 'PER', 'LOC', 'TIME']]
    return entities

# -----------------------------
# Metrics
# -----------------------------

def article_entity_recall(golden_entities: Optional[List[str]] = None,
                          predicted_entities: Optional[List[str]] = None,
                          golden_article: Optional[str] = None,
                          predicted_article: Optional[str] = None):
    """
    Given golden entities and predicted entities, compute entity recall for two articles.
    - golden_entities: list of strings or None; if None, extract from golden_article
    - predicted_entities: list of strings or None; if None, extract from predicted_article
    - golden_article: string or None
    - predicted_article: string or None
    """
    if golden_entities is None:
        assert golden_article is not None, "golden_article and golden_entities cannot both be None."
        sentences = re.split(r'(?<=。|！|？)', golden_article)  # sentence split for Chinese
        golden_entities = []
        for sentence in sentences:
            golden_entities.extend(extract_entities_lac(sentence))
    
    if predicted_entities is None:
        assert predicted_article is not None, "predicted_article and predicted_entities cannot both be None."
        sentences = re.split(r'(?<=。|！|？)', predicted_article)  
        predicted_entities = []
        for sentence in sentences:
            predicted_entities.extend(extract_entities_lac(sentence))

    g = set(golden_entities)
    
    p = set(predicted_entities)
    

    if len(g) == 0:
        return 1.0  

    
    common_entities = g.intersection(p)
    return len(common_entities) / len(g)

# -----------------------------
# Batch evaluation
# -----------------------------

def compute_article_entity_recall(json_path ,input_dir,output_dir,start,end):
    
    articles = load_json(json_path)
    articles = articles[start:end]
    
    all_recall = []
    
    for idx, article in enumerate(articles, start=start):  
        title = article.get("title", "")
        print(f"正在处理第 {idx + 1} 篇文章，标题：{title}")  
        if not title:
            print("  - Empty title, skipping.")
            continue
        golden_article = article.get("content", "")
        
        article_file_path = os.path.join(input_dir, f"{title}.txt")


        if not os.path.exists(article_file_path):
            print(f"  - File not found: {article_file_path}. Skipping.")
        
        
        predicted_article = load_str(article_file_path)
        recall = article_entity_recall(golden_article=golden_article, predicted_article=predicted_article)
        # print("Entity Recall:", recall)
        all_recall.append(recall)


    # Aggregate statistics
    all_avg = np.mean(all_recall) if all_recall else 0
    # Random sample of 100 values, repeated 5 times (if possible)
    random_100_avg = np.mean([np.mean(random.sample(all_recall, min(100, len(all_recall)))) for _ in range(5)]) if len(all_recall) >= 100 else 0
    
    results = {
        "mean_all": all_avg,
        "mean_random_100_x5": random_100_avg,
        
    }
    output_file_path = os.path.join(output_dir, "article_entity_recall_results.json")
    dump_json(results,output_file_path)
    print(f"[done] Finished. Results saved to: {output_file_path}")



if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--json-path", type=str, required=True, help="Path to JSON with golden articles (must include 'title' and 'content').")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing predicted articles named as <title>.txt.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results JSON.")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive).")
    parser.add_argument("--end", type=int, default=1, help="End index (exclusive). Use -1 to run until the end.")

    args = parser.parse_args()  # <-- parse AFTER adding all arguments
    end_arg = None if args.end is None or args.end < 0 else args.end
    compute_article_entity_recall(
        json_path=args.json_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
    )
    
   


    

