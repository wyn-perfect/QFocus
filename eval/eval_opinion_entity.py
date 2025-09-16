
import os
from LAC import LAC
from typing import List, Optional
import numpy as np
import random
from util import load_json, dump_json
import argparse

lac = LAC(mode='lac')

def extract_entities_from_list(l):
    entities = set()  
    for sent in l:
        if len(sent) == 0:
            continue
        lac_result = lac.run(sent)
        entity = [word for word, tag in zip(lac_result[0], lac_result[1]) if tag in ['ORG', 'PER', 'LOC', 'TIME']]
        entities.update(entity)  


    return list(entities)  

def heading_entity_recall(
                          golden_opinions: Optional[List[str]] = None,
                          predicted_opinions: Optional[List[str]] = None):
    """
    Given golden entities and predicted entities, compute entity recall.
        -  golden_entities: list of strings or None; if None, extract from golden_headings
        -  predicted_entities: list of strings or None; if None, extract from predicted_headings
        -  golden_headings: list of strings or None
        -  predicted_headings: list of strings or None
    """

    golden_entities = extract_entities_from_list(golden_opinions)
    predicted_entities = extract_entities_from_list(predicted_opinions)

    g = set(golden_entities)
    p = set(predicted_entities)
    if len(g) == 0:
        return 1
    else:
        return len(g.intersection(p)) / len(g)
    

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
    
def compute_opinion_entity_recall(titles_json_path, golden_frame_dir, predicted_frame_dir,start = None,end=None, result_dir=None):   
    
    try:
        
        articles = load_json(titles_json_path)
        articles = articles[start:end]
        all_recalls = []
    
        for article in articles:
            title = article.get("title", "")
            if not title:
                continue



            golden_frame_path = os.path.join(golden_frame_dir, f"{title}_frame.json")
            if not os.path.exists(golden_frame_path):
                continue

            golden_core_points = extract_core_points(golden_frame_path)
            predicted_frame_path = os.path.join(predicted_frame_dir, f"{title}_frame.json")
            if not os.path.exists(predicted_frame_path):
                continue


            predicted_core_points = extract_core_points(predicted_frame_path)


            recall = heading_entity_recall(golden_core_points, predicted_core_points)
            all_recalls.append(recall)
        

        
        all_avg = float(np.mean(all_recalls)) if all_recalls else 0
        random100_avg = float(np.mean([np.mean(random.sample(all_recalls, min(100, len(all_recalls)))) for _ in range(5)])) if len(all_recalls) >= 100 else 0
        
        
        results_path = os.path.join(result_dir, "opinion_soft_results.json")
        all_results = {
            "average_all": all_avg,
            "average_random_100": random100_avg,
            
        }
        dump_json(all_results, results_path)

        return all_results


    except Exception as e:
        print(f"Error in compute_opinion_entity_recall: {e}")
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

    results = compute_opinion_entity_recall(args.titles_json_path, args.golden_frame_dir, args.predicted_frame_dir, args.start, args.end, args.result_dir)


   

