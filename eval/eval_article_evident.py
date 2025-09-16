import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse
import random
"""
--output-path:
The file saved at this location contains the detailed results of the similarity calculations between the init_argument (from the initial data) and the new_argument (from the new data), as well as the calculated hit rates and similarity pairs.
--summary-output-path:
The summary JSON will contain high-level metrics that summarize the results of your entire evaluation.

"""

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    

def save_json(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

model = SentenceTransformer('BAAI/bge-base-zh-v1.5')

def calculate_similarity(golden_answer, predicted_answer, threshold=0.69):
    """
    Calculate the similarity between golden and predicted answers based on cosine similarity.
    """
    
    if not golden_answer or not predicted_answer:  # Check if the input lists are empty
        print("The input lists are empty, unable to calculate similarity.")
        return 0, []

    # Encode the golden and predicted answers
    encoded_golden_answer = model.encode(golden_answer)
    encoded_predicted_answer = model.encode(predicted_answer)

    # Calculate the cosine similarity matrix
    sim_matrix = cosine_similarity(encoded_golden_answer, encoded_predicted_answer)

    hits = 0
    highest_similarities = []

    for i, core_point1 in enumerate(golden_answer):
        max_similarity = np.max(sim_matrix[i]) 
        max_index = np.argmax(sim_matrix[i])  

        
        if max_similarity >= threshold:
            hits += 1

        highest_similarities.append((core_point1, predicted_answer[max_index], float(max_similarity)))

    # Calculate the hit rate (percentage of matches)
    hit_rate = hits / len(golden_answer) if golden_answer else 0

    return hit_rate, highest_similarities




def extract_all_argument_pairs(init_path, new_path, output_path, summary_output_path, threshold):
    """
    Compare the arguments in two versions (initial and new), calculate similarity, and save the results.
    """
    
    init_data = load_json(init_path)  # Load the initial version data
    new_data = load_json(new_path)  # Load the new version data

    # Create title -> argument mappings for fast lookup
    new_map = {item['title']: item['argument'] for item in new_data}
    init_map = {item['title']: item['argument'] for item in init_data}

    results = []
    hit_rates = []  # List to store hit rates for each article
    total_hit_rate = 0
    count = 0

    # Compare arguments for each article
    for item in new_data:
        title = item['title']
        init_args = init_map.get(title, [])
        new_args = item.get('argument', [])

        # Calculate similarity and hit rate
        hit_rate, similarities = calculate_similarity(init_args, new_args, threshold)

        # Accumulate hit rate for average calculation
        total_hit_rate += hit_rate
        count += 1
        hit_rates.append(hit_rate)

        # Store results for each article
        results.append({
            "title": title,
            "init_argument": init_args,
            "new_argument": new_args,
            "hit_rate": round(hit_rate, 4),
            "similarity_pairs": similarities  # Optional: see pairs and similarity values
        })

    
    avg_hit_rate = total_hit_rate / count if count > 0 else 0
    print(f"Average hit rate: {avg_hit_rate:.4f}")

    # Randomly sample 100 hit rates and calculate the average
    random_sample_100 = random.sample(hit_rates, 100) if len(hit_rates) >= 100 else hit_rates
    avg_random_100 = np.mean(random_sample_100)
    print(f"Random sample 100 average hit rate: {avg_random_100:.4f}")

    
    if output_path:
        save_json(output_path, results)
    
    
    if summary_output_path:
        summary_results = {
            "avg_hit_rate": avg_hit_rate,
            "avg_random_100": avg_random_100,
        }
        save_json(summary_output_path, summary_results)

    return results, avg_hit_rate



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate similarity between golden and predicted argument pairs.")

    parser.add_argument("--golden-path", type=str, required=True, help="Path to the human written JSON file.")
    parser.add_argument("--predicted-path", type=str, required=True, help="Path to the new generation JSON file.")
    parser.add_argument("--output-path", type=str, required=False, help="Path to save the output JSON results.")
    parser.add_argument("--summary-output-path", type=str, required=False, help="Path to save the summary JSON results.")
    parser.add_argument("--threshold", type=float, required=False, help="Similarity threshold for matching .")



    args = parser.parse_args()

    # Call the extract function with arguments from the CLI
    extract_all_argument_pairs(args.golden_path, args.predicted_path, args.output_path, args.summary_output_path,args.threshold)



