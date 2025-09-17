import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse
from util import dump_json, load_json

model = SentenceTransformer('BAAI/bge-base-zh-v1.5')


# Define a function to compute semantic similarity between two lists of core points
def calculate_similarity(core_points1, core_points2, threshold):
    """
    Calculate similarity between two lists of core points and return hit rate and top matches.

    """

    # Return if either list is empty
    if not core_points1 or not core_points2:
        print("Input core point list is empty, cannot calculate similarity.")
        return 0, []

    # Encode core points
    encoded_core_points1 = model.encode(core_points1)
    encoded_core_points2 = model.encode(core_points2)

    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(encoded_core_points1, encoded_core_points2)

    hits = 0
    highest_similarities = []

    # Iterate through each core point
    for i, core_point1 in enumerate(core_points1):
        # Find the most similar point in the second list
        max_similarity = np.max(sim_matrix[i])
        max_index = np.argmax(sim_matrix[i])

        # Count as a hit if similarity exceeds threshold
        if max_similarity >= threshold:
            hits += 1

        # Save the best match and similarity score
        highest_similarities.append(
            (core_point1, core_points2[max_index], float(max_similarity))
        )

    # Compute hit rate
    hit_rate = hits / len(core_points1) if core_points1 else 0

    return hit_rate, highest_similarities



def extract_core_points_and_conclusions(frame_path):
    """
    Extract core points and conclusions from a frame file.

    Args:
        frame_path (str): Path to the frame file.

    Returns:
        tuple: (list of core points, list of conclusions)
    """
    try:
        frame_data = load_json(frame_path)

        # Initialize storage for core points and conclusions
        core_points = []
        conclusions = []

        # Collect core points and conclusions from each section
        for section in frame_data.get("分析部分", []):
            core_points.append(section.get("核心观点", ""))
            conclusions.append(section.get("结论", ""))

        return core_points, conclusions

    except Exception as e:
        print(f"Error in extract_core_points_and_conclusions: {e}")
        return [], []


def calculate_average_values(all_values):
    all_avg = np.mean(all_values)
    top200_avg = np.mean(sorted(all_values, reverse=True)[:200])
    return all_avg, top200_avg



def evaluate_article_restoration(titles_json_path, golden_frame_dir, predicted_frame_dir, threshold, start=0, end=600, result_dir=None):   
    """
    Evaluate the quality of article restoration.

    Args:
        titles_json_path (str): Path to JSON file containing article titles
        golden_frame_dir (str): Directory of golden frame files
        predicted_frame_dir (str): Directory of restored frame files
        threshold (float): Similarity threshold
        start (int): Start index for processing
        end (int): End index for processing
        result_dir (str): Directory to save results
    """
    try:
        # Load article titles
        articles = load_json(titles_json_path)
        articles = articles[start:end]
        
        # Initialize hit rate statistics
        avg_core_points_coverage_rate_init = []
        avg_conclusions_coverage_rate_init = []

        # Iterate over each article
        for article in articles:
            title = article.get("title", "")
            if not title:
                print("Title is empty, skipping.")
                continue

            # Golden frame path
            golden_frame_path = os.path.join(golden_frame_dir, f"{title}_frame.json")
            if not os.path.exists(golden_frame_path):
                print(f"Golden frame file not found: {golden_frame_path}")
                continue

            # Extract golden core points and conclusions
            golden_core_points, golden_conclusions = extract_core_points_and_conclusions(golden_frame_path)
            
            # Predicted frame path
            predicted_frame_path = os.path.join(predicted_frame_dir, f"{title}_frame.json")
            if not os.path.exists(predicted_frame_path):
                print(f"Predicted frame file not found: {predicted_frame_path}")
                continue

            # Extract predicted core points and conclusions
            predicted_core_points, predicted_conclusions = extract_core_points_and_conclusions(predicted_frame_path)

            # Calculate similarities
            core_points_coverage_rate_init, core_points_similarities = calculate_similarity(golden_core_points, predicted_core_points, threshold)
            conclusions_coverage_rate_init, conclusions_similarities = calculate_similarity(golden_conclusions, predicted_conclusions, threshold)

            avg_core_points_coverage_rate_init.append(core_points_coverage_rate_init)
            avg_conclusions_coverage_rate_init.append(conclusions_coverage_rate_init)

            # Save intermediate results
            similarity_results_path = os.path.join(result_dir, "detail", f"{title}_similarity.json")  
            similarity_results = {
                "core_point_similarities": core_points_similarities,
                "conclusion_similarities": conclusions_similarities,
                "core_point_coverage_rate": float(core_points_coverage_rate_init),
                "conclusion_coverage_rate": float(conclusions_coverage_rate_init)
            }
            dump_json(similarity_results, similarity_results_path)

        # Compute averages
        all_avg_results = {
            "avg_core_point_coverage_rate": np.mean(avg_core_points_coverage_rate_init),
            "avg_conclusion_coverage_rate": np.mean(avg_conclusions_coverage_rate_init),
        }
        
        # Save results
        results_path = os.path.join(result_dir, "evaluation_results.json")
        dump_json(all_avg_results, results_path)
        print(f"Evaluation results saved to {results_path}")

        return all_avg_results

    except Exception as e:
        print(f"Error during processing: {e}")
        return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the restoration quality of articles.")
    
    # Adding command line arguments
    parser.add_argument("--titles-json-path", type=str, required=True, help="Path to JSON with titles and articles.")
    parser.add_argument("--golden-frame-dir", type=str, required=True, help="Directory containing the golden frame files.")
    parser.add_argument("--predicted-frame-dir", type=str, required=True, help="Directory containing the predicted frame files.")
    parser.add_argument("--result-dir", type=str, required=True, help="Directory to save the result JSON.")
    parser.add_argument("--start", type=int, default=0, help="Start index of articles to process.")
    parser.add_argument("--end", type=int, default=1, help="End index of articles to process.")
    parser.add_argument("--threshold", type=float, help="Threshold for similarity calculation (default is 0.69).")

    
    args = parser.parse_args()

    evaluate_article_restoration(args.titles_json_path, args.golden_frame_dir, args.predicted_frame_dir, args.threshold, args.start, args.end, args.result_dir)



