
from openai import OpenAI
import re
import json
import os
import concurrent.futures
import argparse


model = 'XXX'
client = OpenAI(
    api_key='XXX',
    
)




def generate_outline_zero_shot_v1(text):
    try:
        
        prompt = f"""
        
        文本内容：
        {text}

        请按照以下步骤分析文章内容并提取关键信息：

        **第一步：划分文章的结构**
        - 根据内容，将文章按顺序划分为四个主要部分：引言部分、分析部分、建议部分和结论部分
        - 可以参考文章的标题、段落主题句、逻辑流程或关键词等判断各部分的划分。

        **第二步：提取文章研究的问题**
        - 明确文章的研究主题是什么？主要问题和核心问题是什么？
        - 这部分的目标是概述文章想要解决的问题及其背景。

        **第三步：分析部分的详细内容,注意建议不属于分析部分**
        - 在分析部分中，根据逻辑划分生成多个小节，其中一个小节可以是一个段落也可以是多个段落组成，每个小节包含以下要素：
          1. 提取每一小节的核心观点是什么？
          2. 总结每小节讨论的主题是什么？既该小节是从哪个角度来分析这个研究问题的。注意主题要求简洁、不带感情色彩，不带偏见，不带倾向，不带观点结论,长度不超过10个字
          3. 总结每个小节是为了回答什么问题？即撰写这个小节的目的是什么
          4. 分析部分的结论是什么？
          5. 这个小节是从哪些方面进行分析 得出这个核心观点的？列出这些角度，用顿号隔开

        **第四步：政策建议内容**
        - 文章是否提供了具体的政策建议？如果没有输出没有 否则，列出不同的建议角度[给出角度 不用给出具体动作]，对每个角度进行简要描述。说明这些建议主要是为了解决哪些关键问题或目标？每个政策建议包含以下内容
        1. 政策建议：[给出角度]
        2. 建议简述:[简短 一句话以内]
        3. 建议主要是为了解决哪些关键问题或目标[]
        
        输出格式如下：
        **第一步：划分文章的结构**

        1. **引言部分**：[]
        2. **分析部分**：[]
        3. **建议部分**：[]
        4. **结论部分**：[]

        **第二步：提取文章研究的问题**

        - **研究主题**：[]
        - **主要问题**：[]

        **第三步：分析部分的详细内容**

        1. **第一节**：[]
        - 核心观点：[]
        - 主题：[]
        - 问题：[]
        - 结论：[]
        - 分析角度：[]
        

        2. **第二节**：[]
        - 核心观点：[]
        - 主题：[]
        - 问题：[]
        - 结论：[]
        - 分析角度：[]
        
        ...
        N. **第N节**：[]
        - 核心观点：[]
        - 主题：[]
        - 问题：[]
        - 结论：[]
        - 分析角度：[] 
        

        **第四步：政策建议内容**
        - **政策建议1**：[]
        - **建议简要概述**：[]
        - **针对的问题或目标**：
        ....
        - **政策建议n**：[]
        - **建议简要概述**：[]
        - **针对的问题或目标**：[]
        请严格按照以上步骤提取关键信息，确保输出内容清晰、简洁、结构化。

        """
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can extract and analyze structured content from complex articles."},
            {"role": "user", "content": prompt},
        ]
        
        # 调用模型生成结果
        chat_response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.01,
            
        )
        
        # 提取并返回生成的答案
        answer = chat_response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error in qwen_request: {e}")
        return []



def extract_key_elements_from_result(result_text):
    try:
       
        research_theme_match = re.search(r"\*\*研究主题\*\*：(.+?)\n", result_text)
        research_theme = research_theme_match.group(1).strip().rstrip("。") if research_theme_match else "未找到研究主题"
        
        main_issues_match = re.search(r"\*\*主要问题\*\*：(.+?)\n", result_text)
        main_issues = main_issues_match.group(1).strip().rstrip("。") if main_issues_match else "未找到主要问题"

        
        analysis_sections = []
        analysis_section_pattern = r"\d+\.\s\*\*([^：]+)\*\*：(.*?)\n\s*-\s*核心观点：(.+?)\n\s*-\s*主题：(.+?)\n\s*-\s*问题：(.+?)\n\s*-\s*结论：(.+?)\n\s*-\s*分析角度：(.*?)\n"
        matches = re.findall(analysis_section_pattern, result_text, re.DOTALL)
        
        for match in matches:
            section = {
                "核心观点": match[2].strip() if match[2] else "",
                "主题": match[3].strip(),
                "问题": match[4].strip(),
                "结论": match[5].strip(),
                "分析角度": [angle.strip() for angle in match[6].split("、")],
            }
            analysis_sections.append(section)

        
        policy_advice_pattern = r"\*\*政策建议\d+\*\*：([^\n]+)"
        policy_advice_matches = re.findall(policy_advice_pattern, result_text)

        policy_target_pattern = r"-\s*(?:\*\*?)?针对的问题或目标(?:\*\*?)?：\s*([^\n]+)"
        policy_target_matches = re.findall(policy_target_pattern, result_text)
        

        policy_target_matches = policy_target_matches if policy_target_matches else []

        
        policy_advice_data = []
        for advice, target in zip(policy_advice_matches, policy_target_matches):
            policy_advice_data.append({
                "政策建议内容": advice.strip().rstrip("。"),
                "针对的问题或目标": target.strip().rstrip("。")
            })

        result = {
            "研究主题": research_theme,
            "主要问题": main_issues,
            "分析部分": analysis_sections,
            "政策建议": policy_advice_data,
        }
        return result

    except Exception as e:
        print(f"Error in extract_key_elements_from_result: {e}")
        result = {
            "研究主题": "提取失败",
            "主要问题": "提取失败",
            "分析部分": [],
            "政策建议": [],
        }
        return result




def extract_outline_txt(title, data_dict, output_dir):
    """
    Save the dictionary content as a .txt file in the specified directory.
    The filename is based on the article's title, with invalid characters replaced.
    """
    # Get the research theme and use it as the main heading
    research_theme = data_dict.get("研究主题", "")
    txt_content = f"# {research_theme}\n"

    # Extract the themes from the analysis section and use them as subheadings
    analysis_parts = data_dict.get("分析部分", [])
    analysis_themes = []

    # Loop through analysis parts and extract each subsection's theme
    for section in analysis_parts:
        theme = section.get("主题")
        if theme:
            analysis_themes.append(theme)
    
    # Add subheadings for each theme
    for theme in analysis_themes:
        txt_content += f"## {theme}\n"

    # Save the content to the specified file path
    output_file_path = os.path.join(output_dir, "arc", "outline", f"{title}_outline.txt")
    write_str(txt_content, output_file_path)
    print(f"Content successfully saved to: {output_file_path}")


def article_generate_frame_batch(json_path, output_dir, start_index=0, n=None):
    """
    Batch process to extract evidence information from articles.

    :param json_path: Path to the JSON file containing article data.
    :param output_dir: Directory to save the extracted article frames.
    :param n: Optional parameter to specify processing only the first n articles. If None, process all articles.
    """
    try:
        # Load articles from the JSON file
        articles = load_json(json_path)

        # If n is specified, process only the first n articles
        if n is not None:
            articles = articles[:n]

        # Process articles starting from start_index
        articles = articles[start_index:]
        
        for index, article in enumerate(articles, start=start_index):
            # Get article title and content
            title = article.get("title", "")
            article_text = article.get("content", "")
            
            if not article_text:
                print(f"Article '{title}' has no content, skipping.")
                continue

            # Generate outline using the zero-shot model
            result_init = generate_outline_zero_shot_v1(article_text)
            result_frame = extract_key_elements_from_result(result_init)

            # Save the frame result to the specified path
            output_file_path = os.path.join(output_dir, "arc", "frame", f"{title}_frame.json")
            dump_json(result_frame, output_file_path)
            print(f"Processing article {index + 1}: '{title}'")
            print(f"Results saved to {output_file_path}")

            # Save the article outline as a text file
            extract_outline_txt(title, result_frame, output_dir)
            print("Outline results saved.")
        
        print("Batch processing completed!")
        return None

    except Exception as e:
        print(f"Error during processing: {e}")
        return None





###################

##################

def dump_json(obj, file_name, encoding="utf-8"):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w', encoding=encoding) as fw:
        json.dump(obj, fw, ensure_ascii=False,indent=4)

def load_json(file_path, n=None):
    
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if n is not None:
            data = data[:n]
        return data
    except Exception as e:
        print(f"Error: {e}")
        return []

def write_str(s, path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(s)


def load_str(path):
    with open(path, 'r') as f:
        return '\n'.join(f.readlines())

def restore_article_frame_extract_direct(json_path, input_dir, output_dir, start=0, end=None):
    """
    Extract article frames from .txt files in the specified folder and save the results.

    Args:
        json_path (str): Path to the JSON file containing article titles.
        input_dir (str): Directory containing the article body files.
        output_dir (str): Directory to save the extracted frames.
        start (int, optional): Starting index for processing articles. Default is 0.
        end (int, optional): Ending index for processing articles. If None, all articles are processed.
    """
    try:
        # Load articles from the JSON file
        articles = load_json(json_path)
        if start is not None and end is not None:
            articles = articles[start:end]

        # Process each article
        for index, article in enumerate(articles):
            title = article.get("title", "")
            if not title:
                print("Article title is empty, skipping.")
                continue
            print(f"Processing article {index + 1}: '{title}'")  # Display current article index and title

            # Construct the file path for the article body
            article_file_path = os.path.join(input_dir, f"{title}.txt")
            if not os.path.exists(article_file_path):
                print(f"File not found: {article_file_path}, skipping.")
                continue

            # Read the article body
            with open(article_file_path, 'r', encoding='utf-8') as file:
                article_text = file.read()

            # Extract the frame using the zero-shot model
            result_init = generate_outline_zero_shot_v1(article_text)
            result_frame = extract_key_elements_from_result(result_init)

            # Construct the output file path and save the extracted result
            output_file_path = os.path.join(output_dir, f"{title}_frame.json")
            dump_json(result_frame, output_file_path)
            print(f"Processed article '{title}' successfully, results saved to: {output_file_path}")

        print("All articles processed successfully!")

    except Exception as e:
        print(f"Error occurred during processing: {e}")

def extract_frames_from_md_folder(input_dir, output_dir):
    """
    Traverse the specified folder for Markdown articles (without file extensions), 
    extract frames, and save the results as JSON files.

    Args:
        input_dir (str): Directory containing Markdown text files.
        output_dir (str): Directory to save the extracted results as JSON files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filenames = os.listdir(input_dir)

    for filename in filenames:
        file_path = os.path.join(input_dir, filename)

        # Skip directories and files with extensions (process only files without extensions)
        if not os.path.isfile(file_path) or '.' in filename:
            continue

        try:
            print(f"Processing file: {filename}")

            # Read the content of the Markdown file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Perform frame extraction
            print(content)
            result_init = generate_outline_zero_shot_v1(content)
            result_frame = extract_key_elements_from_result(result_init)

            # Save the result as a JSON file
            output_file = os.path.join(output_dir, f"{filename}_frame.json")
            dump_json(result_frame, output_file)

            print(f"Extraction completed: {filename} → {output_file}")

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    print("All Markdown files processed successfully!")

def article_generate_frame_one(input_dir, output_dir):
    """
    Read the content of the specified file, extract question-focus, 
    and save the results as a JSON file.

    Args:
        input_dir (str): Path to the Markdown text file.
        output_dir (str): Directory to save the extracted result as a JSON file.
    """
    
    try:
        # Read the Markdown file content
        with open(input_dir, 'r', encoding='utf-8') as file:
            content = file.read()

        # Perform frame extraction
        print(content)
        result_init = generate_outline_zero_shot_v1(content)
        result_frame = extract_key_elements_from_result(result_init)

        # Save the result as a JSON file
        output_file = os.path.join(output_dir, "frame.json")
        dump_json(result_frame, output_file)

        print("Extraction completed")

    except Exception as e:
        print(f"Error processing the file: {e}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process articles in batches using multi-threading.")
    
    # Adding command line arguments
    parser.add_argument("--json-path", type=str, required=True, help="Path to JSON with golden articles (must include 'title' and 'content').")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing predicted articles named as <title>.txt.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the output results.")
    parser.add_argument("--total-articles", type=int, required=True, help="Total number of articles to process.")
    parser.add_argument("--batch-size", type=int, required=True, help="Number of articles to process per batch.")
    
    args = parser.parse_args()

    total_articles = args.total_articles
    batch_size = args.batch_size
    num_threads = total_articles // batch_size

    # Using ThreadPoolExecutor to process articles in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            futures.append(executor.submit(restore_article_frame_extract_direct, args.json_path, args.input_dir, args.output_dir, batch_start, batch_end))

        # Waiting for all threads to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")



if __name__ == "__main__":
    
    extract_frames_from_md_folder(
    input_dir="/Users/wangyini/Desktop/code/storm-cursor-clean/tmp",
    output_dir="/Users/wangyini/Desktop/code/storm-cursor-clean/md_output"
)
    article_generate_frame_one(input_dir='/Users/wangyini/Desktop/code/storm-cursor-clean/tmp/a',output_dir='/Users/wangyini/Desktop/code/storm-cursor-clean/md_output')
    








