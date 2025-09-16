
from openai import OpenAI
import re
import json
import os
from argparse import ArgumentParser


#OpenAI Qwen客户端
model = 'XXX'
client = OpenAI(
    api_key='XXX',
    
)




def extract_evident_3(text):
    try:
        # 设计优化后的prompt
        prompt = f"""
        原文：
        {text}

        请按照以下步骤分析原文内容并提取关键信息：

        **第一步：划分文章的结构**
        - 根据内容，将文章按顺序完整划分为四个主要部分：引言背景部分、分析部分、建议部分和结论部分。
        - 可以参考文章的标题、段落主题句、逻辑流程或关键词来判断各部分的划分。

        **第二步：提取文章引言背景部分的信息**
        - 引言背景部分的关键事实有哪些？

        **第三步：分析部分的详细内容,注意建议不属于分析部分**
        - 在分析部分中，根据逻辑划分生成多个小节，其中一个小节可以是一个段落也可以是多个段落组成，每个小节包含以下要素：
          1. 总结每一小节的核心观点是什么？
          2. 提取该小节的主要结论(观点)有哪些，以及支持这些论点的关键证据（用于支撑观点的事实、数据、例子、统计数字、专家意见等）是什么？ 
             注意观点结论和证据是不一样的，结论是作者的观点或主张，论据是支持结论的证据。

        以##号结束 代表文本输出完成
        输出格式如下：
        **第一步：划分文章的结构**
        1. **引言部分**:[段落范围]
        2. **分析部分**:[段落范围]
        3. **建议部分**:[段落范围]
        4. **结论部分**:[段落范围]
        
        **第二步：提取文章引言背景部分的信息**
        - 事实1：[]
        - 事实2：[]
        ...
        - 事实n：[]
        
        **第三步：分析部分的详细内容**

        1. **第一节**
        - 核心观点：[]
        - 主要结论1：[]
            -论据1：[]
            -论据2：[]
            ...
            -论据n：[]
        - 主要结论2：[]
            -论据1：[]
            -论据2：[]
            ...
            -论据n：[]
        ...
        - 主要结论n：[]
            -论据1：[]
            -论据2：[]
            ...
            -论据n：[]

        2. **第二节**
        - 核心观点：[]
        - 主要结论1：[]
            -论据1：[]
            -论据2：[]
            ...
            -论据n：[]
        - 主要结论2：[]
            -论据1：[]
            -论据2：[]
            ...
            -论据n：[]
        ...
        - 主要结论n：[]
            -论据1：[]
            -论据2：[]
            ...
            -论据n：[]
        ...
        N. **第N节**
        - 核心观点：[]
        - 主要结论1：[]
            -论据1：[]
            -论据2：[]
            ...
            -论据n：[]
        - 主要结论2：[]
            -论据1：[]
            -论据2：[]
            ...
            -论据n：[]
        ...
        - 主要结论n：[]
            -论据1：[]
            -论据2：[]
            ...
            -论据n：[]

        # 结束
        请严格按照以上步骤提取关键信息，确保输出内容清晰、简洁、结构化。
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can extract and analyze structured content from complex articles."},
            {"role": "user", "content": prompt},
        ]
        chat_response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.01,
            top_p=0.8
        )
        
        # 提取并返回生成的答案
        answer = chat_response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error in processing article: {e}")
        return []



def extract_sections_3(output_text):
    """
    Extracts facts, core views, conclusions, and evidence from the provided text using regular expressions.
    """
    # Initialize the result dictionary
    result = {
        "第二部分_事实": [],
        "第三部分_分析": []  # The analysis part will be split into sections
    }

    # Extract all facts from the second part
    facts = re.findall(r"- 事实\d+\s*：\s*(.+)", output_text)
    result["第二部分_事实"] = facts

    # Extract the analysis section, starting from "**第三步：分析部分的详细内容**"
    analysis_part = re.search(r"\*\*第三步：分析部分的详细内容\*\*(.*)", output_text, re.DOTALL)

    if analysis_part:
        analysis_text = analysis_part.group(1).strip()
        
        # Match each analysis section
        sections = re.findall(r"\d+\. \*\*(.+?)\*\*(.*?)(?=\d+\. \*\*|\# 结束)", analysis_text, re.DOTALL)
        
        for section in sections:
            section_info = {
                "核心观点": "",  # Core view
                "主要结论": []  # List to hold conclusions
            }

            # Extract the core view
            core_view = re.search(r"- 核心观点\s*：\s*(.+)", section[1])
            if core_view:
                section_info["核心观点"] = core_view.group(1).strip()

            # Extract the main conclusions and evidence
            conclusions = re.findall(r"- 主要结论\d+\s*：\s*(.+?)(?=\n\s*- 主要结论\d+：|\n\s*# 结束|\n\s*$)", section[1], re.DOTALL)
            
            for conclusion_text in conclusions:
                conclusion_info = {
                    "结论": "",  # Conclusion content
                    "论据": []  # List to hold evidence
                }

                # Extract the conclusion content
                conclusion_match = re.search(r"^(.+?)(?=\n\s*- 论据\d+：|\n\s*$)", conclusion_text, re.DOTALL)
                if conclusion_match:
                    conclusion_info["结论"] = conclusion_match.group(1).strip()

                # Extract evidence supporting the conclusion
                evidences = re.findall(r"- 论据\d+\s*：\s*(.+?)(?=\n\s*- 论据\d+：|\n\s*$|\n\n|\Z)", conclusion_text, re.DOTALL)
                for evidence in evidences:
                    conclusion_info["论据"].append(evidence.strip())

                section_info["主要结论"].append(conclusion_info)

            # Add the analysis section info to the result
            result["第三部分_分析"].append(section_info)
    
    return result


def article_evident_extract_batch(json_path, output_dir, start_index=0, n=None):
    """
    Batch process to extract evidence information from articles.

    Args:
        json_path (str): Path to the JSON file containing article data.
        output_dir (str): Directory to save the extracted results.
        start_index (int): Index to start processing articles from.
    """
    try:
        # Load articles from the JSON file
        articles = load_json(json_path)

        # If n is specified, process only the first n articles
        if n is not None:
            articles = articles[:n]

        # Process articles starting from the start_index
        articles = articles[start_index:]
        
        for index, article in enumerate(articles, start=start_index):
            # Get the title and content of the article
            title = article.get("title", "")
            article_text = article.get("content", "")
            
            if not article_text:
                print(f"Article '{title}' has no content, skipping.")
                continue
            
            # Extract evidence and structured information
            result_init = extract_evident_3(article_text)
            result_extract = extract_sections_3(result_init)
        
            # Save the result to the specified path
            output_file_path = os.path.join(output_dir, "arc", "evident", f"{title}_evident.json")
            dump_json(result_extract, output_file_path)
            print(f"Processing article {index + 1}: '{title}'")
            print(f"Result saved to {output_file_path}")
        
        print("Batch processing completed!")
        return None

    except Exception as e:
        print(f"Error during processing: {e}")
        return None





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







# Main function with argument parsing
def main():
    # Initialize argument parser
    parser = ArgumentParser()

    # Define command line arguments
    parser.add_argument("--json-path", type=str, required=True, 
                        help="Path to JSON with golden articles (must include 'title' and 'content').")
    parser.add_argument("--output-dir", type=str, required=True, 
                        help="Directory to store the output results.")
    parser.add_argument("--start", type=int, required=True, 
                        help="Start index of the articles to process.")
    parser.add_argument("--end", type=int, required=True, 
                        help="End index (exclusive) of the articles to process.")

    # Parse command line arguments
    args = parser.parse_args()

    # Call the batch extraction function with parsed arguments
    article_evident_extract_batch(args.json_path, args.output_dir, args.start, args.end)

if __name__ == "__main__":
    main()



