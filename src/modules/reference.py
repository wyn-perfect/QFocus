
import os
import json

def extract_elements_from_evident(json_path):
    """
    Extract key elements (facts, core views, conclusions, and evidences) from the provided JSON file.

    Args:
        json_path (str): Path to the JSON file containing the evident data.

    Returns:
        dict: A dictionary containing extracted facts, core views, conclusions, and evidences.
    """
    result = load_json(json_path)  # Load the JSON data
    print(result)
    
    # Initialize result lists
    facts = result.get("第二部分_事实", [])  # Extract facts
    core_views = []
    conclusions = []
    evidences = []

    # Extract core views, conclusions, and evidences from the analysis section
    for section in result.get("第三部分_分析", []):
        core_views.append(section.get("核心观点", ""))
        for conclusion_info in section.get("主要结论", []):
            conclusions.append(conclusion_info.get("结论", []))
            evidences.extend(conclusion_info.get("论据", []))
    
    # Return the extracted elements as a dictionary
    extracted_evident = {
        "事实": facts,  # List of facts
        "核心观点": core_views,  # List of core views
        "主要结论": conclusions,  # List of conclusions
        "论据": evidences  # List of evidences
    }
    return extracted_evident

def extract_elements_from_frame(json_path):
    """
    Extract key elements (research topic, main question, core views, etc.) from the provided frame JSON file.

    Args:
        json_path (str): Path to the JSON file containing the frame data.

    Returns:
        dict: A dictionary containing extracted elements like research topic, core views, conclusions, etc.
    """
    data = load_json(json_path)  # Load the JSON data

    # Extract key fields
    research_topic = data.get("研究主题", "")
    main_question = data.get("主要问题", "")
    
    core_views = [item.get("核心观点", "") for item in data.get("分析部分", [])]
    themes = [item.get("主题", "") for item in data.get("分析部分", [])]
    questions = [item.get("问题", "") for item in data.get("分析部分", [])]
    conclusions = [item.get("结论", "") for item in data.get("分析部分", [])]
    analysis_angles = [item.get("分析角度", []) for item in data.get("分析部分", [])]

    # Organize the extracted fields into a new dictionary
    extracted_frame = {
        "研究主题": research_topic,
        "主要问题": main_question,
        "核心观点": core_views,
        "主题": themes,
        "问题": questions,
        "结论": conclusions,
        "分析角度": analysis_angles,
    }
    
    return extracted_frame



####################
#### 读文件 存文件
###################

def dump_json(obj, file_name, encoding="utf-8"):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w', encoding=encoding) as fw:
        json.dump(obj, fw, ensure_ascii=False,indent=4)

def load_json(file_path, n=None):
    # 从指定路径读取 JSON 文件，并返回前 n 条数据作为字典列表
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if n is not None:
            data = data[:n]
        return data
    except Exception as e:
        print(f"Error: {e}")
        return []

