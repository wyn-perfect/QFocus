
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import concurrent.futures
from tqdm import tqdm
import re
import json
import os
from argparse import ArgumentParser


model = 'XXX'
client = OpenAI(
    
    api_key='XXX',  # 替换为您的实际API密钥
)




def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    

def save_json(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def extract_evident(text):
    try:
        prompt = f"""
        文本内容：{text}
        你是一个信息抽取专家，请从文本中抽取所有支持作者观点的“论据”，每一条论据必须是原文中的完整句子，无需改写或缩写。
        请以列表形式输出所有论据句子，并保持原文风格，不进行任何加工。不要输出观点总结或评论，只列出论据句子本身。
        
        输出格式如下：

        论据1：xxx。
        论据2：xxx。
        ...
        论据n：xxx。
        """

        messages = [
            {"role": "system", "content": "You are a helpful assistant that can extract and analyze structured content from complex articles."},
            {"role": "user", "content": prompt},
        ]
        
       
        chat_response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.01,
            
        )
        
        answer = chat_response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error in processing article: {e}")
        return []


def arguments_extract(text):
    pattern = r'论据\d+：(.+?。)'  
    arguments = re.findall(pattern, text)
    
    return arguments



def extract_init_text(filepath,start=0,end=None):
    data = load_json(filepath)
    
    if end is not None:
        data = data[start:end]
    else:
        data = data[start:]

    
    results = []
    for article in tqdm(data):
        title = article.get('title', '无标题')  # 获取文章标题，若无则默认为"无标题"
        content = article.get('content', '无内容')  # 获取文章内容，若无则默认为"无内容"
        result = extract_evident(content)
        #print(result)
        element = arguments_extract(result) # 这是列表的形式
        entry = {
            "title":title,
            "argument":element
        }
        results.append(entry)
    return results
   

def parallel_extract(file_path,output_path,total_articles):
    
    batch_size = 1
    num_threads = total_articles // batch_size
    all_results = []  

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            futures.append(executor.submit(extract_init_text, file_path, batch_start, batch_end))

        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()  
                all_results.extend(result)  
            except Exception as e:
                print(f"An error occurred: {e}")
    save_json(output_path,all_results)



def extract_txt_batch(file_list, folder_path):
    result = []
    for filename in tqdm(file_list):
        title = os.path.splitext(filename)[0]
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            content = ''
        
        tmp = extract_evident(content)
        element = arguments_extract(tmp) 
        entry = {
            "title":title,
            "argument":element
        }
        result.append(entry)
    return result



def parallel_extract_txt(folder_path,output_file):
   
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    total_files = len(all_files)
    batch_size = 1
    num_batches = (total_files + batch_size - 1) // batch_size
    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_batches) as executor:
        futures = []
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_files)
            batch_files = all_files[start:end]
            futures.append(executor.submit(extract_txt_batch, batch_files, folder_path))

        # 收集结果（保证顺序）
        for future in futures:
            try:
                result = future.result()
                all_results.extend(result)
            except Exception as e:
                print(f"An error occurred: {e}")
    save_json(output_file,all_results)



if __name__ == "__main__":
    parser = ArgumentParser()
    
    # 添加命令行参数
    parser.add_argument("--json-path", type=str, required=True, help="Path to JSON with golden articles (must include 'title' and 'content').")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing predicted articles named as <title>.txt.")
    parser.add_argument("--golden-output-dir", type=str, required=True, help="Directory to save golden results JSON.")
    parser.add_argument("--predicted-output-dir", type=str, required=True, help="Directory to save predicted results JSON.")
    parser.add_argument("--total-articles", type=int, default=0, help="Total number of articles to process.")
       
    args = parser.parse_args()

    parallel_extract(args.json_path, args.golden_output_dir,args.total_articles)
    parallel_extract_txt(args.input_dir, args.predicted_output_dir)
    


