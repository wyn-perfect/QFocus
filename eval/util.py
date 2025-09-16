import os
import json

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

def write_str(s, path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(s)

def load_str(path):
    with open(path, 'r') as f:
        return '\n'.join(f.readlines())