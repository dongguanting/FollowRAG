import json
import datetime

def read_json(read_file_path):
    with open(read_file_path,encoding='utf-8') as file:
        raw_data=json.load(file)
    return raw_data

def save_json(save_file_path,data,mode='w'):
    with open(save_file_path,mode,encoding='utf-8')as fp:
        json.dump(data,fp,ensure_ascii=False)

def read_jsonl(read_file_path):
    data = []
    with open(read_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(save_file_path, data, mode='w'):
    assert mode in ['w','a']
    with open(save_file_path, mode, encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')

def read_txt(read_file_path):
    try:
        with open(read_file_path, 'r', encoding='utf-8') as file:
            content = file.read() 
        return content
    except FileNotFoundError:
        return f"Error: The file '{read_file_path}' was not found."
    except Exception as e:
        return f"An error occurred: {e}"
    
def append_to_file(filename, content):
    """
    Append the specified string to the txt file and add a line break after the content
    - param filename: filename (including path)
    - param content: The string to be written
    """
    try:
        with open(filename, 'a', encoding='utf-8') as file:
            file.write(content + '\n')
    except Exception as e:
        print(f"Error occurred while appending to file: {e}")

def parse_str_to_dict(s):
    s=s.replace("'", '"')
    s_dict=json.loads(s)
    return s_dict

def get_current_time():
    current_time = datetime.datetime.now()
    return current_time.strftime("%Y-%m-%d %H:%M")