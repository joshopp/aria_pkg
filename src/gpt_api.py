import re
import string
import pandas as pd



def process_csv_and_find_timestamps(csv_file, question, gpt_output):
    gpt_words = gpt_output.get("words", [])
    timestamp_result = {'startTime_ns': [], 'endTime_ns': []}
    question = {'question': [question]}
    reader = pd.read_csv(csv_file)
        
    for gpt_word in gpt_words:
        gpt_word_cleaned = gpt_word.translate(str.maketrans('', '', string.punctuation)).strip().lower() # delete punctuation and change all words to small
        for index, row in reader.iterrows():
            csv_word = row['written'].lower()
            csv_word_cleaned = csv_word.translate(str.maketrans('', '', string.punctuation)).strip()
            if gpt_word_cleaned == csv_word_cleaned:
                timestamp_result['startTime_ns'].append(row['startTime_ns'])
                timestamp_result['endTime_ns'].append(row['endTime_ns'])
                reader = reader.drop(index) # delete corresbond word  eg pick this cup and that cup
                break
    gpt_output.update(timestamp_result)
    gpt_output.update(question)
    return gpt_output


def combine_written_to_string(csv_file):
    # Read the CSV file and extract 'written' column, clean, and combine into a single string
    combined_string = ""
    with open(csv_file, 'r') as file:
        reader = pd.read_csv(csv_file)
        for _, row in reader.iterrows():
            written = row['written'].lower()
            cleaned_written = written.translate(str.maketrans('', '', string.punctuation))
            combined_string += cleaned_written 
    
    return combined_string.strip()


def ask(client, chat_history, prompt):
    chat_history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    completion =  client.chat.completions.create(
        model="gpt-4o",
        messages=chat_history,
        response_format={
        "type": "text"
        },
        temperature=1
    )
    chat_history.append(
        {
            "role": "assistant",
            "content": completion.choices[0].message.content,
        }
    )
    return chat_history[-1]["content"]


def extract_python_code(content):
    code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)

        if full_code.startswith("json"):
            full_code = full_code[5:]

        return full_code
    else:
        return None


def extract_json(code_block_regex, content):
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)

        if full_code.startswith("json"):
            full_code = full_code[7:]

        return full_code
    else:
        return None

