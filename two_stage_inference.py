# two_stage_inference.py
import json
import time
import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fastchat.serve.inference import load_model

from prefill import generate_text_with_kv_cache
from decode import continue_text_with_kv_cache

def two_stage_inference(input_texts, model_name='gpt2', iter_max_length=50, batch_size=2):
    # 第一阶段：Prefill，生成KV缓存
    # 我们只生成一个token，所以max_length设置为当前文本长度加1
    prefill_texts, kv_caches = generate_text_with_kv_cache(
        input_texts, 
        model_name=model_name, 
        max_length=[len(text) + 1 for text in input_texts],
        batch_size=batch_size
    )

    # 第二阶段：Decode，继续生成文本
    continued_texts, _ = continue_text_with_kv_cache(
        prefill_texts, 
        kv_caches, 
        model_name=model_name, 
        max_length=iter_max_length, 
        batch_size=batch_size
    )

    return continued_texts

def base_inference(input_texts, model_name='gpt2', batch_size=2):
    # 确保CUDA可用
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    device = "cuda"

    # 加载模型和分词器
    model, tokenizer = load_model(model_name = model_name, device = device, num_gpus = 1)

    model.to(device)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.unk_token

    result = []

    input_texts = [input_texts[i:i+batch_size] for i in range(0, len(input_texts), batch_size)]

    for batch in input_texts:
    # 将输入文本编码为批处理
        inputs = tokenizer(batch, return_tensors='pt', padding=True).to(device)
        outputs = model(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'])
        
        for sentence in outputs:
            result.append(tokenizer.decode(sentence, skip_special_tokens=True))

    return result
    

# 示例输入
with open('extracted_human_conversations.json', 'r') as file:
    data = json.load(file)  # 更多文本

input_texts = data[:512]
input_texts = ["Hi, I am a robot", "Hi, I am a human"]

model_name = "../vicuna-1.5-7b"
batch_size = 16
iter_max_length = 20

base_start_time = time.time()
output_texts = base_inference(input_texts, model_name = model_name, batch_size = batch_size)
base_end_time = time.time()
base_execution_time = base_end_time - base_start_time

for text in output_texts:
    print(text)

# 执行两阶段推理
my_start_time = time.time()
output_texts = two_stage_inference(input_texts, model_name = model_name, iter_max_length= iter_max_length, batch_size = batch_size)
my_end_time = time.time()
my_execution_time = my_end_time - my_start_time

# 打印输出文本
for text in output_texts:
    print(text)

print(f"my time : {my_execution_time} sec, base time : {base_execution_time} sec")