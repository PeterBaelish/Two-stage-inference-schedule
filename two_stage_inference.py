# two_stage_inference.py
import json
import time
import torch
import os
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fastchat.serve.inference import load_model

from prefill import generate_text_with_kv_cache
from decode import continue_text_with_kv_cache

def two_stage_inference(input_texts, model_name='gpt2', iter_max_length=50, batch_size=2):
    # 第一阶段：Prefill，生成KV缓存
    # 我们只生成一个token，所以max_length设置为当前文本长度加1
    generated_ids, kv_caches, generated_attention_mask = generate_text_with_kv_cache(
        input_texts, 
        model_name=model_name, 
        batch_size=batch_size
    )

    # 第二阶段：Decode，继续生成文本
    continued_texts, _ = continue_text_with_kv_cache(
        generated_ids, 
        kv_caches, 
        model_name=model_name, 
        max_length=iter_max_length, 
        batch_size=batch_size
    )

    return continued_texts

def interation_level_base_inference(input_texts, model_name='gpt2', max_length=50, batch_size=2):
    # 确保CUDA可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和分词器
    model, tokenizer = load_model(model_name, device, num_gpus = 1)

    model.to(device)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.unk_token

    results = []

    input_texts = [input_texts[i:i+batch_size] for i in range(0, len(input_texts), batch_size)]

    for batch in input_texts:
    # 将输入文本编码为批处理
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        l_input_ids = len(input_ids[0])
        output_ids = input_ids
        attention_mask = inputs.attention_mask.to(device)

        ending = [-1] * len(batch)

        for i in range(max_length):
            # generation
            if i == 0:
                out = model(input_ids, use_cache=True, attention_mask=attention_mask)
            else:
                out = model(
                    input_ids=token,
                    use_cache=True,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )

            # sample
            last_token_logits = out.logits[:, -1]
            token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
            output_ids = torch.cat((output_ids, token), dim=1)

            # update attn & kv cache
            past_key_values = out.past_key_values
            attn_dtype = attention_mask.dtype
            extend_mask = torch.ones(len(token), 1, dtype=attn_dtype).to(device)
            attention_mask = torch.cat((attention_mask, extend_mask), dim=1)

            # ending detection
            num_ended = 0
            for j in range(len(batch)):
                if ending[j] == -1 and token[j] == tokenizer.eos_token_id:
                    ending[j] = i
                if ending[j] != -1:
                    num_ended += 1
            if num_ended == len(batch):
                break

        # collect results
        for i in range(len(output_ids)):
            if ending[i] != -1:
                output_ = output_ids[i][: l_input_ids + ending[i]]
                is_finished = True
            else:
                output_ = output_ids[i]
                is_finished = False
            sentence = tokenizer.decode(output_, skip_special_tokens=True)
            output = sentence[len(batch[i]) :]

            num_input_tokens = len(input_ids[i])
            num_output_tokens = len(tokenizer(output).input_ids)
            num_total_tokens = num_input_tokens + num_output_tokens
            length = output_ids[i].shape[0] - l_input_ids + 1

            # return
            result = dict(
                input=batch[i],
                output=output,
                sentence=sentence,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                num_total_tokens=num_total_tokens,
                is_finished=is_finished,
                length=length,
            )
            results.append(result)
    
    return results
    
def base_inference(input_texts, model_name='gpt2', max_length=50, batch_size=2):
    # 确保CUDA可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和分词器
    model, tokenizer = load_model(model_name, device, num_gpus = 1)

    model.to(device)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.unk_token

    all_results = []

    input_texts = [input_texts[i:i+batch_size] for i in range(0, len(input_texts), batch_size)]

    for batch in input_texts:
    # 将输入文本编码为批处理
        # TODO
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        outputs = model.generate(input_ids = input_ids, attention_mask = attention_mask, max_length = max_length)

        results = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        all_results.append(results)

    return all_results

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# 示例输入
with open('extracted_human_conversations.json', 'r') as file:
    data = json.load(file)  # 更多文本

input_texts = data[:512]
input_texts = ["Hi, I am a robot", "Hi, I am a human"]

model_name = "../vicuna-1.5-7b"
batch_size = 16
iter_max_length = 20
max_length = 256

print("=========================== base inference =====================================")

base_start_time = time.time()
output_texts = base_inference(input_texts, model_name = model_name, max_length = max_length, batch_size = batch_size)
base_end_time = time.time()
base_execution_time = base_end_time - base_start_time

for text in output_texts:
    for item in text:
        print(item)

print("=========================== interation level base inference =====================================")

interation_level_base_start_time = time.time()
output_texts = interation_level_base_inference(input_texts, model_name = model_name, max_length = max_length, batch_size = batch_size)
interation_level_base_end_time = time.time()
interation_level_base_execution_time = interation_level_base_end_time - interation_level_base_start_time

for text in output_texts:
    print(text['sentence'])

print("=========================== two stage inference =====================================")

# 执行两阶段推理
my_start_time = time.time()
output_texts = two_stage_inference(input_texts, model_name = model_name, iter_max_length= iter_max_length, batch_size = batch_size)
my_end_time = time.time()
my_execution_time = my_end_time - my_start_time

# 打印输出文本
for text in output_texts:
    print(text)

print(f"my time : {my_execution_time} sec, base time : {base_execution_time} sec, interation level base execution time : {interation_level_base_execution_time} sec")