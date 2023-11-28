import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.cuda import Stream

def continue_text_with_kv_cache(input_texts, kv_caches, model_name='gpt2', max_length=50, batch_size=2):
    # 确保CUDA可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和分词器
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # 将输入文本编码为批处理
    input_ids = [tokenizer.encode(text, return_tensors='pt') for text in input_texts]
    input_ids = torch.cat(input_ids, dim=0)
    
    # 将批次分割
    input_id_batches = input_ids.split(batch_size)

    generated_texts = []
    updated_kv_caches = []
    streams = [Stream(device=device) for _ in range(2)]
    generated_outputs = []

    for i in range(len(input_id_batches)):
        with torch.cuda.stream(streams[1]):
            # 将下一个批次的数据及其KV缓存拷贝到GPU（如果存在）
            if i < len(input_id_batches) - 1:
                next_gpu_batch = input_id_batches[i + 1].to(device, non_blocking=True)
                next_kv_cache = kv_caches[i + 1].to(device, non_blocking=True) if i + 1 < len(kv_caches) else None

            # 将上一个批次的结果及其KV缓存拷贝回CPU（跳过第一个批次）
            if i > 0:
                prev_batch_output = generated_outputs[i - 1].to('cpu', non_blocking=True)
                prev_kv_cache = updated_kv_caches[i - 1].to('cpu', non_blocking=True)

                generated_outputs[i - 1] = prev_batch_output
                updated_kv_caches[i - 1] = prev_kv_cache

                # 清理GPU上的内存
                del gpu_batch
                del past_kv_cache
                torch.cuda.empty_cache()

            gpu_batch = next_gpu_batch if i < len(input_id_batches) - 1 else None
            past_kv_cache = next_kv_cache if i < len(kv_caches) - 1 else None

        # 在第一个流中执行当前批次的推理
        with torch.cuda.stream(streams[0]):
            # 使用past_key_values继续推理
            model_output = model.generate(gpu_batch, max_length=max_length, pad_token_id=tokenizer.eos_token_id,
                                          use_cache=True, return_dict_in_generate=True, past_key_values=past_kv_cache)

            generated_outputs.append(model_output['sequences'])
            updated_kv_caches.append(model_output['past_key_values'])

        # 等待当前批次推理完成
        streams[0].synchronize()

    # 处理最后一个批次的输出和KV缓存
    with torch.cuda.stream(streams[1]):
        last_batch_output = generated_outputs[-1].to('cpu', non_blocking=True)
        last_kv_cache = updated_kv_caches[-1].to('cpu', non_blocking=True)

        generated_outputs[-1] = last_batch_output
        updated_kv_caches[-1] = last_kv_cache

    # 在CPU上处理所有输出
    for output in generated_outputs:
        output = output.cpu()
        for j in range(output.shape[0]):
            text = tokenizer.decode(output[j], skip_special_tokens=True)
            generated_texts.append(text)

    return generated_texts, updated_kv_caches

# 示例输入和KV缓存
input_texts = ["The future of AI is", "In a distant galaxy", ...] # 更多文本
previous_kv_caches = ... # 这里是之前保存的KV缓存

# 继续基于之前的KV缓存进行文本生成
continued_texts, new_kv_caches = continue_text_with_kv_cache(input_texts, previous_kv_caches)

# 打印继续生成的文本
for text in continued_texts:
    print(text)
