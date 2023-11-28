import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.cuda import Stream

def continue_text_with_kv_cache(input_texts, kv_caches, model_name='gpt2', max_length=50, batch_size=2):
    # 确保CUDA可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和分词器
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # 在CPU端维护一个有序数组
    ordered_sentences = [(len(tokenizer.encode(text)), text, i) for i, text in enumerate(input_texts)]
    ordered_sentences.sort(key=lambda x: x[0])

    generated_texts = [None] * len(input_texts)
    streams = [Stream(device=device) for _ in range(2)]

    # 预先在GPU上加载第一个批次的KV缓存
    if ordered_sentences:
        first_batch_indices = [ordered_sentences[i][2] for i in range(min(batch_size, len(ordered_sentences)))]
        first_kv_caches = [kv_caches[i].to(device) for i in first_batch_indices]
    else:
        first_kv_caches = []

    while ordered_sentences:
        current_indices = [ordered_sentences[i][2] for i in range(min(batch_size, len(ordered_sentences)))]
        current_batch = [ordered_sentences[i][1] for i in range(min(batch_size, len(ordered_sentences)))]

        # 在CPU上准备下一个批次的KV缓存，如果有的话
        next_batch_indices = [ordered_sentences[i + batch_size][2] for i in range(min(batch_size, len(ordered_sentences) - batch_size))]
        next_kv_caches = [kv_caches[i].to(device) for i in next_batch_indices]

        # 将当前批次的文本编码为批处理并传输到GPU
        input_ids = torch.cat([tokenizer.encode(text, return_tensors='pt') for text in current_batch], dim=0).to(device)

        # 在第一个流中执行当前批次的推理
        with torch.cuda.stream(streams[0]):
            model_output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id,
                                          use_cache=True, return_dict_in_generate=True, past_key_values=first_kv_caches)

        # 在第二个流中处理第i-1个批次的结果，并准备第i+1个批次
        with torch.cuda.stream(streams[1]):
            # 如果不是第一个批次，处理上一个批次的结果
            if current_indices[0] != 0:
                for i, idx in enumerate(prev_batch_indices):
                    output = prev_generated_outputs[i].to('cpu')
                    text = tokenizer.decode(output, skip_special_tokens=True)
                    if len(text) >= max_length or text.endswith('.'):
                        generated_texts[idx] = text
                    else:
                        ordered_sentences.append((len(text), text, idx))
                        kv_caches[idx] = prev_updated_kv_caches[i].to('cpu')
            
            # 移除已处理的批次，并重新排序
            ordered_sentences = ordered_sentences[batch_size:]
            ordered_sentences.sort(key=lambda x: x[0])

            # 更新批次信息
            prev_batch_indices = current_indices
            prev_generated_outputs = model_output['sequences']
            prev_updated_kv_caches = model_output['past_key_values']

            # 准备下一个批次的KV缓存
            first_kv_caches = next_kv_caches

        streams[1].synchronize()

    return [text for text in generated_texts if text is not None], kv_caches

# 示例输入和KV缓存
input_texts = ["The future of AI is", "In a distant galaxy", ...] # 更多文本
previous_kv_caches = ... # 之前保存的KV缓存

# 继续基于之前的KV缓存进行文本生成
continued_texts, new_kv_caches = continue_text_with_kv_cache(input_texts, previous_kv_caches)

# 打印继续生成的文本
for text in continued_texts:
    print(text)
