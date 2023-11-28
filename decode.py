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

    generated_texts = []
    updated_kv_caches = []
    streams = [Stream(device=device) for _ in range(2)]
    generated_outputs = []

    while ordered_sentences:
        # 选择有序数组中的第一个批次
        current_batch = [ordered_sentences[i][1] for i in range(min(batch_size, len(ordered_sentences)))]
        current_indices = [ordered_sentences[i][2] for i in range(min(batch_size, len(ordered_sentences)))]

        # 将当前批次的文本编码为批处理
        input_ids = [tokenizer.encode(text, return_tensors='pt') for text in current_batch]
        input_ids = torch.cat(input_ids, dim=0).to(device)

        # 执行推理
        with torch.cuda.stream(streams[0]):
            model_output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id,
                                          use_cache=True, return_dict_in_generate=True, past_key_values=[kv_caches[i] for i in current_indices])
            generated_outputs = model_output['sequences']
            updated_kv_caches = [model_output['past_key_values'][i] for i in current_indices]

        # 在第二个流中更新有序数组
        with torch.cuda.stream(streams[1]):
            for i, output in enumerate(generated_outputs):
                text = tokenizer.decode(output, skip_special_tokens=True)
                if len(text) >= max_length or text.endswith('.'):
                    # 如果句子推理完成，从数组中移除
                    generated_texts[current_indices[i]] = text
                else:
                    # 否则更新句子长度
                    ordered_sentences.append((len(text), text, current_indices[i]))
                    kv_caches[current_indices[i]] = updated_kv_caches[i]
            
            # 移除已处理的批次
            ordered_sentences = ordered_sentences[batch_size:]
            # 重新排序
            ordered_sentences.sort(key=lambda x: x[0])

        streams[1].synchronize()

    return [generated_texts[i] for i in range(len(input_texts))], kv_caches

# 示例输入和KV缓存
input_texts = ["The future of AI is", "In a distant galaxy", ...] # 更多文本
previous_kv_caches = ... # 之前保存的KV缓存

# 继续基于之前的KV缓存进行文本生成
continued_texts, new_kv_caches = continue_text_with_kv_cache(input_texts, previous_kv_caches)

# 打印继续生成的文本
for text in continued_texts:
    print(text)
