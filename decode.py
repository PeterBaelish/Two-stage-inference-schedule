import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.cuda import Stream

def continue_text_with_kv_cache(input_texts, kv_caches, model_name='gpt2', max_length=50, batch_size=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    ordered_sentences = [(len(tokenizer.encode(text)), text, i) for i, text in enumerate(input_texts)]
    ordered_sentences.sort(key=lambda x: x[0])

    generated_texts = [None] * len(input_texts)
    streams = [Stream(device=device) for _ in range(2)]

    first_kv_caches = [kv_caches[i].to(device) for i in range(min(batch_size, len(ordered_sentences)))] if ordered_sentences else []

    while ordered_sentences:
        batch_len = min(batch_size, len(ordered_sentences))
        current_indices = [ordered_sentences[i][2] for i in range(batch_len)]
        current_batch = [ordered_sentences[i][1] for i in range(batch_len)]

        next_batch_indices = [ordered_sentences[i + batch_size][2] for i in range(min(batch_size, len(ordered_sentences) - batch_size))]
        next_kv_caches = [kv_caches[i].to(device) for i in next_batch_indices]

        input_ids = torch.cat([tokenizer.encode(text, return_tensors='pt') for text in current_batch], dim=0).to(device)

        # 调整最大长度限制
        adjusted_max_length = max_length if len(ordered_sentences) >= batch_size else None

        with torch.cuda.stream(streams[0]):
            model_output = model.generate(input_ids, max_length=adjusted_max_length, pad_token_id=tokenizer.eos_token_id,
                                          use_cache=True, return_dict_in_generate=True, past_key_values=first_kv_caches)

        with torch.cuda.stream(streams[1]):
            if current_indices[0] != 0:
                for i, idx in enumerate(prev_batch_indices):
                    output = prev_generated_outputs[i].to('cpu')
                    text = tokenizer.decode(output, skip_special_tokens=True)
                    generated_texts[idx] = text
                    if len(text) < max_length and not text.endswith('.'):
                        ordered_sentences.append((len(text), text, idx))
                        kv_caches[idx] = prev_updated_kv_caches[i].to('cpu')

            ordered_sentences = ordered_sentences[batch_len:]
            ordered_sentences.sort(key=lambda x: x[0])

            prev_batch_indices = current_indices
            prev_generated_outputs = model_output['sequences']
            prev_updated_kv_caches = model_output['past_key_values']

            first_kv_caches = next_kv_caches

        streams[1].synchronize()

    return [text for text in generated_texts if text is not None], kv_caches

# 示例输入和KV缓存
input_texts = ["The future of AI is", "In a distant galaxy", ...]
previous_kv_caches = ...

continued_texts, new_kv_caches = continue_text_with_kv_cache(input_texts, previous_kv_caches)

for text in continued_texts:
    print(text)
