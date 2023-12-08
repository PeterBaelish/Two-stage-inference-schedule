import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fastchat.serve.inference import load_model
from torch.cuda import Stream

def generate_text_with_kv_cache(input_texts, model_name='gpt2', batch_size=2):
    # 确保CUDA可用
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda"
    
    # 加载模型和分词器
    model, tokenizer = load_model(model_name, device, num_gpus = 1)

    model.to(device)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"
    
    # 对输入文本按长度排序
    input_texts.sort(key=lambda text: len(tokenizer.tokenize(text)))

    # 将批次分割
    input_ids = [input_texts[i:i+batch_size] for i in range(0, len(input_texts), batch_size)]

    input_id_batches = []

    for batch in input_ids:
        inputs = tokenizer(batch, padding=True, return_tensors='pt')
        input_id_batches.append(inputs)

    generated_ids = []
    generated_attention_mask = []
    kv_caches = []
    streams = [Stream(device=device) for _ in range(2)]

    # 在第二个流中预加载第一个批次到GPU
    with torch.cuda.stream(streams[1]):
        gpu_batch = {key: value.to(device, non_blocking=True) for key, value in input_id_batches[0].items()}

    streams[1].synchronize()

    for i in range(len(input_id_batches)):
        with torch.cuda.stream(streams[1]):
            # 将下一个批次的数据拷贝到GPU（如果存在）
            next_gpu_batch = None

            if i < len(input_id_batches) - 1:
                next_gpu_batch = {key: value.to(device, non_blocking=True) for key, value in input_id_batches[i+1].items()}

            # 将上一个批次的结果及其KV缓存拷贝回CPU（跳过第一个批次）
            if i > 0:
                logits_cpu = prev_model_output.logits.to('cpu')
                past_key_values_cpu = [tuple(kv.to('cpu') for kv in layer_kv) for layer_kv in prev_model_output.past_key_values]
                del prev_model_output
                torch.cuda.empty_cache()

                kv_caches.append(past_key_values_cpu)

                last_token_logits = logits_cpu[:, -1]
                token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
                output_ids = torch.cat((input_id_batches[i-1]['input_ids'], token), dim=1)

                generated_ids.append(output_ids)

                attn_dtype = input_id_batches[i-1]['attention_mask'].dtype
                extend_mask = torch.ones(len(token), 1, dtype=attn_dtype)
                input_id_batches[i-1]['attention_mask'] = torch.cat((input_id_batches[i-1]['attention_mask'], extend_mask), dim=1)
                generated_attention_mask.append(input_id_batches[i-1]['attention_mask'])

        # 在第一个流中执行当前批次的推理
        with torch.cuda.stream(streams[0]):
            # 启用past_key_values来保存KV缓存
            current_model_output = model(input_ids=gpu_batch['input_ids'], attention_mask=gpu_batch['attention_mask'], use_cache=True)

        # 等待当前批次推理完成
        streams[0].synchronize()
        streams[1].synchronize()

        gpu_batch = next_gpu_batch if i < len(input_id_batches) - 1 else None
        if next_gpu_batch is not None:
            del next_gpu_batch

        prev_model_output = current_model_output
        del current_model_output

    # 处理最后一个批次的输出和KV缓存
    with torch.cuda.stream(streams[1]):
        logits_cpu = prev_model_output.logits.to('cpu')
        past_key_values_cpu = [tuple(kv.to('cpu') for kv in layer_kv) for layer_kv in prev_model_output.past_key_values]
        del prev_model_output
        torch.cuda.empty_cache()

        kv_caches.append(past_key_values_cpu)

        last_token_logits = logits_cpu[:, -1]
        token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
        output_ids = torch.cat((input_id_batches[len(input_id_batches) - 1]['input_ids'], token), dim=1)

        generated_ids.append(output_ids)

        attn_dtype = input_id_batches[len(input_id_batches) - 1]['attention_mask'].dtype
        extend_mask = torch.ones(len(token), 1, dtype=attn_dtype)
        input_id_batches[len(input_id_batches) - 1]['attention_mask'] = torch.cat((input_id_batches[len(input_id_batches) - 1]['attention_mask'], extend_mask), dim=1)
        generated_attention_mask.append(input_id_batches[len(input_id_batches) - 1]['attention_mask'])

    streams[1].synchronize()

    return generated_ids, kv_caches, generated_attention_mask
