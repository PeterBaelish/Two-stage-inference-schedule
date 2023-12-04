import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.cuda import Stream

def generate_text_with_kv_cache(input_texts, model_name='gpt2', max_length=50, batch_size=2):
    # 确保CUDA可用
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    # 加载模型和分词器
    model = GPT2LMHeadModel.from_pretrained('../gpt2').to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('../gpt2')

    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    pad_token = '<PAD>'
    if pad_token not in tokenizer.get_added_vocab():
        tokenizer.add_tokens([pad_token])
        tokenizer.pad_token = pad_token
    
    model.resize_token_embeddings(len(tokenizer))
    # 对输入文本按长度排序
    # input_texts = [str(text) for text in input_texts]
    input_texts.sort(key=lambda text: len(tokenizer.tokenize(text)))

    # 将批次分割
    input_ids = [input_texts[i:i+batch_size] for i in range(0, len(input_texts), batch_size)]

    input_id_batches = []

    for batch in input_ids:
        inputs = tokenizer(batch, padding=True, return_tensors='pt')
        # inputs = torch.cat(inputs, dim=0)
        input_id_batches.append(inputs)

    generated_texts = []
    generated_outputs = []
    kv_caches = []
    streams = [Stream(device=device) for _ in range(2)]

    # 在第二个流中预加载第一个批次到GPU
    with torch.cuda.stream(streams[1]):
        gpu_batch = {key: value.to(device, non_blocking=True) for key, value in input_id_batches[0].items()}

    for i in range(len(input_id_batches)):
        with torch.cuda.stream(streams[1]):
            # 将下一个批次的数据拷贝到GPU（如果存在）
            if i < len(input_id_batches) - 1:
                next_gpu_batch = {key: value.to(device, non_blocking=True) for key, value in input_id_batches[i+1].items()}

            # 将上一个批次的结果及其KV缓存拷贝回CPU（跳过第一个批次）
            if i > 0:
                prev_batch_output = generated_outputs[i - 1].to('cpu', non_blocking=True)
                prev_kv_cache = kv_caches[i - 1].to('cpu', non_blocking=True)

                generated_outputs[i - 1] = prev_batch_output
                kv_caches[i - 1] = prev_kv_cache

                # 清理GPU上的内存
                del gpu_batch
                torch.cuda.empty_cache()
            
            gpu_batch = next_gpu_batch if i < len(input_id_batches) - 1 else None

        # 在第一个流中执行当前批次的推理
        with torch.cuda.stream(streams[0]):
            # 启用past_key_values来保存KV缓存
            model_output = model.generate(input_ids=gpu_batch['input_ids'], attention_mask=gpu_batch['attention_mask'], max_length=max_length, pad_token_id=tokenizer.eos_token_id,
                                          use_cache=True, return_dict_in_generate=True)

            generated_outputs.append(model_output['sequences'])
            kv_caches.append(model_output['past_key_values'])

        # 等待当前批次推理完成
        streams[0].synchronize()

    # 处理最后一个批次的输出和KV缓存
    with torch.cuda.stream(streams[1]):
        last_batch_output = generated_outputs[-1].to('cpu', non_blocking=True)
        last_kv_cache = kv_caches[-1].to('cpu', non_blocking=True)

        generated_outputs[-1] = last_batch_output
        kv_caches[-1] = last_kv_cache

    # 在CPU上处理所有输出
    for output in generated_outputs:
        output = output.cpu()
        for j in range(output.shape[0]):
            text = tokenizer.decode(output[j], skip_special_tokens=True)
            generated_texts.append(text)

    return generated_texts, kv_caches
