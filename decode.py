import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fastchat.serve.inference import load_model
from torch.cuda import Stream

def continue_text_with_kv_cache(input_texts, kv_caches, model_name='gpt2', max_length=50, batch_size=2):
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    device = "cuda"

    model, tokenizer = load_model(model_name = model_name, device = device, num_gpus = 1)

    model.to(device)
    
    ordered_sentences = [(len(tokenizer.encode(text)), text, i) for i, text in enumerate(input_texts)]
    ordered_sentences.sort(key=lambda x: x[0])

    generated_texts = [None] * len(input_texts)
    streams = [Stream(device=device) for _ in range(2)]

    batch_len = min(batch_size, len(ordered_sentences))
    is_first = True

    # 每个batch维护4个列表
    # current_kv_caches: kv cache on GPU
    # current_batch_indices: batch中的句子在原来存储列表中的位置
    # current_batch: text on CPU
    # current_input_ids: last token of text on GPU

    #initialize
    current_kv_caches = [kv_caches[i].to(device, non_blocking=True) for i in range(batch_len)] if ordered_sentences else []
    current_batch_indices = [ordered_sentences[i][2] for i in range(batch_len)]
    current_batch = [ordered_sentences[i][1] for i in range(batch_len)]
    current_input_ids = torch.cat([tokenizer.encode(text.split()[-1], return_tensors='pt') for text in current_batch], dim=0).to(device, non_blocking=True)

    while ordered_sentences:
        batch_len = min(batch_size, len(ordered_sentences))

        # 调整最大长度限制
        adjusted_max_length = max_length if len(ordered_sentences) >= batch_size else model.config.max_length

        with torch.cuda.stream(streams[0]):
            current_model_output = model.generate(current_input_ids, max_length=adjusted_max_length, pad_token_id=tokenizer.eos_token_id,
                                          use_cache=True, return_dict_in_generate=True, past_key_values=current_kv_caches)

        with torch.cuda.stream(streams[1]): 
            ordered_sentences = [ordered_sentences[i] for i in range(len(ordered_sentences)) if i not in current_batch_indices]
            
            if is_first == False:
                prev_generated_outputs = prev_model_output['sequences']
                prev_updated_kv_caches = prev_model_output['past_key_values']
                
                for i, idx in enumerate(prev_batch_indices):
                    output = prev_generated_outputs[i].to('cpu', non_blocking=True)
                    text = tokenizer.decode(output, skip_special_tokens=True)
                    generated_texts[idx] = text
                    if not text.endswith('.'): # maybe not endwith EOS
                        ordered_sentences.append((len(text), text, idx))
                        kv_caches[idx] = prev_updated_kv_caches[i].to('cpu', non_blocking=True)

                del prev_model_output
                del prev_generated_outputs
                del prev_updated_kv_caches
                
                ordered_sentences.sort(key=lambda x: x[0])
            else:
                is_first = False
            
            next_batch_indices = [ordered_sentences[i][2] for i in range(min(batch_size, len(ordered_sentences)))]
            next_batch = [ordered_sentences[i][1] for i in range(min(batch_size, len(ordered_sentences)))]
            next_input_ids = torch.cat([tokenizer.encode(text.split()[-1], return_tensors='pt') for text in next_batch], dim=0).to(device, non_blocking=True)
            next_kv_caches = [kv_caches[i].to(device, non_blocking=True) for i in next_batch_indices]
            
            prev_batch_indices = current_batch_indices
            current_batch_indices = next_batch_indices

        streams[1].synchronize()
        streams[0].synchronize()

        current_input_ids = next_input_ids
        current_kv_caches = next_kv_caches
        
        prev_model_output = current_model_output
        del current_model_output

    #post processing
    prev_generated_outputs = prev_model_output['sequences']
    prev_updated_kv_caches = prev_model_output['past_key_values']
                
    for i, idx in enumerate(prev_batch_indices):
        output = prev_generated_outputs[i].to('cpu', non_blocking=True)
        text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts[idx] = text

    del prev_model_output
    del prev_generated_outputs
    del prev_updated_kv_caches
    
    return [text for text in generated_texts if text is not None], kv_caches
