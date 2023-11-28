# two_stage_inference.py
from prefill import generate_text_with_kv_cache
from decode import continue_text_with_kv_cache

def two_stage_inference(input_texts, model_name='gpt2', max_length=50, batch_size=2):
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
        input_texts, 
        kv_caches, 
        model_name=model_name, 
        max_length=max_length, 
        batch_size=batch_size
    )

    return continued_texts

# 示例输入
input_texts = ["The future of AI is", "In a distant galaxy", ...]  # 更多文本

# 执行两阶段推理
output_texts = two_stage_inference(input_texts)

# 打印输出文本
for text in output_texts:
    print(text)
