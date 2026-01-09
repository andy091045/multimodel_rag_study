import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
import json
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

model_name = "jinaai/jina-embeddings-v4"
print(f"正在載入embed模型: {model_name}...")
embed_model = SentenceTransformer(model_name, trust_remote_code=True)

def query(texts):
    return embed_model.encode(texts, task="retrieval")   # numpy array

# ✅ 正確載入 npy
dataset_np = np.load("dataset_embeddings.npy")
dataset_embeddings = torch.from_numpy(dataset_np).float()

# 載入 texts
with open("dataset_texts.json", "r", encoding="utf-8") as f:
    texts = json.load(f)

question = ["幫我分析新新小舖有限公司的財報"]
output = query(question)

query_embeddings = torch.from_numpy(output).float()

hits = semantic_search(
    query_embeddings,
    dataset_embeddings,
    top_k=3 #篩選幾個最相關的資料
)

# print([texts[hits[0][i]["corpus_id"]] for i in range(len(hits[0]))])

# for rank, h in enumerate(hits[0], start=1):
#     corpus_id = int(h["corpus_id"]) # 強制轉為整數確保索引正確
#     score = h["score"]
    
#     # 取得原始文字內容
#     full_text = texts[corpus_id]
    
#     # 對字串內容進行切片摘要
#     summary = full_text[:120] if isinstance(full_text, str) else str(full_text)[:120]
    
#     print(f"Top {rank} | score={score:.4f} | text={summary}...")

contexts_text = []
contexts_images = []

# 準備餵給llm的相關資料
for h in hits[0]:
    r = texts[int(h["corpus_id"])]
    
    if r["type"] == "text":
        contexts_text.append(r["text"])
    elif r["type"] == "image":
        contexts_images.append(r["path"])

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
vl_model = AutoModelForVision2Seq.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    dtype=torch.float16,
    device_map="auto"
)

images = [Image.open(p).convert("RGB") for p in contexts_images]

# 1. 構建對話內容
content = []

# 加入圖片訊息
for p in contexts_images:
    content.append({"type": "image", "image": p})

# 加入文字訊息
text_info = chr(10).join(contexts_text)
full_prompt = f"Question: {question[0]}\n\nRelevant text info:\n{text_info}\n\nPlease analyze images and text together."
content.append({"type": "text", "text": full_prompt})

messages = [
    {
        "role": "user",
        "content": content,
    }
]

# 2. 使用 apply_chat_template (這是關鍵！)
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# 3. 處理輸入
# 注意：這裡的 images 是 PIL Image 列表
inputs = processor(
    text=[text],
    images=images if images else None,
    padding=True,
    return_tensors="pt",
).to(vl_model.device)

# 4. 生成結果
outputs = vl_model.generate(**inputs, max_new_tokens=256)

# 5. 解碼（跳過 Prompt 部分）
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
]
answer = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(answer)