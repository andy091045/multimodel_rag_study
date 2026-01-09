import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image

# ====== 設定 ======
DOCS_FOLDER = "./docs"
IMAGE_FOLDER = os.path.join(DOCS_FOLDER, "images")

TEXTS_JSON = "dataset_texts.json"
EMB_NPY = "dataset_embeddings.npy"

model_name = "jinaai/jina-embeddings-v4"
print(f"正在載入模型: {model_name}...")
model = SentenceTransformer(model_name, trust_remote_code=True)

# 確保資料夾存在
os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# ====== 1) 讀取已存在的 records + embeddings ======
existing_records = []
existing_ids = set()
existing_embeddings = None

if os.path.exists(TEXTS_JSON):
    with open(TEXTS_JSON, "r", encoding="utf-8") as f:
        existing_records = json.load(f)
        existing_ids = {r["id"] for r in existing_records}

print(f"已存在 records: {len(existing_records)} 筆")

if os.path.exists(EMB_NPY):
    existing_embeddings = np.load(EMB_NPY)
    print(f"已存在 embeddings shape: {existing_embeddings.shape}")
else:
    print("尚未找到 dataset_embeddings.npy，將建立新檔。")

if existing_embeddings is not None and len(existing_records) != existing_embeddings.shape[0]:
    raise ValueError(
        f"❌ 資料不一致：records({len(existing_records)}) != embeddings({existing_embeddings.shape[0]})"
    )

# ====== 2) 掃描 docs（text + image）=====
new_texts = []
new_images = []

new_records = []

print(f"開始掃描文字資料夾: {DOCS_FOLDER}")

# ---- TXT ----
for filename in os.listdir(DOCS_FOLDER):
    if not filename.lower().endswith(".txt"):
        continue

    current_id = f"text_{filename}"
    if current_id in existing_ids:
        continue

    filepath = os.path.join(DOCS_FOLDER, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            continue

    new_texts.append(content)
    new_records.append({
        "id": current_id,
        "type": "text",
        "source": filename,
        "text": content
    })

# ---- IMAGE ----
print(f"開始掃描圖片資料夾: {IMAGE_FOLDER}")

for filename in os.listdir(IMAGE_FOLDER):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        continue

    current_id = f"image_{filename}"
    if current_id in existing_ids:
        continue

    filepath = os.path.join(IMAGE_FOLDER, filename)
    image = Image.open(filepath).convert("RGB")

    new_images.append(image)
    new_records.append({
        "id": current_id,
        "type": "image",
        "source": filename,
        "path": filepath
    })

print(f"新 text: {len(new_texts)} | 新 image: {len(new_images)}")

# ====== 3) 產生 embeddings（只處理新的）=====
new_embeddings_list = []

if new_texts:
    print("正在生成 Text Embeddings...")
    text_embeddings = model.encode(
        new_texts,
        task="retrieval"
    )
    new_embeddings_list.append(text_embeddings)

if new_images:
    print("正在生成 Image Embeddings...")
    image_embeddings = model.encode(
        new_images,
        task="retrieval"
    )
    new_embeddings_list.append(image_embeddings)

if not new_embeddings_list:
    print("沒有新資料，全部都 embed 過了。")
    exit(0)

new_embeddings = np.vstack(new_embeddings_list)

# ====== 4) 合併 + 存檔 ======
if existing_embeddings is None:
    merged_embeddings = new_embeddings
else:
    if existing_embeddings.shape[1] != new_embeddings.shape[1]:
        raise ValueError("❌ embedding 維度不一致，請重建資料庫")
    merged_embeddings = np.vstack([existing_embeddings, new_embeddings])

merged_records = existing_records + new_records

np.save(EMB_NPY, merged_embeddings)
with open(TEXTS_JSON, "w", encoding="utf-8") as f:
    json.dump(merged_records, f, ensure_ascii=False, indent=2)

print("✅ 更新完成")
print(f"總 records: {len(merged_records)}")
print(f"embeddings shape: {merged_embeddings.shape}")
