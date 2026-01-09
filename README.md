# Multimodal RAG with Jina-v4 & Qwen2.5-VL

這個專案實作了一個完整的**多模態檢索增強生成 (Multimodal RAG)** 流程。它可以同時處理文字 (.txt) 與圖片 (.png, .jpg...) 資料，將其轉化為向量存儲，並在查詢時透過語義搜尋找到最相關的圖文內容，最後交由 **Qwen2.5-VL-7B** 進行綜合分析。

## 🌟 核心功能

- **多模態向量化**：使用 `jinaai/jina-embeddings-v4` 同時對文字與圖片進行 Embedding。
- **增量更新**：自動掃描資料夾，僅針對「新加入」的檔案進行向量運算，並儲存至 `.npy` 與 `.json`。
- **混合檢索**：支援以文搜文、以文搜圖。
- **視覺理解生成**：整合 `Qwen/Qwen2.5-VL-7B-Instruct`，能同時理解檢索到的圖片與文字上下文。

## 📂 檔案結構

- `jina_v4_embedding_to_npy.py`: 資料預處理與索引建立腳本。
- `jina_v4_test.py`: 檢索與 LLM 推理的主程式。
- `docs/`: 存放文字檔案 (.txt)。
- `docs/images/`: 存放圖片檔案。
- `dataset_embeddings.npy`: 儲存向量資料。
- `dataset_texts.json`: 儲存對應的元數據 (Metadata)。

---

## 🚀 快速開始

### 1. 安裝環境
確保你的設備擁有足夠的顯存 (建議 16GB VRAM 以上以執行 Qwen2.5-VL-7B FP16)。並且建議使用虛擬環境。

```bash
pip install -r requirements.txt

```

### 2. embedding 資料
將你想要embedding的資料放入 **./docs** 資料夾中

```bash
python .\jina_v4_embedding_to_npy.py

```

### 3. 測試效果

```bash
python .\jina_v4_test.py

```
### 運行截圖
![Demo Result](assets/demo_result.png)