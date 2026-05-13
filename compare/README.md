# RAG Pipeline cho HotPotQA — LangChain

Dự án xây dựng hệ thống **Retrieval-Augmented Generation (RAG)** chuẩn sử dụng **LangChain** để trả lời câu hỏi multi-hop từ bộ dữ liệu **HotPotQA**.

## 📁 Cấu trúc dự án

```
RagLangChain/
├── config/
│   ├── __init__.py
│   └── settings.py              # Cấu hình tập trung (env vars)
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Tải & xử lý HotPotQA
│   ├── vectorstore.py           # Chroma vector store + retriever
│   ├── rag_chain.py             # RAG chain (2 cách xây dựng)
│   └── evaluation.py            # Đánh giá EM / F1 / ROUGE-L
├── data/                        # Dữ liệu & kết quả (auto-created)
├── vectorstore/                 # Chroma DB persist (auto-created)
├── main.py                      # Entry point chính
├── requirements.txt             # Dependencies
├── .env.example                 # Template cấu hình environment
├── .gitignore
└── README.md
```

## 🏗️ Kiến trúc RAG Pipeline

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  HotPotQA   │────▶│  Text Splitter   │────▶│  OpenAI         │
│  Dataset    │     │  (Recursive)     │     │  Embeddings     │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  User       │────▶│  Retriever       │◀───▶│  Chroma         │
│  Question   │     │  (Top-K)         │     │  Vector Store   │
└─────────────┘     └────────┬─────────┘     └─────────────────┘
                             │
                             ▼
                    ┌──────────────────┐     ┌─────────────────┐
                    │  Prompt Template │────▶│  ChatOpenAI     │
                    │  (System+Human) │     │  (GPT-4o-mini)  │
                    └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │  Answer         │
                                              └─────────────────┘
```

## 🔧 Cài đặt

### 1. Clone & cài dependencies

```bash
cd RagLangChain
pip install -r requirements.txt
```

### 2. Cấu hình API key

```bash
cp .env.example .env
# Sửa .env và điền OPENAI_API_KEY
```

### 3. (Tùy chọn) Tải NLTK data

```python
import nltk
nltk.download('punkt')
```

## 🚀 Sử dụng

### Chạy toàn bộ pipeline (index + evaluate + query)

```bash
python main.py
```

### Chỉ index dữ liệu

```bash
python main.py --mode index --sample-size 100
```

### Chỉ hỏi đáp (cần đã index trước)

```bash
python main.py --mode query
python main.py --mode query --chain-type lcel    # Dùng LCEL chain
```

### Chỉ đánh giá

```bash
python main.py --mode evaluate --eval-size 50
```

### Tùy chỉnh cấu hình qua biến môi trường

| Biến | Mô tả | Mặc định |
|------|--------|----------|
| `OPENAI_API_KEY` | API key OpenAI | (bắt buộc) |
| `HOTPOTQA_SUBSET` | Tập con: `fullwiki` / `distractor` | `fullwiki` |
| `HOTPOTQA_SPLIT` | Split: `train` / `validation` | `train` |
| `HOTPOTQA_SAMPLE_SIZE` | Số mẫu tải về | `500` |
| `EMBEDDING_MODEL` | Model embedding | `text-embedding-3-small` |
| `CHUNK_SIZE` | Kích thước chunk | `1000` |
| `CHUNK_OVERLAP` | Overlap giữa chunks | `200` |
| `RETRIEVER_TOP_K` | Số documents truy xuất | `5` |
| `LLM_MODEL` | Model LLM | `gpt-4o-mini` |
| `LLM_TEMPERATURE` | Temperature | `0` |
| `EVAL_SAMPLE_SIZE` | Số mẫu đánh giá | `50` |

## 📊 Metrics đánh giá

| Metric | Mô tả |
|--------|--------|
| **Exact Match (EM)** | Đáp án dự đoán khớp chính xác với ground truth (sau chuẩn hóa) |
| **F1 Score** | Token-level F1 giữa prediction và ground truth |
| **ROUGE-L** | Longest Common Subsequence F-measure |
| **Retrieval Precision** | Tỉ lệ supporting facts được retriever trả về |

## 🔗 Hai cách xây dựng RAG Chain

### Cách 1: `create_retrieval_chain` (Chuẩn LangChain)

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
result = rag_chain.invoke({"input": "câu hỏi"})
# result = {"input": ..., "context": [...], "answer": "..."}
```

### Cách 2: LCEL (LangChain Expression Language)

```python
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
answer = rag_chain.invoke("câu hỏi")
# answer = "..." (string trực tiếp)
```

## 📄 LangChain Components sử dụng

| Component | Module | Vai trò |
|-----------|--------|---------|
| `ChatOpenAI` | `langchain_openai` | LLM sinh câu trả lời |
| `OpenAIEmbeddings` | `langchain_openai` | Tạo embeddings |
| `Chroma` | `langchain_chroma` | Vector store lưu trữ |
| `RecursiveCharacterTextSplitter` | `langchain_text_splitters` | Chia nhỏ văn bản |
| `ChatPromptTemplate` | `langchain_core.prompts` | Mẫu prompt |
| `RunnablePassthrough` | `langchain_core.runnables` | LCEL composition |
| `StrOutputParser` | `langchain_core.output_parsers` | Parse output |
| `create_retrieval_chain` | `langchain.chains` | Tạo retrieval chain |
| `create_stuff_documents_chain` | `langchain.chains.combine_documents` | Kết hợp documents |

## 📚 Tham khảo

- [HotPotQA](https://hotpotqa.github.io/) — Multi-hop QA Dataset
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [LangChain API Reference](https://api.python.langchain.com/)
