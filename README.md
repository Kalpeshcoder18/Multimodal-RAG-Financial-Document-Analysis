# Multimodal RAG – Financial Document Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/try)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG%20Framework-111827)](https://www.llamaindex.ai/)
[![OpenAI GPT-4V](https://img.shields.io/badge/OpenAI-GPT--4V-412991?logo=openai&logoColor=white)](https://openai.com/)
[![Deep Lake](https://img.shields.io/badge/Deep%20Lake-Vector%20DB-0F766E)](https://www.deeplake.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end notebook-based reference for building a multimodal RAG pipeline for financial documents. The workflow combines text, tables, and chart descriptions so the retriever can answer questions that would be missed by a text-only pipeline.

## 📌 Project Overview

This project demonstrates how to process investor presentations and financial PDFs into a retrieval system that captures both textual and visual signal.

```text
Financial PDF → Text/Table Extraction (Unstructured.io) → Chart Description (GPT-4V)
→ Embeddings → Deep Lake Vector DB → RAG Chatbot
```

The main idea is simple: financial documents are multimodal, so retrieval should be multimodal too. Tables and narrative text are useful, but chart descriptions often carry the key trend information needed for precise answers.

## 📈 Results

| Comparison | Outcome |
| --- | --- |
| Text-only RAG vs Multimodal RAG | Multimodal retrieval surfaces chart-driven evidence that text-only chunking often misses. |
| Deep Memory impact | ~25% retrieval improvement observed with Deep Memory in the shown experiment. |

![Trend chart demo](trend.avif)

The chart above illustrates the kind of visual evidence the chatbot can ground its answers in when chart descriptions are included in the vector store.

## 🧰 Tech Stack

| Component | Tool |
| --- | --- |
| Notebook environment | Jupyter Notebook |
| PDF parsing | Unstructured.io |
| Vision analysis | OpenAI GPT-4V |
| LLM orchestration | LlamaIndex |
| Vector database | Deep Lake |
| Embeddings | OpenAI embeddings |
| Question generation | GPT-3.5 Turbo |
| Deep retrieval optimization | Activeloop Deep Memory |

## ⚡ Quick Start

```bash
git clone https://github.com/Kalpeshcoder18/Multimodal-RAG-Financial-Document-Analysis
cd Multimodal-RAG-Financial-Document-Analysis

python -m venv .venv
.venv\Scripts\activate

pip install --upgrade pip
pip install jupyterlab notebook \
  unstructured[all-docs] llama-index deeplake openai \
  pdf2image pydantic tqdm langchain pillow

jupyter notebook "Tesla Investor Presentations.ipynb"
```

If you are on Windows, install Poppler and Tesseract before running the notebook cells that extract PDF text and pages.

## 📁 Project Structure

| File | Purpose |
| --- | --- |
| README.md | Project overview, setup, and technical notes. |
| Tesla Investor Presentations.ipynb | Main notebook with the multimodal RAG workflow. |
| categorized_elements.pkl | Cached text and table elements extracted from the source document. |
| graphs_description.pkl | Cached chart descriptions generated from the PDF pages. |
| trend.avif | Visual result used in the README demo section. |

## 🧠 Technical Walkthrough

The detailed implementation notes are kept below in collapsible sections so the README stays readable while preserving the original technical depth.

<details>
<summary><strong>Extracting Text and Tables</strong></summary>

The preprocessing pipeline starts with `unstructured.partition.pdf.partition_pdf`, which extracts structured elements from the PDF. Text chunks and tables are separated into a normalized `Element` model so each item can later be tagged with its source type.

```python
from unstructured.partition.pdf import partition_pdf

raw_pdf_elements = partition_pdf(
    filename="./TSLA-Q3-2023-Update-3.pdf",
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
)
```

```python
from pydantic import BaseModel
from typing import Any

class Element(BaseModel):
    type: str
    text: Any

categorized_elements = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        categorized_elements.append(Element(type="table", text=str(element)))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        categorized_elements.append(Element(type="text", text=str(element)))
```

This keeps the extracted information easy to trace back to the originating PDF content.

</details>

<details>
<summary><strong>Describing Charts and Graphs</strong></summary>

Financial PDFs often encode key signals in charts rather than in prose. To capture that signal, each page is converted to an image and sent to GPT-4V for chart detection and description. Pages without charts return an empty JSON payload so the pipeline can skip them cleanly.

```python
from pdf2image import convert_from_path

convertor = convert_from_path("./TSLA-Q3-2023-Update-3.pdf")
```

```python
payload = {
  "model": "gpt-4-vision-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "You are an assistant that find charts, graphs, or diagrams from an image and summarize their information."},
        {"type": "text", "text": "Return JSON only in the format {\"graphs\": [<chart_1>, <chart_2>]}"},
      ]
    }
  ],
  "max_tokens": 1000
}
```

The resulting graph descriptions are stored alongside the text and table chunks, giving the retriever access to both numerical and visual context.

</details>

<details>
<summary><strong>Storing Embeddings in Deep Lake</strong></summary>

The combined set of text, tables, and chart descriptions is converted into LlamaIndex documents and written to Deep Lake. This allows the notebook to use a single vector store for retrieval across all modalities.

```python
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index import Document, VectorStoreIndex

vector_store = DeepLakeVectorStore(
    dataset_path=dataset_path,
    runtime={"tensor_db": True},
    overwrite=False,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
documents = [Document(text=t.text, metadata={"category": t.type}) for t in categorized_elements]

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
```

This is the step that makes the repository practical: everything is stored in a retrieval layer that the chatbot can query directly.

</details>

<details>
<summary><strong>Deep Memory Training</strong></summary>

Activeloop Deep Memory is used to improve retrieval quality by training on synthetic questions generated from relevant chunks. The notebook reports a noticeable boost in recall, which is where the ~25% improvement claim comes from.

```python
job_id = db.vectorstore.deep_memory.train(
    queries=questions,
    relevance=relevances,
    embedding_function=embeddings.embed_documents,
)
```

The dataset must be queried with `deep_memory=True` during inference to benefit from the improved retrieval space.

</details>

<details>
<summary><strong>Chatbot In Action</strong></summary>

The chatbot is built on top of the vector store with `index.as_query_engine()`. Once the Deep Memory-enabled retriever is active, questions about delivery trends and report-level changes can be answered with references grounded in both text and chart descriptions.

```python
query_engine = index.as_query_engine(vector_store_kwargs={"deep_memory": True})
response = query_engine.query("What are the trends in vehicle deliveries?")
```

The multimodal version performs better on questions that depend on chart trends, while the text-only version can point to nearby but incorrect passages.

</details>

<details>
<summary><strong>Notes and Limitations</strong></summary>

This repository is a reference implementation, not a benchmark suite. It is designed to show how multimodal retrieval can improve answer quality on financial PDFs, especially when the relevant evidence lives in charts or slides rather than in surrounding prose.

The notebook workflow is intentionally inspectable and reproducible, making it a good starting point for extending the pipeline to other financial filings, presentations, or report archives.

</details>

## 👤 Footer

Built by Kalpesh Paliwal | B.E. copc @ Thapar University
