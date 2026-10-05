# Egyptian Civil Code — Arabic/English Legal RAG

A bilingual Arabic/English Retrieval-Augmented Generation (RAG) system for question answering over the Egyptian Civil Code.

## Project

- Corpus: Egyptian Civil Code (Law No. 131 of 1948)
- Languages: Arabic and English
- Embedding model: `BAAI/bge-m3`
- Reranker: `BAAI/bge-reranker-v2-m3`
- Vector database: Qdrant
- LLM providers: Ollama, Groq, Google Gemini
- API: FastAPI
- Experiment tracking: MLflow
- Data/version tracking: DVC
- Evaluation: Retrieval metrics and RAGAS

## Corpus

The corpus contains 1,149 legal articles extracted and validated from the bilingual Civil Code PDF.

The production chunking configuration is article-aware with:

- Maximum tokens: 512
- Overlap: 0
- Tokenizer: `BAAI/bge-m3`

The canonical corpus and chunk outputs are managed through DVC.

## Reproducibility

Initialize the environment and reproduce the corpus pipeline with:

```bash
dvc repro
```
Run the API
```bash
uvicorn main:app
```
```bash
Project Structure
data/
├── raw/
├── processed/
└── evaluation/

src/
├── evaluation/
├── llm/
├── mlflow/
├── retrieval/
├── routes/
├── schemas/
└── scripts/

dvc.yaml
dvc.lock
main.py
```
Notes
Secrets and local MLflow tracking files are intentionally excluded from Git.
EOF

Then stage the README update:

```bash
git add README.md .gitignore
