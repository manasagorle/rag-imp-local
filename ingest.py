# ingest.py
from sentence_transformers import SentenceTransformer
import os, glob
import faiss
import numpy as np
from pathlib import Path
import json
import PyPDF2
import re

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast
EMB_DIM = 384

IMPORTANT_KEYS = ["body", "content", "text", "description"]
def extract_text_from_pdf(path):
    txt = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            txt.append(p.extract_text() or "")
    return "\n".join(txt)


def extract_text_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    def flatten_json(obj):
        """Recursively extract all text-like content from JSON."""
        texts = []

        if isinstance(obj, dict):
            for value in obj.values():
                texts.extend(flatten_json(value))

        elif isinstance(obj, list):
            for item in obj:
                texts.extend(flatten_json(item))

        elif isinstance(obj, (str, int, float)):
            texts.append(str(obj))

        return texts

    
    raw_texts = flatten_json(data)

    # --- 3. Join & clean for NLP ---
    text = " ".join(raw_texts)

    # Remove extra spaces, line breaks, punctuation spacing issues
    text = re.sub(r"\s+", " ", text).strip()

    return text
    

def extract_text_from_txt(path):
    try:
        return open(path, encoding="utf-8").read().strip()
    except Exception as e:
        print(f"[TXT ERROR] Could not read {path}: {e}")
        return ""

        
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def ingest(data_dir="data", index_path="index.faiss", meta_path="index_meta.json"):
    model = SentenceTransformer(MODEL_NAME)
    all_texts = []
    metas = []
    
    for filepath in glob.glob(os.path.join(os.getcwd(), os.pardir, data_dir, "*")):
        ext = Path(filepath).suffix.lower()

        # choose extractor
        if ext == ".pdf":
            text = extract_text_from_pdf(filepath)
        elif ext == ".json":
            text = extract_text_from_json(filepath)
        elif ext == ".txt":
            text = open(filepath, encoding="utf-8").read()
        else:
            print(f"Skipping unsupported file type: {filepath}")
            continue
            
        chunks = chunk_text(text)

        for i, c in enumerate(chunks):
            metas.append({"source": filepath, "chunk_id": i, "text": c})
            all_texts.append(c)

    # embedding
    embeddings = model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)

    # FAISS
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)

    # metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    print(f"Index created: {index_path}")
    print(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    ingest()
