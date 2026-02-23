# retriever.py
import faiss, json, numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, index_path="index.faiss", meta_path="index_meta.json"):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metas = json.load(f)

    def get_relevant(self, query, k=5):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        D,I = self.index.search(q_emb, k)
        results = []
        for idx in I[0]:
            results.append(self.metas[idx])
        return results

if __name__ == "__main__":
    r = Retriever()
    print(r.get_relevant("What is RAG?", k=3))
