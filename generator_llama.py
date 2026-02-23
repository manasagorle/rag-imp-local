# generator_llama.py
from llama_cpp import Llama

llm = Llama(model_path=r"D:\lstm\rag-local-demo\notebooks\llama-2-7b.Q4_K_M.gguf", n_ctx=8192)

def answer_from_context(question, contexts):
    prompt = "Answer concisely using the context below.\n\n"
    for i,c in enumerate(contexts):
        prompt += f"Context {i+1}:\n{c}\n\n"
    prompt += f"Question: {question}\nAnswer:"
    resp = llm(prompt, max_tokens=256, temperature=0.1)
    return resp["choices"][0]["text"]