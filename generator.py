# generator.py (transformers backend)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MODEL = "facebook/opt-350m"  # example small model; pick one you can run locally
#MODEL = "transformers/BertLarge-CrossEntropy"
#MODEL =  'transformers/BertLarge-CrossEntropy'
# MODEL = 'google/gemma-3-270m-it'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

def answer_from_context(question, contexts):
    prompt = "You are a helpful assistant. Use the following context to answer the question.\n\n"
    for i,c in enumerate(contexts):
        prompt += f"Context {i+1}:\n{c}\n\n"
    prompt += f"Question: {question}\nAnswer:"
    out = gen(prompt, max_length=300, do_sample=False)[0]["generated_text"]
    return out
