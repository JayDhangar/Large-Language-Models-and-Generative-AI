from transformers import pipeline

gen = pipeline("text-generation",model="gpt2",device=0)
print(gen("AI is the future because")[0]["generated_text"])
