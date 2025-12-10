#I have created my own API and using a prompt generated text in it.
import os
from flask import Flask, jsonify
from transformers import pipeline, set_seed

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt2")
PROMPT = "AI is the future because"
gen = pipeline("text-generation", model=MODEL_NAME, device=-1)
set_seed(50)

print("Model loaded ,API is ready.")

app = Flask(__name__)

@app.route("/generate", methods=["GET"])
def generate():
    out = gen(
        "AI is the future because",
        max_new_tokens=60,
        do_sample=True,
        temperature=0.5,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        pad_token_id=50256
    )
    return jsonify({"generated":out[0].get("generated_text", "")})

if __name__=="__main__":
    app.run(debug=True,port=6000)
