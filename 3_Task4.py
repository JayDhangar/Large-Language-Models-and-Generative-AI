import os
from flask import Flask, jsonify, request
from transformers import pipeline, set_seed

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt2")
gen = pipeline("text-generation", model=MODEL_NAME, device=-1)
set_seed(60)

print("Model loaded ,API is ready.")
app = Flask(__name__)

PROMPT_TEMPLATE = ("Write a short paragraph about AI for {audience} in a {tone} tone. "
                   "Start with: '{lead_sentence}'. Make it ~{length} words and include one practical example.")
PROMPT = "AI is the future because"

def prompt(params):
    variables = {
        "audience": params.get("audience", "general public"),
        "tone": params.get("tone", "friendly"),
        "lead_sentence": params.get("lead_sentence", "AI makes everyday life easier"),
        "length": params.get("length", 50)
    }
    try:
        return PROMPT_TEMPLATE.format(**variables)
    except KeyError:
        return PROMPT

@app.route("/generate", methods=["GET"])
def generate():

    explicit_prompt = request.args.get("prompt")
    base_prompt = explicit_prompt if explicit_prompt else prompt(request.args)

    # how many variations
    try:
        n = max(1, int(request.args.get("n", 3)))
    except ValueError:
        n = 3

    raw = request.args.get("modifiers")
    if raw:
        mods = [m.strip() for m in raw.split("|") if m.strip()]
        if not mods:
            mods = None
    else:
        mods = None

    default_mods = [
        "Keep it concise and easy to read.",
        "Make it emotional and persuasive.",
        "Use a short practical example and a data point."
    ]
    modifiers_list = mods if mods else default_mods

    max_new_tokens = int(request.args.get("max_new_tokens", 60))
    do_sample = request.args.get("do_sample", "true").lower() == "true"
    temperature = float(request.args.get("temperature", 0.5))
    top_k = int(request.args.get("top_k", 50))
    top_p = float(request.args.get("top_p", 0.95))

    results = []
    for i in range(n):
        modifier = modifiers_list[i % len(modifiers_list)]
        var_prompt = f"{base_prompt.strip()} {modifier}"
        out = gen(
            var_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1,
            pad_token_id=50256,
            return_full_text=False
        )

        generated_text = ""
        if isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], dict):
            # safe get
            generated_text = out[0].get("generated_text", "") or out[0].get("text", "")
        else:
            generated_text = str(out)

        results.append({
            "variation_index": i + 1,
            "modifier": modifier,
            "prompt_used": var_prompt,
            "generated": generated_text.strip()
        })

    return jsonify({"base_prompt": base_prompt, "variations": results})


DEFAULT_VARIATIONS = [
    {"audience": "high school students", "tone": "energetic", "lead_sentence": "AI helps us learn faster", "length": 40},
    {"audience": "business executives", "tone": "professional", "lead_sentence": "AI drives measurable ROI", "length": 50},
    {"audience": "general public", "tone": "friendly", "lead_sentence": "AI makes everyday life easier", "length": 45},
    {"audience": "software developers", "tone": "technical", "lead_sentence": "AI optimizes workflows", "length": 60},
]

if __name__ == "__main__":
    app.run(debug=True, port=7000)
