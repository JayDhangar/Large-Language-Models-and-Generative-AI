from transformers import pipeline

chat = pipeline("text-generation", model="gpt2")
print("AI Chatbot (type 'exit' to quit)")

while True:
    user = input("You: ").strip()
    if user.lower() == "exit":
        break

    raw = chat(user, max_length=40, num_return_sequences=1)[0]["generated_text"]
    reply = raw.replace(user, "").strip()

    if "." in reply:
        reply = reply.split(".")[0].strip()

    print("AI:", reply)

