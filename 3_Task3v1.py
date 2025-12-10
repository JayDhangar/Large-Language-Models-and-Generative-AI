#I have used random API(from HuggingFace) to generate text
import requests

API_URL = "https://ecarbo-text-generator-gpt-neo.hf.space/api/predict/"

payload = {"data": ["Hello how are ",5.0]}
res = requests.post(API_URL,json=payload)
print(res.json())




