import os
from dotenv import load_dotenv
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
import google.generativeai as genai

load_dotenv()

API_key_Openai=os.getenv("API_Text_gen")
API_key_Gemeni=os.getenv("API_Image_gen")

text_gen=pipeline("text-generation",model="gpt2",token="API_key_Openai")

def generate_text(prompt):
    output=text_gen(prompt,temperature=0.7,top_p=0.95)
    return output[0]["generated_text"]

# def generate_image(prompt):

#     genai.configure(api_key=os.getenv("API_Image_gen"))
#     model = genai.GenerativeModel("gemini-2.5-flash")
#     response = model.generate_content(prompt)
#     image=response.candidates[0].content.parts[0].inline_data.data
#     with open("generated_image.png", "wb") as a:
#         a.write(image)


def generate_image(prompt):
    pipe=StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",torch_dtype=torch.float32)
    pipe.to("cpu")
    image=pipe(prompt).images[0]
    image.save("ai_gen_img.png")


print(generate_text("AI is good for future or not"))
generate_image("A boy playing football on a turf")

    


