#image gen using nano banana
from google import genai
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()
API_key_Gemeni=os.getenv("API_Image_gen")
client = genai.Client(api_key=API_key_Gemeni)

prompt = (
    "Create a picture of my cat eating a nanobanana in a "
)

prompt = "A boy playing football on a turf, digital art"
response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=[prompt],
)

for part in response.parts:
    if part.text is not None:
        print(part.text)
    elif part.inline_data is not None:
        image = part.as_image()
        image.save("gene_image.png")


