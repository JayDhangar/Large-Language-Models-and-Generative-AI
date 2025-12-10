import os
import sys

try:
    from transformers import pipeline, set_seed
    have_transformers = True
except Exception:
    have_transformers = False

try:
    from diffusers import StableDiffusionPipeline
    import torch
    have_diffusers = True
except Exception:
    have_diffusers = False

try:
    import google.generativeai as genai
    have_genai = True
except Exception:
    have_genai = False

google_api_key = os.environ.get("google_api_key")

LOCAL_TEXT_MODEL = os.environ.get("LOCAL_TEXT_MODEL", "distilgpt2")
LOCAL_IMAGE_MODEL = os.environ.get("LOCAL_IMAGE_MODEL", "runwayml/stable-diffusion-v1-5")


def gen_text_local(prompt, max_new_tokens=40, temperature=0.7):
    if not have_transformers:
        raise RuntimeError("Local text generation unavailable: install 'transformers'.")
    pipe = pipeline("text-generation", model=LOCAL_TEXT_MODEL, device=-1)  # CPU
    set_seed(42)
    out = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.92,
        top_k=50,
        pad_token_id=50256,
        num_return_sequences=1,
    )
    generated = out[0]["generated_text"]
    reply = generated[len(prompt):].strip()
    if "." in reply:
        reply = reply.split(".")[0].strip()
    return reply or generated


def gen_image_local(prompt, output_path="local_image.png", steps=20, guidance_scale=7.5):
    if not have_diffusers:
        raise RuntimeError("Local image generation unavailable: install 'diffusers' and 'torch'.")
    print("Loading Stable Diffusion model locally (may take time & memory).")
    pipe = StableDiffusionPipeline.from_pretrained(LOCAL_IMAGE_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance_scale).images[0]
    image.save(output_path)
    return output_path


def gen_text_gemini(prompt, model="models/text-bison-001", max_output_tokens=128):
    if not have_genai:
        raise RuntimeError(
            "Gemini client not installed. Install 'google-generativeai' (note: use Python 3.11) or set up GOOGLE_API_KEY."
        )

    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
    print("Calling google.generativeai client...")
    resp = genai.generate_text(model=model, prompt=prompt, max_output_tokens=max_output_tokens)
    return getattr(resp, "text", str(resp))


def main():
    print("Multi-tool generator — choose what to run.")
    print("Available tools detection:")
    print(f" - transformers installed: {have_transformers}")
    print(f" - diffusers installed: {have_diffusers}")
    print(f" - google.generativeai client installed: {have_genai}")
    print(f" - GOOGLE_API_KEY present: {bool(google_api_key)}")
    print()

    prompt_text = input("Enter a text prompt (for short reply): ").strip()
    if not prompt_text:
        print("No text prompt provided. Exiting.")
        sys.exit(1)

    prompt_image = input("Enter an image prompt (describe image): ").strip()
    if not prompt_image:
        print("No image prompt provided. Exiting.")
        sys.exit(1)

    print("\nChoose combination:")
    print("1) local text  + local image")
    print("2) Gemini text + local image")
    print("3) test available tools (auto choose best)")
    choice = input("Enter 1-3: ").strip()

    try:
        if choice == "1":
            # ensure both available
            if not have_transformers:
                raise RuntimeError("Option 1 requires 'transformers' installed.")
            if not have_diffusers:
                raise RuntimeError("Option 1 requires 'diffusers' and 'torch' installed.")
            print("Generating local text...")
            txt = gen_text_local(prompt_text)
            print("TEXT (local):", txt)
            print("Generating local image... (may be slow on CPU)")
            imgpath = gen_image_local(prompt_image)
            print("IMAGE saved to:", imgpath)

        elif choice == "2":
            if not have_genai:
                raise RuntimeError("Option 2 requires 'google-generativeai' client installed.")
            if not have_diffusers:
                raise RuntimeError("Option 2 requires 'diffusers' and 'torch' installed for local image generation.")
            print("Generating Gemini text...")
            txt = gen_text_gemini(prompt_text)
            print("TEXT (Gemini):", txt)
            print("Generating local image... (may be slow on CPU)")
            imgpath = gen_image_local(prompt_image)
            print("IMAGE saved to:", imgpath)

        elif choice == "3":
            # auto selection:
            # Text: prefer local transformers, else Gemini
            if have_transformers:
                print("Generating text using local transformers...")
                txt = gen_text_local(prompt_text)
                print("TEXT (local):", txt)
            elif have_genai:
                print("Generating text using Gemini...")
                txt = gen_text_gemini(prompt_text)
                print("TEXT (Gemini):", txt)
            else:
                print("No text generation tool available (install 'transformers' or 'google-generativeai').")

            # Image: prefer local diffusers if available
            if have_diffusers:
                print("Generating image locally...")
                imgpath = gen_image_local(prompt_image)
                print("IMAGE saved to:", imgpath)
            else:
                print("No local image generator available (install 'diffusers' and 'torch').")
        else:
            print("Unknown choice. Please restart and select 1, 2 or 3.")
    except Exception as e:
        print("Error:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
