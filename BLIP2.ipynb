# instructblip_colab.py

"""
InstructBLIP Image Captioning Script

- Model: Salesforce/instructblip-vicuna-7b
- Dependencies: transformers, accelerate, bitsandbytes, peft, huggingface_hub, torch, Pillow, pandas

Run this script with GPU support (CUDA) for best performance.
"""

import os
import argparse
from PIL import Image
from datetime import datetime
import torch
import pandas as pd
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration


def load_model():
    print("Loading InstructBLIP processor and model...")
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
    model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-vicuna-7b",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return processor, model


def process_images(image_paths, processor, model):
    results = []
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((384, 384))
        except Exception as e:
            print(f"Failed to open image {image_name}: {e}")
            continue

        creation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        prompt = "You are a helpful AI assistant. Describe this image in detail."

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100)
            description = processor.batch_decode(output, skip_special_tokens=True)[0]

        results.append({
            "Filename": image_name,
            "Capture Time": creation_time,
            "Description": description
        })

    return results


def main(image_dir, output_csv):
    print("Checking device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    processor, model = load_model()

    # Load images
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_paths:
        print("No valid images found in the specified directory.")
        return

    print(f"{len(image_paths)} image(s) found.")

    # Process and save results
    results = process_images(image_paths, processor, model)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InstructBLIP Image Description Generator")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_csv", type=str, default="instructblip_results.csv", help="Output CSV file name")
    args = parser.parse_args()

    main(args.image_dir, args.output_csv)
