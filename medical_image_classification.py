# -*- coding: utf-8 -*-
"""medical_image_classification.ipynb

Original file is located at
    https://colab.research.google.com/drive/1geP2_Tzf4vUJjTJH288Ddn4NGMNk92B0
"""

!pip install -q transformers torchvision pdf2image PyMuPDF beautifulsoup4 requests

import torch
from PIL import Image
import requests
from io import BytesIO
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import time
from transformers import CLIPProcessor, CLIPModel
from IPython.display import display

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def download_image(url):
    try:
        response = requests.get(url, timeout=15)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except:
        return None

def extract_images_from_url(web_url, max_images=20):
    print(f"🔗 Extracting images from: {web_url}")
    soup = BeautifulSoup(requests.get(web_url).text, 'html.parser')
    images = []
    for tag in soup.find_all("img"):
        src = tag.get("src")
        if src:
            img_url = urljoin(web_url, src)
            img = download_image(img_url)
            if img:
                images.append(img)
    return images

def extract_images_from_pdf(pdf_path):
    print(f"📄 Extracting images from PDF: {pdf_path}")
    return convert_from_path(pdf_path, dpi=200)

def classify_images(images, batch_size=5):
    results = []
    start_time = time.time()
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = processor(text=["medical image", "non-medical image"], images=batch, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).tolist()
        for p in probs:
            label = "medical" if p[0] > p[1] else "non-medical"
            results.append((label, p))
    total_time = time.time() - start_time
    return results, total_time

def run_pipeline(input_path_or_url):
    if input_path_or_url.endswith(".pdf"):
        images = extract_images_from_pdf(input_path_or_url)
    elif input_path_or_url.startswith("http"):
        images = extract_images_from_url(input_path_or_url)
    else:
        raise ValueError("Input must be a valid URL or a PDF file path.")

    print(f"🖼️ Total images extracted: {len(images)}")

    results, inference_time = classify_images(images)

    for idx, (label, probs) in enumerate(results):
        print(f"Image {idx+1}: {label.upper()} (Confidence: {probs})")
        display(images[idx])

    print(f"\n⏱️ Total inference time: {inference_time:.2f} seconds")

run_pipeline("https://www.healthline.com/")

# from google.colab import files
# uploaded = files.upload()
# pdf_path = list(uploaded.keys())[0]
# run_pipeline(pdf_path)