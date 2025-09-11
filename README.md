   MEDICAL_IMAGE_CLASSIFICATION

PROJECT DESCRIPTION:
   This project focuses on Medical Image Classification using a pretrained Vision-Language Model (OpenAI’s CLIP: clip-vit-base-patch32). The goal is to classify images as either “medical” or “non-medical”, regardless of whether they are     sourced from a website URL or extracted from a PDF document.

WE BUILT PIPELINE THAT:
1.Data Input: Accepts images from
   -Web pages (via BeautifulSoup & requests), or
   -PDF files (using pdf2image & PyMuPDF).

2.Preprocessing: Images are cleaned and converted into a format suitable for the model.

3.Model Inference: Uses Hugging Face Transformers with CLIP (Contrastive Language-Image Pretraining) to compare images against text prompts (“medical image” vs. “non-medical image”).

4.Classification: Assigns each image a label and confidence score.

5.Evaluation: Tracks inference speed and classification results for scalability testing.

PLATFORMS AND TOOLS USED:
1.Python (Jupyter Notebook) for development.

2.PyTorch backend with GPU acceleration (if available).

3.Hugging Face Transformers for loading CLIP model.

4.Libraries:
  -torchvision, PIL for image handling
  -pdf2image, PyMuPDF for PDF extraction
  -BeautifulSoup4, requests for web scraping

OUTCOME:
Created an end-to-end image classification pipeline that automatically extracts images from multiple sources and categorizes them.
Designed to be scalable, fast, and model-agnostic, allowing future replacement with fine-tuned or domain-specific models.
