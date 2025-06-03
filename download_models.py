"""
Download models script for RecipeSnap

This script downloads all required models in advance, which can be
helpful for offline use or to avoid delays when first starting the application.
"""

import os
import torch
from transformers import (
    VisionEncoderDecoderModel, 
    ViTImageProcessor, 
    AutoTokenizer,
    DetrImageProcessor, 
    DetrForObjectDetection,
    AutoModelForCausalLM
)

def download_models():
    """Download all models required by RecipeSnap."""
    print("RecipeSnap Model Downloader")
    print("===========================")
    
    models = {
        "Image Captioning": "nlpconnect/vit-gpt2-image-captioning",
        "Object Detection": "facebook/detr-resnet-50",
        "Recipe Generation": "mistralai/Mistral-7B-Instruct-v0.2"
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    for model_name, model_id in models.items():
        print(f"\nDownloading {model_name} model: {model_id}")
        
        try:
            if model_name == "Image Captioning":
                print("- Downloading model...")
                model = VisionEncoderDecoderModel.from_pretrained(model_id)
                
                print("- Downloading image processor...")
                feature_extractor = ViTImageProcessor.from_pretrained(model_id)
                
                print("- Downloading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                
            elif model_name == "Object Detection":
                print("- Downloading processor...")
                processor = DetrImageProcessor.from_pretrained(model_id)
                
                print("- Downloading model...")
                model = DetrForObjectDetection.from_pretrained(model_id)
                
            elif model_name == "Recipe Generation":
                print("- Downloading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                
                print("- Downloading model...")
                print("  Note: This is a large model (~13GB) and may take time.")
                print("  If you're running on CPU, this might be very slow.")
                
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        low_cpu_mem_usage=True
                    )
                except Exception as e:
                    print(f"  Warning: Failed to download the full model: {e}")
                    print("  The application will use fallback recipes instead.")
            
            print(f"✅ {model_name} model downloaded successfully!")
            
        except Exception as e:
            print(f"❌ Error downloading {model_name} model: {e}")
            print("  The application may use fallback features instead.")
    
    print("\nModel download completed!")
    print("You can now run the application with: python app.py")

if __name__ == "__main__":
    download_models() 