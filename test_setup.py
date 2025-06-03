"""
Test script to verify that all the required libraries are correctly installed
and the environment is properly set up.
"""

def check_dependencies():
    """Check if all required dependencies are installed."""
    dependencies = [
        "torch", "torchvision", "transformers", "flask", 
        "PIL", "numpy", "dotenv"
    ]
    
    missing = []
    for dep in dependencies:
        try:
            if dep == "PIL":
                import PIL
                print(f"✅ {dep} (Pillow) v{PIL.__version__}")
            elif dep == "dotenv":
                import dotenv
                print(f"✅ {dep} v{dotenv.__version__}")
            else:
                module = __import__(dep)
                version = getattr(module, "__version__", "unknown")
                print(f"✅ {dep} v{version}")
                
                # Additional check for CUDA availability in PyTorch
                if dep == "torch":
                    cuda_available = module.cuda.is_available()
                    device = "GPU (CUDA)" if cuda_available else "CPU only"
                    print(f"   - Running on: {device}")
        except ImportError:
            missing.append(dep)
            print(f"❌ {dep} not found")
    
    return missing

def check_models_load():
    """Check if the models can be loaded."""
    print("\nTesting model initialization...")
    
    try:
        from models.image_captioning import ImageCaptioningModel
        model = ImageCaptioningModel()
        print("✅ Image captioning model initialized")
    except Exception as e:
        print(f"❌ Image captioning model failed: {e}")
    
    try:
        from models.object_detection import ObjectDetectionModel
        model = ObjectDetectionModel()
        print("✅ Object detection model initialized")
    except Exception as e:
        print(f"❌ Object detection model failed: {e}")
    
    try:
        from models.recipe_generation import RecipeGenerationModel
        model = RecipeGenerationModel()
        print("✅ Recipe generation model initialized")
        print("   - Note: Recipe model may use fallback mode if Mistral-7B cannot be loaded")
    except Exception as e:
        print(f"❌ Recipe generation model failed: {e}")

if __name__ == "__main__":
    print("RecipeSnap Setup Test")
    print("=====================")
    
    print("\nChecking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print("\n❌ Some dependencies are missing. Please install them with:")
        print(f"pip install {' '.join(missing)}")
    else:
        print("\n✅ All dependencies are installed")
        
        # Only try to check models if all dependencies are installed
        check_models_load()
        
        print("\nSetup test completed!")
        print("If all checks passed, you can run the application with: python app.py") 