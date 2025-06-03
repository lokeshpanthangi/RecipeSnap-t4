import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

class ImageCaptioningModel:
    def __init__(self, model_name="nlpconnect/vit-gpt2-image-captioning"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model.to(self.device)
        
        # Set generation parameters
        self.max_length = 16
        self.num_beams = 4
        self.gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams}
    
    def generate_caption(self, image_path):
        """
        Generate a caption for an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: The generated caption
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Process image
            pixel_values = self.feature_extractor(images=[image], return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate caption
            output_ids = self.model.generate(pixel_values, **self.gen_kwargs)
            
            # Decode caption
            caption = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            return caption
        except Exception as e:
            print(f"Error in generating caption: {e}")
            return "Failed to generate caption for this image." 