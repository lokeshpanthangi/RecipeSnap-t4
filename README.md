# RecipeSnap - AI Cooking Assistant

RecipeSnap is an AI-powered cooking assistant that recommends delicious recipes based on the ingredients in your fridge. Simply take a picture of your ingredients, and RecipeSnap will identify them and suggest recipes you can make!

## Features

- Upload or take photos of ingredients
- AI-powered ingredient detection
- Recipe recommendations based on available ingredients
- User-friendly interface

## Models Used

- **Image Captioning**: nlpconnect/vit-gpt2-image-captioning
- **Object Detection**: facebook/detr-resnet-50
- **Recipe Generation**: mistralai/Mistral-7B-Instruct-v0.2

## Setup Instructions

1. Clone this repository
   ```
   git clone https://github.com/lokeshpanthangi/RecipeSnap-t4.git
   cd recipe-snap
   ```

2. Create a virtual environment (recommended)
   ```
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. (Optional) Test your setup
   ```
   python test_setup.py
   ```

5. (Optional) Download models in advance
   ```
   python download_models.py
   ```

6. Run the application
   ```
   python app.py
   ```

7. Open your web browser and navigate to `http://localhost:5000`

## Important Notes

- The recipe generation model (Mistral-7B) requires significant computational resources. For optimal performance, a system with a dedicated GPU is recommended.
- The application includes fallback recipes in case the model loading fails due to resource limitations.
- For the best ingredient detection results, ensure your photos have good lighting and clearly visible ingredients.

## Usage

1. Launch the application
2. Upload an image of ingredients from your fridge
3. View the detected ingredients
4. Get recipe recommendations based on those ingredients
5. Enjoy cooking!

## Project Structure

```
recipe-snap/
├── app.py                # Main Flask application
├── download_models.py    # Script to download models in advance
├── test_setup.py         # Script to verify the setup
├── requirements.txt      # Project dependencies
├── models/               # AI model implementations
│   ├── __init__.py
│   ├── image_captioning.py
│   ├── object_detection.py
│   └── recipe_generation.py
├── static/               # Static files (CSS, JS, uploads)
│   └── uploads/          # Uploaded images are stored here
└── templates/            # HTML templates
    ├── index.html        # Home page
    └── results.html      # Recipe results page
```

## License

MIT