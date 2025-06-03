import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import uuid
from models.image_captioning import ImageCaptioningModel
from models.object_detection import ObjectDetectionModel
from models.recipe_generation import RecipeGenerationModel
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-dev-only')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize models
image_captioning_model = ImageCaptioningModel()
object_detection_model = ObjectDetectionModel()
recipe_model = RecipeGenerationModel()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['image']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Create unique filename
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{file_extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        caption = image_captioning_model.generate_caption(filepath)
        ingredients = object_detection_model.detect_ingredients(filepath)
        
        # Store in session for the next step
        return redirect(url_for('results', filename=filename))
    
    flash('Invalid file type. Please upload a JPG, JPEG or PNG image.')
    return redirect(url_for('index'))

@app.route('/results/<filename>')
def results(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Get ingredients from the image
    caption = image_captioning_model.generate_caption(filepath)
    ingredients = object_detection_model.detect_ingredients(filepath)
    
    # Generate recipe recommendations
    recipes = recipe_model.generate_recipes(ingredients)
    
    return render_template('results.html', 
                          filename=filename,
                          caption=caption,
                          ingredients=ingredients,
                          recipes=recipes)

@app.route('/api/ingredients', methods=['POST'])
def api_detect_ingredients():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Create unique filename
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{file_extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        caption = image_captioning_model.generate_caption(filepath)
        ingredients = object_detection_model.detect_ingredients(filepath)
        
        return jsonify({
            'caption': caption,
            'ingredients': ingredients,
            'image_path': url_for('static', filename=f'uploads/{filename}')
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/recipes', methods=['POST'])
def api_generate_recipes():
    data = request.json
    if not data or 'ingredients' not in data:
        return jsonify({'error': 'No ingredients provided'}), 400
    
    ingredients = data['ingredients']
    recipes = recipe_model.generate_recipes(ingredients)
    
    return jsonify({
        'recipes': recipes
    })

if __name__ == '__main__':
    app.run(debug=True) 