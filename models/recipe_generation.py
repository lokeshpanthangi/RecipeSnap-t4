import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class RecipeGenerationModel:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            
            # Generation parameters
            self.max_length = 1024
            self.temperature = 0.7
            self.top_p = 0.9
            self.repetition_penalty = 1.1
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
            self.fallback_recipes = self._get_fallback_recipes()
        else:
            self.model_loaded = True
    
    def _get_fallback_recipes(self):
        """Return pre-defined recipes for common ingredient combinations"""
        recipes = {
            "tomato_onion_chicken": [
                {
                    "name": "Quick Chicken Cacciatore",
                    "ingredients": [
                        "2 chicken breasts", "1 onion, diced", "2 tomatoes, chopped",
                        "2 cloves garlic, minced", "1 tbsp olive oil",
                        "1 tsp dried oregano", "Salt and pepper to taste"
                    ],
                    "instructions": [
                        "Season chicken with salt and pepper.",
                        "Heat olive oil in a skillet over medium heat.",
                        "Cook chicken until browned on both sides, about 5 minutes per side.",
                        "Add onions and garlic, cook until softened.",
                        "Add tomatoes and oregano, simmer for 15 minutes until sauce thickens.",
                        "Serve hot with pasta or rice."
                    ]
                }
            ],
            "apple_cinnamon": [
                {
                    "name": "Simple Apple Crumble",
                    "ingredients": [
                        "3 apples, peeled and sliced", "1/2 cup flour", 
                        "1/2 cup rolled oats", "1/4 cup brown sugar",
                        "1/4 cup butter, cold and cubed", "1 tsp cinnamon",
                        "1/4 tsp nutmeg", "Pinch of salt"
                    ],
                    "instructions": [
                        "Preheat oven to 350°F (175°C).",
                        "Place apple slices in a baking dish.",
                        "Mix flour, oats, sugar, cinnamon, nutmeg, and salt in a bowl.",
                        "Cut in butter until mixture resembles coarse crumbs.",
                        "Sprinkle topping over apples.",
                        "Bake for 35-40 minutes until golden and bubbly.",
                        "Serve warm with ice cream if desired."
                    ]
                }
            ],
            "pasta_tomato_cheese": [
                {
                    "name": "Quick Pasta Marinara",
                    "ingredients": [
                        "8 oz pasta", "2 tomatoes, diced", "1/4 cup grated cheese",
                        "2 cloves garlic, minced", "1 tbsp olive oil",
                        "1 tsp dried basil", "Salt and pepper to taste"
                    ],
                    "instructions": [
                        "Cook pasta according to package directions.",
                        "In a separate pan, heat olive oil over medium heat.",
                        "Add garlic and cook for 30 seconds until fragrant.",
                        "Add tomatoes and basil, cook for 5-7 minutes.",
                        "Drain pasta and add to the sauce.",
                        "Top with cheese and serve immediately."
                    ]
                }
            ]
        }
        return recipes
    
    def _format_prompt(self, ingredients):
        """Format the ingredients into a prompt for the model"""
        ingredients_str = ", ".join(ingredients)
        prompt = f"""<s>[INST] You are a helpful cooking assistant. Create a delicious recipe using all or some of these ingredients: {ingredients_str}.
        
Please provide:
1. A creative recipe name
2. List of ingredients with measurements
3. Step-by-step cooking instructions
4. A brief description of the dish
5. Preparation time and cooking time

Keep the recipe practical and make it taste great! [/INST]"""
        return prompt
    
    def _parse_response(self, response):
        """Parse the model's response into a structured recipe format"""
        try:
            lines = response.split('\n')
            recipe = {}
            
            # Extract recipe name (typically the first non-empty line)
            for line in lines:
                if line.strip() and not line.lower().startswith(('recipe', 'here', '#')):
                    recipe['name'] = line.strip().strip('#').strip()
                    break
            
            # Extract ingredients and instructions
            in_ingredients = False
            in_instructions = False
            ingredients = []
            instructions = []
            
            for line in lines:
                line = line.strip()
                
                # Check section headers
                if "ingredient" in line.lower() and (":" in line or "-" in line or "#" in line):
                    in_ingredients = True
                    in_instructions = False
                    continue
                elif "instruction" in line.lower() and (":" in line or "-" in line or "#" in line):
                    in_ingredients = False
                    in_instructions = True
                    continue
                elif "preparation time" in line.lower() or "cooking time" in line.lower():
                    in_ingredients = False
                    in_instructions = False
                
                # Skip empty lines and section headers
                if not line or line.startswith("#") or ":" in line and len(line) < 20:
                    continue
                
                # Collect ingredients and instructions
                if in_ingredients and line:
                    if line.startswith("-") or line.startswith("*"):
                        line = line[1:].strip()
                    if any(char.isdigit() for char in line):  # Likely an ingredient with measurement
                        ingredients.append(line)
                
                if in_instructions and line:
                    if line.startswith("-") or line.startswith("*"):
                        line = line[1:].strip()
                    if line[0].isdigit() and line[1] in ['.', ')']: # Numbered instruction
                        line = line[2:].strip()
                    instructions.append(line)
            
            recipe["ingredients"] = ingredients
            recipe["instructions"] = instructions
            
            return recipe
        
        except Exception as e:
            print(f"Error parsing recipe response: {e}")
            return {
                "name": "Simple Recipe",
                "ingredients": ["Ingredient 1", "Ingredient 2", "Ingredient 3"],
                "instructions": ["Step 1: Cook ingredients", "Step 2: Serve and enjoy"]
            }
    
    def generate_recipes(self, ingredients):
        """
        Generate recipe recommendations based on detected ingredients
        
        Args:
            ingredients: List of detected ingredients
            
        Returns:
            list: List of recipe dictionaries
        """
        # If model failed to load, use fallback recipes
        if not self.model_loaded or not ingredients:
            # Return a default fallback recipe
            return [self.fallback_recipes["tomato_onion_chicken"][0]]
        
        try:
            # Create prompt from ingredients
            prompt = self._format_prompt(ingredients)
            
            # Generate recipe
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=True
                )
            
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract the response after the prompt
            response = response.split("[/INST]")[-1].strip()
            
            # Parse the response
            recipe = self._parse_response(response)
            
            # Return as a list of recipes
            return [recipe]
        
        except Exception as e:
            print(f"Error generating recipes: {e}")
            # Return a default fallback recipe
            return [self.fallback_recipes["tomato_onion_chicken"][0]] 