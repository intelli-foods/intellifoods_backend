from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel, Field
from langchain_aws import ChatBedrockConverse
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain.chains import RetrievalQA
from typing import List, Dict, Any, Set, Optional
import json
import re
import random

# Pydantic models for request/response validation
class IngredientsRequest(BaseModel):
    main_ingredients: List[str] = Field(..., min_items=1, description="Main ingredients that must be included")
    pantry_ingredients: List[str] = Field(default=[], description="Additional available ingredients")

class RecipeResponse(BaseModel):
    status: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    details: Optional[str] = None

class EnhancedRecipeRecommender:
    def __init__(self, knowledge_base_id: str, model_id: str):
        self.retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id,
            retrieval_config={
                "vectorSearchConfiguration": {
                    "numberOfResults": 10
                }
            },
        )

        self.llm = ChatBedrockConverse(
            model_id=model_id,
            temperature=0.7,
            max_tokens=2048,
        )

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True
        )

    def _clean_json_string(self, json_str: str) -> str:
        """Clean and prepare JSON string for parsing"""
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'\s*```', '', json_str)
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        return json_str.strip()

    def _extract_recipe_from_source(self, source_doc: Any) -> Dict:
        """Extract and parse recipe information from a source document"""
        try:
            content = source_doc.page_content
            content = self._clean_json_string(content)
            recipe_data = json.loads(content)
            return recipe_data
        except Exception as e:
            print(f"Error extracting recipe: {str(e)}")
            return None

    def _calculate_match_scores(
        self, 
        recipe: Dict, 
        main_ingredients: Set[str], 
        pantry_ingredients: Set[str]
    ) -> Dict[str, float]:
        """Calculate detailed matching scores for main and pantry ingredients"""
        recipe_ingredients = {ing['name'].lower(): ing for ing in recipe.get('ingredients', [])}
        
        # Main ingredients matching (required ingredients)
        main_matches = sum(1 for ing in main_ingredients if ing in recipe_ingredients)
        main_score = (main_matches / len(main_ingredients) * 100) if main_ingredients else 0
        
        # If not all main ingredients are present, return zero scores
        if main_matches != len(main_ingredients):
            return {
                'main_score': 0,
                'pantry_score': 0,
                'total_score': 0,
                'main_matches': 0,
                'pantry_matches': 0
            }
        
        # Pantry ingredients matching (optional ingredients)
        pantry_matches = sum(1 for ing in pantry_ingredients if ing in recipe_ingredients)
        pantry_score = (pantry_matches / len(recipe_ingredients) * 100) if recipe_ingredients else 0
        
        # Calculate weighted total score (70% main ingredients, 30% pantry ingredients)
        total_score = (main_score * 0.7) + (pantry_score * 0.3)
        
        return {
            'main_score': round(main_score, 2),
            'pantry_score': round(pantry_score, 2),
            'total_score': round(total_score, 2),
            'main_matches': main_matches,
            'pantry_matches': pantry_matches
        }

    def _clean_text(self, text: str) -> str:
        """Clean special characters and format temperatures"""
        text = text.replace('\u00b0', '°')
        text = text.replace('\u2109', '°F')
        text = text.replace('\u2103', '°C')
        text = re.sub(r'(\d+)C\b', r'\1°C', text)
        text = re.sub(r'(\d+)F\b', r'\1°F', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        return text.strip()

    def _format_instructions(self, instructions: str) -> List[str]:
        """Format cooking instructions into a list of clear steps"""
        steps = instructions.split('\r\n')
        formatted_steps = []
        
        for step in steps:
            cleaned_step = self._clean_text(step.strip())
            if cleaned_step:
                formatted_steps.append(cleaned_step)
                
        return formatted_steps

    def _process_source_documents(
        self, 
        source_docs: List[Any], 
        main_ingredients: Set[str], 
        pantry_ingredients: Set[str]
    ) -> Dict:
        """Process source documents and return a random valid recipe"""
        valid_recipes = []
        
        for doc in source_docs:
            recipe = self._extract_recipe_from_source(doc)
            if recipe:
                # Calculate match scores
                match_scores = self._calculate_match_scores(
                    recipe, 
                    main_ingredients, 
                    pantry_ingredients
                )
                
                # Only include recipes that match all main ingredients
                if match_scores['main_score'] == 100:
                    ingredients_info = []
                    for ingredient in recipe['ingredients']:
                        ingredient_name = self._clean_text(ingredient['name'].lower())
                        ingredients_info.append({
                            'name': self._clean_text(ingredient['name']),
                            'measure': self._clean_text(ingredient.get('measure', '')),
                            'is_main': ingredient_name in main_ingredients,
                            'is_available': (
                                ingredient_name in main_ingredients or 
                                ingredient_name in pantry_ingredients
                            )
                        })
                    
                    valid_recipes.append({
                        'recipe_name': self._clean_text(recipe['name']),
                        'category': recipe.get('category', 'Uncategorized'),
                        'cuisine': recipe.get('cuisine', 'Various'),
                        'match_scores': match_scores,
                        'ingredients': ingredients_info,
                        'steps': self._format_instructions(recipe['instructions']),
                        'image_url': recipe.get('image_url', ''),
                        'total_score': match_scores['total_score']
                    })
        
        # Return a random recipe from the valid ones
        if valid_recipes:
            # Filter to get recipes with good pantry match (above 10% match)
            good_matches = [r for r in valid_recipes if r['match_scores']['pantry_score'] > 10]
            
            # If we have good matches, select from those, otherwise select from all valid recipes
            selection_pool = good_matches if good_matches else valid_recipes
            return random.choice(selection_pool)
            
        return None

    def recommend_recipes(
        self, 
        main_ingredients: List[str], 
        pantry_ingredients: List[str]
    ) -> Dict:
        """Get a random recipe recommendation based on main and pantry ingredients"""
        try:
            # Convert ingredients to sets and normalize
            main_ingredients_set = {ing.lower().strip() for ing in main_ingredients}
            pantry_ingredients_set = {ing.lower().strip() for ing in pantry_ingredients}
            
            # Construct query emphasizing main ingredients
            query = f"""Find recipes that MUST include these main ingredients: {', '.join(main_ingredients)}. 
            The following pantry ingredients are also available: {', '.join(pantry_ingredients)}.
            Prioritize recipes that use more of the available ingredients."""
            
            response = self.qa.invoke(query)
            
            if "source_documents" not in response:
                return {
                    "status": "error",
                    "error": "No recipes found"
                }
            
            recipe = self._process_source_documents(
                response["source_documents"],
                main_ingredients_set,
                pantry_ingredients_set
            )
            
            if not recipe:
                return {
                    "status": "error",
                    "error": "No suitable recipes found with the required main ingredients"
                }
            
            return {
                "status": "success",
                "data": {
                    "recipe": recipe,
                    "metadata": {
                        "main_ingredients": main_ingredients,
                        "pantry_ingredients": pantry_ingredients
                    }
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "details": str(type(e))
            }

# Create FastAPI application
app = FastAPI(
    title="Recipe Recommender API",
    description="API for recommending recipes based on available ingredients",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
KNOWLEDGE_BASE_ID = "8EJQZBKWZD"
MODEL_ID = "ai21.jamba-1-5-mini-v1:0"

@app.get("/")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecipeResponse)
async def recommend_recipes(request: IngredientsRequest):
    """
    Get recipe recommendations based on available ingredients
    
    Example request body:
    ```json
    {
        "main_ingredients": ["chicken"],
        "pantry_ingredients": [
            "soy sauce",
            "garlic",
            "ginger",
            "onion",
            "vegetable oil",
            "salt",
            "pepper"
        ]
    }
    ```
    """
    try:
        recommender = EnhancedRecipeRecommender(KNOWLEDGE_BASE_ID, MODEL_ID)
        result = recommender.recommend_recipes(
            request.main_ingredients,
            request.pantry_ingredients
        )
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=404,
                detail=result["error"]
            )
            
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
# Create Lambda handler
handler = Mangum(app)

# Keep the main block for local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)