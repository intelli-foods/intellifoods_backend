from fastapi import FastAPI, HTTPException, status
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

# Pydantic models
class BaseIngredientsRequest(BaseModel):
    main_ingredients: List[str] = Field(..., min_items=1, description="Main ingredients that must be included")
    pantry_ingredients: List[str] = Field(default=[], description="Additional available ingredients")

class RecipeResponse(BaseModel):
    status: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    details: Optional[str] = None

class BaseRecipeRecommender:
    def __init__(self, knowledge_base_id: str, model_id: str):
        self.retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id,
            retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 10}}
        )
        self.llm = ChatBedrockConverse(
            model_id=model_id,
            temperature=0.7,
            max_tokens=2048
        )
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True
        )

    def _clean_text(self, text: str) -> str:
        """Clean special characters and format temperatures"""
        replacements = {
            '\u00b0': '°',
            '\u2109': '°F',
            '\u2103': '°C'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = re.sub(r'(\d+)C\b', r'\1°C', text)
        text = re.sub(r'(\d+)F\b', r'\1°F', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        return text.strip()

    def _clean_json_string(self, json_str: str) -> str:
        """Clean and prepare JSON string for parsing"""
        json_str = re.sub(r'```json\s*|\s*```', '', json_str)
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        return json_str.strip()

    def _extract_recipe_from_source(self, source_doc: Any) -> Dict:
        """Extract and parse recipe information from a source document"""
        try:
            content = self._clean_json_string(source_doc.page_content)
            return json.loads(content)
        except Exception as e:
            print(f"Error extracting recipe: {str(e)}")
            return None

    def _calculate_match_scores(self, recipe: Dict, main_ingredients: Set[str], pantry_ingredients: Set[str]) -> Dict[str, float]:
        """Calculate detailed matching scores for main and pantry ingredients"""
        recipe_ingredients = {ing['name'].lower(): ing for ing in recipe.get('ingredients', [])}
        
        main_matches = sum(1 for ing in main_ingredients if ing in recipe_ingredients)
        main_score = (main_matches / len(main_ingredients) * 100) if main_ingredients else 0
        
        if main_matches != len(main_ingredients):
            return {
                'main_score': 0,
                'pantry_score': 0,
                'total_score': 0,
                'main_matches': 0,
                'pantry_matches': 0
            }
        
        pantry_matches = sum(1 for ing in pantry_ingredients if ing in recipe_ingredients)
        pantry_score = (pantry_matches / len(recipe_ingredients) * 100) if recipe_ingredients else 0
        
        total_score = (main_score * 0.7) + (pantry_score * 0.3)
        
        return {
            'main_score': round(main_score, 2),
            'pantry_score': round(pantry_score, 2),
            'total_score': round(total_score, 2),
            'main_matches': main_matches,
            'pantry_matches': pantry_matches
        }

    def _format_instructions(self, instructions: str) -> List[str]:
        """Format cooking instructions into a list of clear steps"""
        return [self._clean_text(step.strip()) for step in instructions.split('\r\n') if step.strip()]

    def _get_ingredient_info(self, ingredient: Dict, main_ingredients: Set[str], pantry_ingredients: Set[str], include_substitutes: bool = False) -> Dict:
        """Get ingredient information including availability and optional substitutes"""
        ingredient_name = self._clean_text(ingredient['name'].lower())
        ingredient_measure = self._clean_text(ingredient.get('measure', ''))
        
        is_main = ingredient_name in main_ingredients
        is_available = ingredient_name in main_ingredients or ingredient_name in pantry_ingredients
        
        info = {
            'name': self._clean_text(ingredient['name']),
            'measure': ingredient_measure,
            'is_main': is_main,
            'available': is_available
        }
        
        if include_substitutes and not is_available:
            substitutes = self._get_ingredient_substitutions(ingredient['name'], ingredient_measure)
            if substitutes:
                info['substitutes'] = substitutes
        
        return info, is_available

    def _get_ingredient_substitutions(self, ingredient: str, measure: str) -> List[Dict[str, str]]:
        """Get possible substitutions for a missing ingredient"""
        try:
            prompt = f"""Suggest 2-3 common substitute ingredients for {measure} {ingredient} in cooking. 
            Return the response in this exact JSON format:
            [
                {{"ingredient": "substitute1", "measure": "amount1"}},
                {{"ingredient": "substitute2", "measure": "amount2"}}
            ]
            Only suggest real, practical substitutions that would work in most recipes.
            If no good substitution exists, return an empty list."""

            response = self.llm.invoke(prompt)
            
            try:
                suggestions = json.loads(response.content)
                return suggestions[:3] if isinstance(suggestions, list) else []
            except json.JSONDecodeError:
                matches = re.findall(r'{\s*"ingredient":\s*"([^"]+)",\s*"measure":\s*"([^"]+)"\s*}', response.content)
                return [{"ingredient": ing, "measure": mea} for ing, mea in matches[:3]]
                
        except Exception as e:
            print(f"Error getting substitutions for {ingredient}: {str(e)}")
            return []

    def _process_source_documents(self, source_docs: List[Any], main_ingredients: Set[str], pantry_ingredients: Set[str], include_substitutes: bool = False) -> Dict:
        """Process source documents and return a random valid recipe"""
        valid_recipes = []
        
        for doc in source_docs:
            recipe = self._extract_recipe_from_source(doc)
            if not recipe:
                continue
                
            match_scores = self._calculate_match_scores(recipe, main_ingredients, pantry_ingredients)
            if match_scores['main_score'] != 100:
                continue
                
            ingredients_info = []
            missing_ingredients = []
            
            for ingredient in recipe['ingredients']:
                ing_info, is_available = self._get_ingredient_info(
                    ingredient, 
                    main_ingredients, 
                    pantry_ingredients,
                    include_substitutes
                )
                ingredients_info.append(ing_info)
                
                if include_substitutes and not is_available:
                    missing_ingredients.append({
                        'name': ingredient['name'],
                        'measure': ingredient.get('measure', ''),
                        'substitutes': ing_info.get('substitutes', [])
                    })
            
            recipe_info = {
                'recipe_name': self._clean_text(recipe['name']),
                'category': recipe.get('category', 'Uncategorized'),
                'cuisine': recipe.get('cuisine', 'Various'),
                'match_scores': match_scores,
                'ingredients': ingredients_info,
                'steps': self._format_instructions(recipe['instructions']),
                'image_url': recipe.get('image_url', ''),
                'total_score': match_scores['total_score']
            }
            
            if include_substitutes:
                recipe_info['missing_ingredients'] = missing_ingredients
                
            valid_recipes.append(recipe_info)
        
        if valid_recipes:
            good_matches = [r for r in valid_recipes if r['match_scores']['pantry_score'] > (30 if include_substitutes else 10)]
            selection_pool = good_matches if good_matches else valid_recipes
            return random.choice(selection_pool)
            
        return None

    async def recommend_recipes(self, request: BaseIngredientsRequest, include_substitutes: bool = False) -> RecipeResponse:
        """Get a recipe recommendation based on ingredients"""
        try:
            main_ingredients = {ing.lower().strip() for ing in request.main_ingredients if ing.strip()}
            pantry_ingredients = {ing.lower().strip() for ing in request.pantry_ingredients if ing.strip()}
            
            if not main_ingredients:
                raise ValueError("At least one main ingredient is required")
            
            query = f"""Find recipes that MUST include these main ingredients: {', '.join(request.main_ingredients)}. 
            The following pantry ingredients are also available: {', '.join(request.pantry_ingredients)}.
            Prioritize recipes that use more of the available ingredients."""
            
            response = self.qa.invoke(query)
            
            if "source_documents" not in response:
                return RecipeResponse(status="error", error="No recipes found")
            
            recipe = self._process_source_documents(
                response["source_documents"],
                main_ingredients,
                pantry_ingredients,
                include_substitutes
            )
            
            if not recipe:
                return RecipeResponse(
                    status="error",
                    error="No suitable recipes found with the required main ingredients"
                )
            
            return RecipeResponse(
                status="success",
                data={
                    "recipe": recipe,
                    "metadata": {
                        "main_ingredients": list(main_ingredients),
                        "pantry_ingredients": list(pantry_ingredients)
                    }
                }
            )
            
        except Exception as e:
            return RecipeResponse(
                status="error",
                error=str(e),
                details=str(type(e))
            )

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

# Initialize recommender
recommender = BaseRecipeRecommender(KNOWLEDGE_BASE_ID, MODEL_ID)

@app.get("/")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecipeResponse)
async def recommend_recipes(request: BaseIngredientsRequest):
    """
    Get recipe recommendations based on available ingredients without substitutions
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
        result = await recommender.recommend_recipes(request, include_substitutes=False)
        if result.status == "error":
            raise HTTPException(status_code=404, detail=result.error)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/substitute", response_model=RecipeResponse)
async def get_recipe_with_substitutions(request: BaseIngredientsRequest):
    """
    Get recipe recommendations with substitutions for missing ingredients
    
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
        result = await recommender.recommend_recipes(request, include_substitutes=True)
        if result.status == "error":
            if "No recipes found" in str(result.error):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No recipes found with the required ingredients: {', '.join(request.main_ingredients)}"
                )
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(result.error))
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request"
        )

# Create Lambda handler
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)