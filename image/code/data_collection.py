import requests
import json
import time
from typing import Dict, Any
import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timezone

class RecipeDataCollector:
    def __init__(self):
        self.base_url = "https://www.themealdb.com/api/json/v1/1"
        self.s3_client = boto3.client('s3')
        
    def create_recipe_document(self, meal: Dict[str, Any]) -> Dict[str, Any]:
        """Transform meal data into recipe document format"""
        ingredients = []
        for i in range(1, 21):
            ing_name = meal.get(f'strIngredient{i}')
            measure = meal.get(f'strMeasure{i}')
            if ing_name and ing_name.strip() and measure and measure.strip():
                ingredients.append({
                    'name': ing_name.lower().strip(),
                    'measure': measure.strip(),
                    'is_main': i <= 3
                })

        return {
            "recipe_id": meal["idMeal"],
            "name": meal["strMeal"],
            "category": meal["strCategory"],
            "cuisine": meal["strArea"],
            "ingredients": ingredients,
            "instructions": meal["strInstructions"],
            "tags": meal.get("strTags", "").split(",") if meal.get("strTags") else [],
            "image_url": meal.get("strMealThumb", ""),
            "youtube_url": meal.get("strYoutube", ""),
            "created_at": datetime.now(timezone.utc).isoformat()
        }

    def create_recipe_metadata(self, recipe: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata file content for a recipe"""
        return {
            "metadataAttributes": {
                "recipe_id": recipe["recipe_id"],
                "category": recipe["category"],
                "cuisine": recipe["cuisine"],
                "tags": ",".join(recipe["tags"]),
                "main_ingredients": ",".join([ing["name"] for ing in recipe["ingredients"] if ing["is_main"]]),
                "created_at": recipe["created_at"]
            }
        }

    def save_recipe_to_s3(self, bucket: str, recipe: Dict[str, Any]):
        """Save recipe and its metadata to S3"""
        try:
            # Save recipe JSON
            recipe_key = f"recipes/{recipe['category']}/{recipe['recipe_id']}.json"
            self.s3_client.put_object(
                Bucket=bucket,
                Key=recipe_key,
                Body=json.dumps(recipe, ensure_ascii=False, indent=2).encode('utf-8'),
                ContentType='application/json'
            )

            # Save metadata JSON
            metadata = self.create_recipe_metadata(recipe)
            metadata_key = f"{recipe_key}.metadata.json"
            self.s3_client.put_object(
                Bucket=bucket,
                Key=metadata_key,
                Body=json.dumps(metadata, ensure_ascii=False, indent=2).encode('utf-8'),
                ContentType='application/json'
            )
            
            print(f"Saved recipe and metadata: {recipe['name']}")
            
        except ClientError as e:
            print(f"Error saving recipe {recipe['recipe_id']}: {str(e)}")
            raise

    def collect_recipes(self, bucket: str):
        """Collect and save recipes with metadata"""
        try:
            # Get categories
            response = requests.get(f"{self.base_url}/list.php?c=list")
            categories = response.json().get('meals', [])
            
            for category in categories:
                category_name = category['strCategory']
                print(f"Processing category: {category_name}")
                
                # Get meals in category
                meals_response = requests.get(f"{self.base_url}/filter.php?c={category_name}")
                meals = meals_response.json().get('meals', [])
                
                for meal in meals:
                    time.sleep(0.5)  # Respect API rate limits
                    
                    # Get meal details
                    details_response = requests.get(f"{self.base_url}/lookup.php?i={meal['idMeal']}")
                    meal_details = details_response.json().get('meals', [])[0]
                    
                    if meal_details:
                        recipe = self.create_recipe_document(meal_details)
                        self.save_recipe_to_s3(bucket, recipe)
                        
        except Exception as e:
            print(f"Error collecting recipes: {str(e)}")
            raise

def main():
    collector = RecipeDataCollector()
    bucket_name = "mongodb-recipe"  # Replace with your bucket name
    collector.collect_recipes(bucket_name)

if __name__ == "__main__":
    main()