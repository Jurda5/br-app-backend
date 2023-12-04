from fastapi import FastAPI
import os
from book_identifier import identify_book_from_prompt
from book_recommender import Recommendation, RecommendationRequest, RecommendationResponse, get_book_recommendations_by_title, get_book_title_variants, refresh_book_recommendation_datasets

# Set the path to the directory containing the kaggle.json file
os.environ['KAGGLE_CONFIG_DIR'] = './config'
 
app = FastAPI()

@app.get("/datasets/refresh")
async def refresh_datasets():
    """
    Refresh the book recommendation datasets.

    This endpoint triggers a refresh of the book recommendation datasets by downloading the latest versions from Kaggle.

    Returns:
        dict: A success or failure message.
    """
    success = refresh_book_recommendation_datasets()

    if not success:
        return {"message": "Kaggle datasets refresh failed"}

    return {"message": "Kaggle datasets refreshed"}

@app.get("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get book recommendations based on a prompt.

    This endpoint identifies a book from the given prompt and returns a list of recommended books.

    Args:
        request (RecommendationRequest): The request body containing the prompt and the number of recommendations to return.

    Returns:
        RecommendationResponse: The identified book and the list of recommended books.
    """
    identified_book = identify_book_from_prompt(request.prompt)
    
    book_title_variants = get_book_title_variants(identified_book['title'], identified_book['author'])

    chatGPT_rec, recommendations = get_book_recommendations_by_title(identified_book['title'], identified_book['author'], book_title_variants, request.number_of_recommendations)

    recommendation_models = [Recommendation(title=rec['title'], author=rec['author'], summary=rec['summary'], rating=rec.get('rating')) for rec in recommendations]

    return RecommendationResponse(title=identified_book['title'], author=identified_book['author'], title_alternative=identified_book['title_alternative'], summary=identified_book['summary'], chatGPT_rec=chatGPT_rec, recommendations=recommendation_models)

