
# Book Recommender

## Overview
This Python-based application leverages FastAPI, Pandas, OpenAI, and Kaggle libraries to provide book recommendations. Users can input a brief description of a book they have read, and the system will identify the book, retrieve its details, and suggest similar books based on user ratings from a Kaggle dataset.

## Features
- **Book Identification**: Utilizes OpenAI's new assistant, Book Identifier, to recognize books from user descriptions.
- **Data-driven Recommendations**: Fetches book title variants from a Kaggle dataset and recommends books based on user ratings.
- **AI-driven Recommendations**: In cases where data-driven recommendations are not available, the system uses Book Identifier to generate AI-powered suggestions.

## API Endpoints
- **GET `/recommendations`**: Based on the identified book, provides a list of recommended books.
- **GET `/datasets/refresh`**: Downloads and updates the dataset from Kaggle.

## Configuration
- **OpenAI API Key**: Set your OpenAI API key in the .env file for the Book Identifier functionality.
- **Kaggle config**: Set up your API Token for Kaggle.

