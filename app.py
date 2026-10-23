
import os
import json
import pickle
import requests
import bs4 as bs
import numpy as np
import pandas as pd
import urllib.request

from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "Artifacts")
DATA_PATH = os.path.join(ARTIFACTS_DIR, "main_data.csv")

# Load the trained sentiment model and vectorizer
try:
    with open(os.path.join(ARTIFACTS_DIR, "nlp_model.pkl"), "rb") as f:
        clf = pickle.load(f)

    # Your actual filename is tranform.pkl
    with open(os.path.join(ARTIFACTS_DIR, "tranform.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

except Exception as e:
    print("Error loading model artifacts:", e)
    clf = None
    vectorizer = None


# Load dataset and create similarity matrix
def create_similarity():
    data = pd.read_csv(DATA_PATH)

    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data["comb"].fillna(""))
    similarity_matrix = cosine_similarity(count_matrix)

    return data, similarity_matrix


# Get dataset and similarity matrix once
try:
    data, similarity_matrix = create_similarity()
except Exception as e:
    print("Error loading movie dataset:", e)
    data = None
    similarity_matrix = None


# Recommend similar movies
def rcmd(movie):
    global data, similarity_matrix

    if data is None or similarity_matrix is None:
        return "Movie dataset could not be loaded. Check main_data.csv."

    movie = movie.strip().casefold()

    titles = (
        data["movie_title"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    matches = data.index[titles == movie]

    if len(matches) == 0:
        return (
            "Movie not found. Please select a movie from "
            "the suggestions list or check the spelling."
        )

    movie_index = matches[0]

    scores = list(enumerate(similarity_matrix[movie_index]))
    scores.sort(key=lambda item: item[1], reverse=True)

    # Exclude the selected movie
    scores = [
        item for item in scores if item[0] != movie_index
    ][:10]

    return data.iloc[
        [item[0] for item in scores]
    ]["movie_title"].tolist()


# Convert serialized lists to Python lists
def convert_to_list(my_list):
    my_list = my_list.split('","')

    if not my_list:
        return []

    my_list[0] = my_list[0].replace('["', "")
    my_list[-1] = my_list[-1].replace('"]', "")

    return my_list


# Get movie title suggestions
def get_suggestions():
    movie_data = pd.read_csv(DATA_PATH)

    return (
        movie_data["movie_title"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )


# Home page
@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template(
        "home.html",
        suggestions=suggestions
    )


# Movie similarity API
@app.route("/similarity", methods=["POST"])
def similarity():
    movie = request.form.get("name", "").strip()

    if not movie:
        return "Please enter a movie name."

    result = rcmd(movie)

    if isinstance(result, str):
        return result

    return "---".join(result)


# Movie recommendation details page
@app.route("/recommend", methods=["POST"])
def recommend():
    # Get data from AJAX request
    title = request.form.get("title", "")
    cast_ids = request.form.get("cast_ids", "")
    cast_names = request.form.get("cast_names", "")
    cast_chars = request.form.get("cast_chars", "")
    cast_bdays = request.form.get("cast_bdays", "")
    cast_bios = request.form.get("cast_bios", "")
    cast_places = request.form.get("cast_places", "")
    cast_profiles = request.form.get("cast_profiles", "")
    imdb_id = request.form.get("imdb_id", "")
    poster = request.form.get("poster", "")
    genres = request.form.get("genres", "")
    overview = request.form.get("overview", "")
    vote_average = request.form.get("rating", "")
    vote_count = request.form.get("vote_count", "")
    release_date = request.form.get("release_date", "")
    runtime = request.form.get("runtime", "")
    status = request.form.get("status", "")
    rec_movies = request.form.get("rec_movies", "[]")
    rec_posters = request.form.get("rec_posters", "[]")

    suggestions = get_suggestions()

    # Convert serialized lists
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)

    cast_ids = cast_ids.strip("[]").split(",") if cast_ids else []

    cast_bios = [
        bio.replace(r"\n", "\n").replace(r'\"', '"')
        for bio in cast_bios
    ]

    # Create dictionaries safely
    movie_cards = dict(zip(rec_posters, rec_movies))

    casts = {
        cast_names[i]: [
            cast_ids[i] if i < len(cast_ids) else "",
            cast_chars[i] if i < len(cast_chars) else "",
            cast_profiles[i] if i < len(cast_profiles) else ""
        ]
        for i in range(len(cast_names))
    }

    cast_details = {
        cast_names[i]: [
            cast_ids[i] if i < len(cast_ids) else "",
            cast_profiles[i] if i < len(cast_profiles) else "",
            cast_bdays[i] if i < len(cast_bdays) else "",
            cast_places[i] if i < len(cast_places) else "",
            cast_bios[i] if i < len(cast_bios) else ""
        ]
        for i in range(len(cast_names))
    }

    # Fetch and classify IMDb reviews
    movie_reviews = {}

    if clf is not None and vectorizer is not None and imdb_id:
        try:
            url = (
                f"https://www.imdb.com/title/{imdb_id}/reviews/"
            )

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                soup = bs.BeautifulSoup(
                    response.read(),
                    "lxml"
                )

            review_elements = soup.select(
                ".text.show-more__control"
            )

            for review in review_elements:
                review_text = review.get_text(strip=True)

                if review_text:
                    review_vector = vectorizer.transform(
                        [review_text]
                    )
                    prediction = clf.predict(review_vector)[0]

                    movie_reviews[review_text] = (
                        "Good" if prediction == 1 else "Bad"
                    )

        except Exception as e:
            print("Could not retrieve IMDb reviews:", e)

    return render_template(
        "recommend.html",
        title=title,
        poster=poster,
        overview=overview,
        vote_average=vote_average,
        vote_count=vote_count,
        release_date=release_date,
        runtime=runtime,
        status=status,
        genres=genres,
        movie_cards=movie_cards,
        reviews=movie_reviews,
        casts=casts,
        cast_details=cast_details
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
if __name__ == "__main__":
    print("Avatar test:", rcmd("Avatar"))
    app.run(debug=True, host="127.0.0.1", port=5000)