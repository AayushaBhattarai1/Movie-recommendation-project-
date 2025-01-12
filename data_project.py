import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
movies = pd.read_csv('movies.csv')  # Ensure the dataset has 'title' and 'genres' columns

# Combine 'title' and 'genres' to form a single feature
movies['features'] = movies['title'] + " " + movies['genres']

# Convert text features into vectors
vectorizer = CountVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(movies['features'])

# Compute similarity
similarity = cosine_similarity(feature_matrix)

# Function to recommend movies
def recommend(title, n=5):
    if title not in movies['title'].values:
        return ["Movie not found!"]
    
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    recommended_indices = [i[0] for i in sim_scores[1:n+1]]
    return movies['title'].iloc[recommended_indices].tolist()

# Test the system
movie_title = "Toy Story (1995)"  # Replace with a movie from your dataset
print(f"Recommendations for '{movie_title}':")
print(recommend(movie_title))
