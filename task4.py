import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset
data = {
    'title': ['The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 'Tenet'],
    'description': [
        'A computer hacker learns about the true nature of reality and his role in the war against its controllers.',
        'A thief who steals corporate secrets through the use of dream-sharing technology.',
        'A team of explorers travel through a wormhole in space in an attempt to ensure humanity’s survival.',
        'Batman faces the Joker, a criminal mastermind who wants to plunge Gotham into chaos.',
        'A secret agent manipulates the flow of time to prevent World War III.'
    ]
}

df = pd.DataFrame(data)

# Step 1: Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['description'])

# Step 2: Compute similarity (cosine)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 3: Recommend function
def recommend(movie_title):
    idx = df[df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Top 3 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Sample usage
print("Recommended for 'Inception':")
print(recommend('Inception'))
