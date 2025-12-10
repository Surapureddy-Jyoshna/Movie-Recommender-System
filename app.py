import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

animated_bg_css = """
<style>
/* Full-page animated gradient */
.stApp {
    background: linear-gradient(-45deg, #ff9a9e, #fad0c4, #fad0c4, #fbc2eb, #a18cd1, #fbc2eb);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
}

/* Animation keyframes */
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Styling widget containers */
.css-ffhzg2, .css-1d391kg, .stSelectbox, .stButton button {
    background: rgba(255,255,255,0.4) !important;
    backdrop-filter: blur(6px);
    border-radius: 12px !important;
    padding: 8px;
}

/* Button styling */
.stButton button {
    font-weight: bold;
    border: none;
    color: #333;
}

/* Title text */
h1 {
    color: white !important;
    text-shadow: 0px 2px 6px rgba(0,0,0,0.3);
    text-align: center;
}
</style>
"""
st.markdown(animated_bg_css, unsafe_allow_html=True)

def recommend(movie):
    # movie_index =df[df['title']==movie].index
    movie_index = movies[movies['title'].str.lower() == movie.lower()].index[0]

    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    recommended_movies=[]
    for i in movies_list:
        movie_id=i[0]
        #poster fetch from API
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies
# similarity=pickle.load(open('similar.pkl','rb'))
movies_dict=pickle.load(open('movies_dict.pkl','rb'))
movies =pd.DataFrame(movies_dict)
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)
st.title("Movie Recommender System")
selected_movie=st.selectbox(
    'how would you like to be contacted',
    movies['title'].values
)
if st.button('recommend'):
    recommendations=recommend(selected_movie)
    for i in recommendations:
        st.write(i)


api_key = "1a3b9c5f6e8d7402c1aab4dd7e2bb5af"



















