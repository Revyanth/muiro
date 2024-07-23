import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import yt_dlp
from streamlit_player import st_player
import json

# Set Streamlit theme and layout
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load and preprocess the dataset
data = pd.read_csv('Spotify.csv')

# Drop rows with invalid 'duration_ms' values (non-numeric)
data = data[pd.to_numeric(data['duration_ms'], errors='coerce').notna()]

# Extract numerical features (excluding 'time_signature')
numerical_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness',
                      'speechiness', 'tempo', 'valence']

# Extract categorical features and perform one-hot encoding
categorical_features = ['key']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Scale numerical features
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(data[numerical_features])

# Streamlit web app
st.title("🎵 Music Recommendation System")

# Sidebar with user input
st.sidebar.header("User Input")

# Create a dropdown bar for track names
track_names = data['track_name'].unique()
selected_song = st.sidebar.selectbox(
    "Choose a song from the list:",
    track_names,
    index=0,
    key="song_select"
)

num_recommendations = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=20, value=10)

# Function to get the first YouTube video URL based on search query
def get_first_youtube_result(query):
    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'quiet': True,
        'default_search': 'ytsearch',
        'max_downloads': 1
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(query, download=False)
        if 'entries' in result:
            video = result['entries'][0]
        else:
            video = result
        return f"https://www.youtube.com/watch?v={video['id']}"

# Recommendation function by song name
def recommend_tracks_by_name(song_name, num_recommendations):
    # Filter tracks that contain the given song name in their title
    matching_tracks = data[data['track_name'].str.contains(song_name, case=False, na=False)]

    if matching_tracks.empty:
        st.error(f"No tracks found with the name '{song_name}' in the dataset.")
        return

    # Calculate the mean of matching tracks' feature values to represent the song
    song_features = matching_tracks[numerical_features].mean().values.reshape(1, -1)

    # Calculate cosine similarity between the song and all other tracks
    song_similarity = cosine_similarity(song_features, data[numerical_features])

    # Get the indices of most similar tracks
    recommended_track_indices = song_similarity.argsort()[0][::-1][1:num_recommendations + 1]

    return data.iloc[recommended_track_indices][['artist_name', 'track_name']]


if st.sidebar.button("play"):
    recommendations = recommend_tracks_by_name(selected_song, num_recommendations)
    if recommendations is not None:
        

 # Ensure selected_song is a dictionary
        selected_song_info = data[data['track_name'] == selected_song].iloc[0]
        search_query = f"{selected_song_info['track_name']} {selected_song_info['artist_name']}"
        youtube_url = get_first_youtube_result(search_query)
        st.subheader(f"Playing: {selected_song_info['track_name']} by {selected_song_info['artist_name']}")
        st_player(youtube_url)

# Display recommendations and play the selected song's audio
if st.sidebar.button("Recommend"):
    recommendations = recommend_tracks_by_name(selected_song, num_recommendations)
    if recommendations is not None:
        st.sidebar.header("Recommended Tracks")
        st.sidebar.table(recommendations)

        # Ensure selected_song is a dictionary
        # if st.sidebar.button("play"):
        #     selected_song_info = data[data['track_name'] == selected_song].iloc[0]
        #     search_query = f"{selected_song_info['track_name']} {selected_song_info['artist_name']}"
        #     youtube_url = get_first_youtube_result(search_query)
        #     st.subheader(f"Playing: {selected_song_info['track_name']} by {selected_song_info['artist_name']}")
        #     st_player(youtube_url)



# Custom CSS for colorful and creative interface
st.markdown(
    """
    <style>
    body {
        background-color: #F7CAC9;
    }
    .st-bc {
        background-color: #000000;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .st-cv {
        background-color: #63D471;
        color: #FFFFFF;
        font-size: 24px;
        padding: 10px;
        border-radius: rem;
        text-align: center;
    }
    .st-bs {
        background-color: #000000;
        padding: 1rem;
        border-radius: 1rem;
    }
    .st-ek {
        background-color: #6495ED;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .st-co {
        color: #FF6B6B;
    }
    </style>
    """,
    unsafe_allow_html=True
)