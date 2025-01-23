import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from dotenv import load_dotenv
import os

# Load and apply CSS
def load_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the CSS file
load_css("style.css")

# Load environment variables
load_dotenv()

# Spotify API credentials
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
if not CLIENT_ID or not CLIENT_SECRET:
    raise Exception("Spotify CLIENT_ID and CLIENT_SECRET are not set correctly.")

# Spotify authentication
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_excel('songdata.xlsx', sheet_name='Sheet1')

data = load_data()

# Preprocess the dataset
data = data.dropna(subset=['track_name', 'popularity', 'duration_ms', 'danceability', 
                           'energy', 'key', 'loudness', 'speechiness', 'acousticness', 
                           'instrumentalness', 'liveness', 'valence', 'tempo', 'track_genre'])
data = data.drop_duplicates(subset=['track_name'])

# Assign mood to songs
def assign_mood(row):
    mood_score = row['valence'] * 0.4 + row['energy'] * 0.4 + row['danceability'] * 0.2
    if mood_score > 0.9:
        return 'Happiness'
    elif mood_score > 0.75:
        return 'Pleasant Surprise'
    elif mood_score > 0.6:
        return 'Neutral'
    elif mood_score > 0.5:
        return 'Disgust'
    elif mood_score > 0.4:
        return 'Sadness'
    elif mood_score > 0.3:
        return 'Fear'
    elif mood_score > 0.2:
        return 'Anger'
    else:
        return 'Melancholic'

data['mood'] = data.apply(assign_mood, axis=1)

# Normalize features
features = ['popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 
            'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Recommendation functions
def recommend_songs_by_input(song_name, data, num_recommendations=5):
    song_row = data[data['track_name'].str.contains(song_name, case=False, na=False)]
    if song_row.empty:
        return None, None

    song_id = song_row.index[0]
    song_features = data.loc[song_id, features].values.reshape(1, -1)
    song_mood = data.loc[song_id, 'mood']

    mood_filtered_data = data[data['mood'] == song_mood]
    if mood_filtered_data.shape[0] <= 1:
        return mood_filtered_data[['track_name', 'mood']], None

    mood_filtered_features = mood_filtered_data[features]
    similarity_matrix = cosine_similarity(song_features, mood_filtered_features)
    similarity_scores = similarity_matrix.flatten()

    similar_song_indices = similarity_scores.argsort()[-num_recommendations-1:-1][::-1]
    recommended_songs = mood_filtered_data.iloc[similar_song_indices].copy()
    recommended_songs['similarity'] = similarity_scores[similar_song_indices]

    # Reset index and drop unnecessary columns
    recommended_songs = recommended_songs.reset_index(drop=True)

    return recommended_songs[['track_name', 'mood', 'similarity']], song_mood

def recommend_songs_by_mood(user_mood, data, num_recommendations=5):
    mood_filtered_data = data[data['mood'] == user_mood]
    if mood_filtered_data.empty:
        return None
    mood_filtered_data = mood_filtered_data.sample(n=min(num_recommendations, len(mood_filtered_data)))

    # Reset index and drop unnecessary columns
    mood_filtered_data = mood_filtered_data.reset_index(drop=True)

    return mood_filtered_data[['track_name', 'mood']]

# Spotify API integration for song playback
def search_song_on_spotify(song_name):
    results = sp.search(q=song_name, limit=1, type="track")
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        return {
            "name": track['name'],
            "artist": ", ".join(artist['name'] for artist in track['artists']),
            "url": track['external_urls']['spotify'],
            "preview_url": track['preview_url'],
            "album_art": track['album']['images'][0]['url'] if track['album']['images'] else None
        }
    return None

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Streamlit app
st.title("🎶 Song Recommendation ")
st.markdown("### 🎵 Discover Songs Tailored to Your Preferences and Mood!")

# Sidebar
st.sidebar.header("User Settings")
num_recommendations = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10, value=5)
selected_genre = st.sidebar.selectbox("Filter by Genre (Optional)", ["All"] + list(data['track_genre'].unique()))

import streamlit as st

# Custom CSS for Styling Tabs
st.markdown(
    """
    <style>
        div[data-testid="stTabs"] button {
            font-size: 18px !important;  /* Increase font size */
            padding: 12px 20px !important;  /* Increase padding */
            transition: all 0.3s ease-in-out; /* Smooth transition effect */
        }
        div[data-testid="stTabs"] button:hover {
            background-color: #FFD700 !important; /* Gold color on hover */
            color: black !important; /* Change text color on hover */
            transform: scale(1.05); /* Slight zoom effect */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Define Tabs
tab1, tab2, tab3 = st.tabs(["🎵 By Song Name", "😊 By Mood", "📜 Search History"])


# Tab 1: Recommendations by song name
with tab1:
    st.header("Recommend Songs Based on Song Name")
    song_name = st.text_input("Enter a Song Name", placeholder="Type a song name here...", key="song_input")

    if song_name != st.session_state.get("song_name", None):
        st.session_state["song_name"] = song_name

    if st.button("Recommend by Song"):
        if song_name:
            recommendations, mood = recommend_songs_by_input(song_name, data, num_recommendations)
            if recommendations is not None:
                st.success(f"🎧 Songs based on '{song_name}' (Mood: {mood}):")
                st.session_state["history"].append({"Type": "Song Search", "Input": song_name, "Mood": mood})

                # Display input song
                input_song_info = search_song_on_spotify(song_name)
                if input_song_info:
                    st.markdown("### **Your Input Song:**")
                    st.markdown(
                        f"""
                        <div style="text-align: center; background-color: #f9ca24; padding: 18px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);">
                            <img src="{input_song_info['album_art']}" alt="Album Art" width="150">
                            <h3 style="color: #1e272e;">🎵 {input_song_info['name']}</h3>
                            <p style="color: #1e272e;">by {input_song_info['artist']}</p>
                            <a href="{input_song_info['url']}" target="_blank">
                                <button style="background-color:#1DB954; color:white; padding:10px 20px; border:none; border-radius:5px; font-size:16px; cursor:pointer;">Play on Spotify</button>
                            </a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Display recommended songs
                st.markdown("### **Recommended Songs:**")
                cols = st.columns(2)  # Create 2 equal-width columns

                for idx, (_, row) in enumerate(recommendations.iterrows()):
                    col = cols[idx % 2]  # Alternate between columns
                    recommended_song_name = row['track_name']
                    recommended_song_info = search_song_on_spotify(recommended_song_name)

                    if recommended_song_info:
                        with col:
                            st.markdown(
                                f"""
                                <div style="background-color: #26309e; padding: 11px; border-radius: 10px;margin-bottom:12px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3); text-align: center;">
                                    <img src="{recommended_song_info['album_art']}" alt="Album Art" width="150">
                                    <h4 style="color: #ffffff;">🎵 {recommended_song_info['name']}</h4>
                                    <p style="color: #ffffff;">by {recommended_song_info['artist']}</p>
                                    <a href="{recommended_song_info['url']}" target="_blank">
                                        <button style="background-color:#ab265f; color:white; padding:10px 20px; border:none; border-radius:5px; font-size:16px; cursor:pointer;">Play on Spotify</button>
                                    </a>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Display recommended songs in table format
                st.markdown("### **Recommended Songs Table:**")
                st.dataframe(
                    recommendations[['track_name', 'mood', 'similarity']].rename(
                        columns={"track_name": "Song Name", "mood": "Mood", "similarity": "Similarity"}
                    ),
                    use_container_width=True,
                )
            else:
                st.error("No similar songs found. Try a different song name.")
        else:
            st.warning("Please enter a song name.")

# Tab 2: Recommendations by mood
with tab2:
    st.header("Recommend Songs Based on Your Mood")
    user_mood = st.selectbox("Select Your Mood", 
                             ['Happiness', 'Pleasant Surprise', 'Neutral', 'Disgust', 
                              'Sadness', 'Fear', 'Anger'])

    if user_mood != st.session_state.get("user_mood", None):
        st.session_state["user_mood"] = user_mood

    if st.button("Recommend by Mood"):
        recommendations = recommend_songs_by_mood(user_mood, data, num_recommendations)
        if recommendations is not None:
            st.success(f"🌟 Songs matching your mood ({user_mood}):")
            
            # Add search to history
            st.session_state["history"].append({"Type": "Mood Search", "Input": user_mood, "Mood": user_mood})
            
            # Horizontal layout for songs
            cols = st.columns(2)  # Create 2 equal-width columns

            for idx, (_, row) in enumerate(recommendations.iterrows()):
                col = cols[idx % 2]  # Alternate between columns
                recommended_song_name = row['track_name']
                recommended_song_info = search_song_on_spotify(recommended_song_name)

                if recommended_song_info:
                    with col:
                        st.markdown(
                            f"""
                            <div style="background-color: #26309e; padding: 11px; border-radius: 10px;margin-bottom:12px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3); text-align: center;">
                                <img src="{recommended_song_info['album_art']}" alt="Album Art" width="150">
                                <h4 style="color: #ffffff;">🎵 {recommended_song_info['name']}</h4>
                                <p style="color: #ffffff;">by {recommended_song_info['artist']}</p>
                                <a href="{recommended_song_info['url']}" target="_blank">
                                    <button style="background-color:#ab265f; color:white; padding:10px 20px; border:none; border-radius:5px; font-size:16px; cursor:pointer;">Play on Spotify</button>
                                </a>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

# Tab 3: View Search History
with tab3:
    st.header("Search History")

    # Check if the history exists in the session
    if st.session_state["history"]:
        history_df = pd.DataFrame(st.session_state["history"])

        # Add emojis to indicate the type of search
        history_df["Search Type Icon"] = history_df["Type"].apply(
            lambda x: "🎵" if x == "Song Search" else "🔍" if x == "Mood Search" else "❓"
        )

        # Define color mapping for moods
        mood_colors = {
            'Happiness': '#FFD700',  # Gold
            'Pleasant Surprise': '#FF6347',  # Tomato Red
            'Neutral': '#A9A9A9',  # Dark Gray
            'Disgust': '#8B0000',  # Dark Red
            'Sadness': '#6495ED',  # Cornflower Blue
            'Fear': '#FF4500',  # Orange Red
            'Anger': '#DC143C'  # Crimson
        }

        # Define color mapping for search types
        type_colors = {
            'Song Search': '#228B22',  # Forest Green
            'Mood Search': '#4682B4'  # Steel Blue
        }

        # Define emoji mapping for moods
        mood_emojis = {
            'Happiness': "😃",
            'Pleasant Surprise': "🤩",
            'Neutral': "😐",
            'Disgust': "🤢",
            'Sadness': "😢",
            'Fear': "😨",
            'Anger': "😡"
        }

        # Append emojis to Mood column
        if 'Mood' in history_df.columns:
            history_df["Mood"] = history_df["Mood"].apply(lambda x: f"{mood_emojis.get(x, '❓')} {x}")

        # Styling function
        def style_history_df(df):
            if 'Mood' in df.columns:
                return df.style.applymap(lambda x: f'background-color: {mood_colors.get(x.split(" ")[-1], "#1E1E1E")}', subset=["Mood"]) \
                               .applymap(lambda x: f'background-color: {type_colors.get(x, "#1E1E1E")}', subset=["Type"]) \
                               .set_table_styles([
                                   {'selector': 'td:hover', 'props': [('background-color', '#5F9EA0')]},  # Hover effect
                                   {'selector': 'th:hover', 'props': [('background-color', '#708090')]},  # Header hover
                                   {'selector': 'td', 'props': [('color', 'white')]},  # White text
                                   {'selector': 'th', 'props': [('color', 'white'), ('font-weight', 'bold')]},  # Bold white headers
                                   {'selector': 'table', 'props': [('background-color', '#262626'), ('border', '1px solid white')]}  # Dark mode
                               ])
            else:
                return df.style.applymap(lambda x: f'background-color: {type_colors.get(x, "#1E1E1E")}', subset=["Type"]) \
                               .set_table_styles([
                                   {'selector': 'td:hover', 'props': [('background-color', '#5F9EA0')]},
                                   {'selector': 'th:hover', 'props': [('background-color', '#708090')]},
                                   {'selector': 'td', 'props': [('color', 'white')]},
                                   {'selector': 'th', 'props': [('color', 'white'), ('font-weight', 'bold')]},
                                   {'selector': 'table', 'props': [('background-color', '#262626'), ('border', '1px solid white')]}
                               ])

        # Display the styled dataframe
        st.dataframe(
            style_history_df(history_df),
            width=1000,  # Wider table
            height=400   # Taller table
        )

        # Add filter option to display only mood-based or song-based search history
        search_filter = st.selectbox("Filter History by Search Type", ["All", "Song Search", "Mood Search"])

        if search_filter != "All":
            filtered_history = history_df[history_df["Type"] == search_filter]
            st.dataframe(
                style_history_df(filtered_history),
                width=1000,
                height=400
            )

        # Show the total number of searches in history
        st.markdown(f"### Total searches: {len(st.session_state['history'])}")

    else:
        st.warning("No search history found yet.")
