from typing import List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics.pairwise import euclidean_distances
import streamlit as st

st.set_page_config(
    page_title="Spotify Recommender System",
    layout="wide",
    initial_sidebar_state="expanded"
)

AUDIO_FEATURES: List[str] = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

CLUSTER_NAMES: dict[int, str] = {
    0: "Party & High Energy",
    1: "Chill & Acoustic Mood",
    2: "Instrumental & Focus",
    3: "Speechy & Urban Vibe",
    4: "Balanced Pop / Melodic"
}

# --- Custom Typography & Industrial Layout Styling ---
st.markdown("""
<style>
    .main { 
        background-color: #121212; 
    }
    .stApp { 
        color: #E0E0E0; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Styling */
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #FFFFFF;
        margin-bottom: 4px;
        border-bottom: 2px solid #1DB954;
        padding-bottom: 8px;
        display: inline-block;
    }
    .hero-subtitle {
        font-size: 0.9rem;
        color: #A0A0A0;
        margin-top: 8px;
        margin-bottom: 24px;
    }
    
    /* Card Components */
    .song-card {
        background-color: #181818;
        border-radius: 4px;
        padding: 14px 16px;
        margin-bottom: 12px;
        border: 1px solid #282828;
        transition: border-color 0.15s ease-in-out;
    }
    .song-card:hover {
        border-color: #1DB954;
    }
    .song-title {
        font-size: 1rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 2px;
    }
    .song-artist {
        font-size: 0.85rem;
        color: #1DB954;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .song-meta {
        font-size: 0.78rem;
        color: #888888;
    }
    
    /* Badges */
    .badge-cluster {
        display: inline-block;
        background-color: #222222;
        color: #1DB954;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 0.72rem;
        font-weight: 600;
        border: 1px solid #1DB954;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-distance {
        display: inline-block;
        background-color: #1DB954;
        color: #000000;
        padding: 2px 6px;
        border-radius: 2px;
        font-size: 0.72rem;
        font-weight: 700;
        float: right;
        font-family: monospace;
    }
    
    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #121212;
        border-bottom: 1px solid #282828;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #181818;
        border-radius: 4px 4px 0px 0px;
        color: #888888;
        padding: 8px 16px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #282828 !important;
        color: #1DB954 !important;
        border-bottom: 2px solid #1DB954 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Resource Loading ---
@st.cache_resource
def load_artifacts() -> Tuple[Optional[object], Optional[object]]:
    try:
        scaler = joblib.load('scaler.pkl')
        model = joblib.load('model_kmeans.pkl')
        
        # Patching dtypes ke float64 secara eksplisit
        for attr in ['mean_', 'scale_', 'var_']:
            if hasattr(scaler, attr) and getattr(scaler, attr) is not None:
                setattr(scaler, attr, getattr(scaler, attr).astype(np.float64))
        if hasattr(model, 'cluster_centers_'):
            model.cluster_centers_ = model.cluster_centers_.astype(np.float64)
            
        return scaler, model
    except Exception as e:
        st.error(f"Gagal memuat artefak model (.pkl): {e}")
        return None, None


@st.cache_data
def load_dataset() -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet('spotify_cleaned.parquet')
    except Exception as e:
        st.error(f"Gagal memuat dataset (.parquet): {e}")
        return None


# --- Core Logic Helpers ---
def transform_scaled(scaler_obj: object, data_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(data_array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return scaler_obj.transform(arr).astype(np.float64)


def format_duration(ms: float) -> str:
    seconds = int((ms / 1000) % 60)
    minutes = int((ms / (1000 * 60)) % 60)
    return f"{minutes}:{seconds:02d}"


def recommend_songs(
    input_vector: np.ndarray,
    target_cluster: int,
    df: pd.DataFrame,
    scaler_obj: object,
    top_n: int = 5,
    exclude_track_id: Optional[str] = None
) -> pd.DataFrame:
    cluster_songs = df[df['cluster_id'] == target_cluster].copy()
    if cluster_songs.empty:
        return pd.DataFrame()
    
    input_scaled = transform_scaled(scaler_obj, input_vector)
    candidate_raw = cluster_songs[AUDIO_FEATURES].to_numpy(dtype=np.float64)
    candidate_scaled = scaler_obj.transform(candidate_raw).astype(np.float64)
    
    distances = euclidean_distances(input_scaled, candidate_scaled)[0]
    cluster_songs['distance'] = distances
    
    if exclude_track_id:
        cluster_songs = cluster_songs[cluster_songs['track_id'] != exclude_track_id]
        
    return cluster_songs.sort_values(by='distance', ascending=True).head(top_n)


def create_radar_chart(input_features: np.ndarray, recommended_df: pd.DataFrame) -> go.Figure:
    def normalize_features(values: np.ndarray) -> np.ndarray:
        val = np.array(values, dtype=np.float64).copy()
        val[2] = np.clip((val[2] + 60.0) / 60.0, 0.0, 1.0)
        val[8] = np.clip((val[8] - 50.0) / 170.0, 0.0, 1.0)
        return val

    input_norm = normalize_features(input_features)
    rec_avg_norm = normalize_features(recommended_df[AUDIO_FEATURES].mean().to_numpy(dtype=np.float64))
    labels = [f.capitalize() for f in AUDIO_FEATURES]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=input_norm, theta=labels, fill='toself', name='Target Input', line_color='#1DB954'
    ))
    fig.add_trace(go.Scatterpolar(
        r=rec_avg_norm, theta=labels, fill='toself', name='Rata-Rata Rekomendasi', line_color='#00BFFF'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor='#282828'),
            angularaxis=dict(gridcolor='#282828'),
            bgcolor='#181818'
        ),
        paper_bgcolor='#121212',
        plot_bgcolor='#121212',
        font=dict(color='#E0E0E0', size=11),
        showlegend=True,
        margin=dict(l=40, r=40, t=30, b=30)
    )
    return fig


# --- Main Application Execution ---
scaler, kmeans_model = load_artifacts()
df_songs = load_dataset()

# Sidebar Controls
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", width=36)
st.sidebar.title("Parameter Kontrol")

input_mode = st.sidebar.radio("Metode Input:", ["Cari Berdasarkan Lagu", "Custom Audio Mood (Slider)"])
top_n = st.sidebar.slider("Jumlah Rekomendasi (Top-N):", 3, 15, 5)

use_genre_filter = st.sidebar.checkbox("Batasi Genre Spesifik?", False)
selected_genre = None
if use_genre_filter and df_songs is not None:
    selected_genre = st.sidebar.selectbox("Pilih Genre Target:", sorted(df_songs['track_genre'].dropna().unique()))

# Main Header
st.markdown('<div class="hero-title">Spotify Track Clustering & Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Sistem rekomendasi musik berbasis K-Means Clustering dan analisis kemiripan fitur audio (Euclidean Distance).</div>', unsafe_allow_html=True)

if df_songs is None or scaler is None or kmeans_model is None:
    st.error("Artefak model atau dataset gagal dimuat. Pastikan file tersimpan di direktori yang sesuai.")
    st.stop()

# Mode Selection Logic
if input_mode == "Cari Berdasarkan Lagu":
    st.subheader("1. Pilih Lagu Acuan")
    
    if 'song_label' not in df_songs.columns:
        df_songs['song_label'] = df_songs['track_name'] + " — " + df_songs['artists']
        
    selected_song_label = st.selectbox("Ketik judul lagu atau nama artis:", df_songs['song_label'].values)
    
    target_song = df_songs[df_songs['song_label'] == selected_song_label].iloc[0]
    input_audio_vector = target_song[AUDIO_FEATURES].to_numpy(dtype=np.float64)
    target_cluster = int(target_song['cluster_id'])
    target_track_id = str(target_song['track_id'])
    
    col1, col2 = st.columns([1.3, 1])
    with col1:
        duration_str = format_duration(target_song.get('duration_ms', 0))
        st.markdown(f"""
        <div class="song-card" style="border-color: #1DB954;">
            <div class="song-title">{target_song['track_name']}</div>
            <div class="song-artist">by {target_song['artists']}</div>
            <div class="song-meta"><b>Album:</b> {target_song['album_name']} | <b>Genre:</b> {target_song['track_genre']} | <b>Durasi:</b> {duration_str}</div>
            <br>
            <span class="badge-cluster">{CLUSTER_NAMES.get(target_cluster, f'Cluster {target_cluster}')}</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f'''
        <iframe src="https://open.spotify.com/embed/track/{target_track_id}" width="100%" height="115" 
        frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
        ''', unsafe_allow_html=True)

    filtered_df = df_songs if not (use_genre_filter and selected_genre) else df_songs[df_songs['track_genre'] == selected_genre]
    recommendations = recommend_songs(input_audio_vector, target_cluster, filtered_df, scaler, top_n, target_track_id)

else:
    st.subheader("1. Parameter Mood Audio")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.65, 0.01)
        energy = st.slider("Energy", 0.0, 1.0, 0.70, 0.01)
        valence = st.slider("Valence", 0.0, 1.0, 0.50, 0.01)
    with col2:
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.20, 0.01)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.00, 0.01)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.08, 0.01)
    with col3:
        liveness = st.slider("Liveness", 0.0, 1.0, 0.15, 0.01)
        loudness = st.slider("Loudness (dB)", -60.0, 0.0, -8.0, 0.5)
        tempo = st.slider("Tempo (BPM)", 50.0, 220.0, 120.0, 1.0)
        
    input_audio_vector = np.array([
        danceability, energy, loudness, speechiness, 
        acousticness, instrumentalness, liveness, valence, tempo
    ], dtype=np.float64)
    
    input_scaled = transform_scaled(scaler, input_audio_vector)
    target_cluster = int(kmeans_model.predict(input_scaled)[0])
    
    st.markdown(f"**Cluster Terprediksi:** <span class=\"badge-cluster\">{CLUSTER_NAMES.get(target_cluster, f'Cluster {target_cluster}')}</span>", unsafe_allow_html=True)
    
    filtered_df = df_songs if not (use_genre_filter and selected_genre) else df_songs[df_songs['track_genre'] == selected_genre]
    recommendations = recommend_songs(input_audio_vector, target_cluster, filtered_df, scaler, top_n)

# Recommendations Output Section
st.markdown("---")
st.subheader("2. Hasil Rekomendasi Lagu")

if recommendations.empty:
    st.warning("Tidak ditemukan lagu yang memenuhi kriteria filter.")
else:
    tab1, tab2, tab3 = st.tabs(["Daftar Rekomendasi", "Profil Sinyal Audio", "Statistik Cluster"])
    
    with tab1:
        for _, row in recommendations.iterrows():
            c1, c2 = st.columns([1.4, 1])
            with c1:
                dur_str = format_duration(row.get('duration_ms', 0))
                dist_val = round(float(row['distance']), 4)
                st.markdown(f"""
                <div class="song-card">
                    <span class="badge-distance">DIST: {dist_val}</span>
                    <div class="song-title">{row['track_name']}</div>
                    <div class="song-artist">by {row['artists']}</div>
                    <div class="song-meta"><b>Album:</b> {row['album_name']} | <b>Genre:</b> {row['track_genre']} | <b>Durasi:</b> {dur_str}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f'''
                <iframe src="https://open.spotify.com/embed/track/{row['track_id']}" width="100%" height="80" 
                frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
                ''', unsafe_allow_html=True)
                
    with tab2:
        st.plotly_chart(create_radar_chart(input_audio_vector, recommendations), use_container_width=True)
        
    with tab3:
        st.markdown(f"##### Kategori Audio: **{CLUSTER_NAMES.get(target_cluster, f'Cluster {target_cluster}')}**")
        populasi = len(df_songs[df_songs['cluster_id'] == target_cluster])
        st.write(f"Jumlah lagu di dalam cluster ini: **{populasi:,} lagu**")
        
        avg_series = df_songs[df_songs['cluster_id'] == target_cluster][AUDIO_FEATURES].mean()
        avg_df = pd.DataFrame(avg_series, columns=['Nilai Rata-Rata'])
        st.dataframe(avg_df.T, use_container_width=True)