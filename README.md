
# Spotify Track Clustering & Recommender

An interactive web application that groups tracks by audio features using K-Means clustering and delivers song recommendations based on Euclidean distance similarity. Built with Python and Streamlit.

## Features

- **Song-Based Search**: Select a song from the dataset to find tracks with similar audio characteristics.
- **Custom Audio Mood**: Adjust individual audio parameters (danceability, energy, valence, tempo, etc.) to discover matching tracks.
- **Radar Chart Comparison**: Visual spider plot comparing the target audio profile against recommendation averages using Plotly.
- **Embedded Player**: Direct song preview integration via the Spotify Web Player iframe.

## Dataset

This project uses the [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) from Hugging Face.

Model training relies on 9 key acoustic features: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, and `tempo`.

## Repository Structure

```text
.
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── scaler.pkl              # Fitted StandardScaler artifact
├── model_kmeans.pkl        # Trained K-Means model artifact
└── spotify_cleaned.parquet # Processed dataset with cluster labels

```

## Local Setup

### 1. Prerequisites

Ensure Python 3.9+ is installed.

### 2. Installation

Install the required packages:

```bash
pip install -r requirements.txt

```

### 3. Execution

Run the Streamlit application:

```bash
streamlit run app.py

```

```

```