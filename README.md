# social_network_analysis_spotify

This repository contains the code and dataset used for analyzing the most popular Spotify songs (2010–2023). The project combines network analysis, clustering, and NLP to study how songs, artists, writers, and producers are connected through collaboration and lyrical themes.

Features: 

 * Dataset Creation: Gathers top songs (2010–2023) using Spotify API.
 * Lyrics & Metadata: Retrieves lyrics, writers, and producers via Genius API.
 * Network Graphs: Builds graphs with songs as nodes, edges for shared collaborators.
 * Analysis Methods:
   * Centrality measures (degree, betweenness, closeness, eigenvector)
   * Louvain community detection
   * Homophily & Index of Qualitative Variation (IQV)
   * K-Means clustering on audio features
   * Topic modeling on lyrics
 * Visualization: Interactive and static graphs using NetworkX, Matplotlib, and Streamlit.

Requirements
Install the dependencies and main libraries:

spotipy – Spotify Web API
lyricsgenius – Lyrics & metadata
nltk, scikit-learn – NLP & clustering
networkx, igraph, community_louvain – Network analysis
matplotlib, seaborn, streamlit – Visualization


Setup
1. Create a Spotify developer account and get credentials.
2. Create a Genius API account and generate a token.
3. Store them in your notebook/environment


Example Results:
Networks of songs connected by shared producers/writers.
Clusters of songs grouped by audio features.
Topic modeling showing dominant lyrical themes.
Ego networks for individual tracks (e.g., Sabrina Carpenter’s Please Please Please).
