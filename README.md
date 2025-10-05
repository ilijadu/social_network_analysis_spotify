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


The main code is stored in the master_thesis_code.ipynb file.

The additional code for artist clustering is located in the Artist_Clustering.ipynb file.
This notebook focuses on grouping artists based on their aggregated audio features (danceability, energy, valence, tempo, etc.) obtained from the Spotify Web API. It applies K-Means clustering to identify stylistic similarities between artists and uses PCA for dimensionality reduction and visualization. Additionally, it calculates cosine and Euclidean distances, highlighting artists with comparable musical styles.
The dataset that has been used is all_tracks_please_please_please.csv. It has been collected by combining the top2023.csv, top2022.csv, top 2021.csv and top2020.csv datasets. They have been collected using the Spotify's WEB API and spotipy Python library to collect the data from the most listened songs for each year.
