# removing excess info from the track name
import ast
import pandas as pd
import igraph as ig
import community as community_louvain

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import lyricsgenius
from bs4 import BeautifulSoup
from selenium import webdriver
import networkx as nx
import seaborn as sns
from string import punctuation
import nltk
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sn
from random import seed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

import re
from sklearn.metrics import silhouette_score
from scipy import stats
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from wordcloud import WordCloud

radius_columns = ['node radius',
                  'danceability',
                  'energy',
                  'key',
                  'loudness',
                  'mode',
                  'speechiness',
                  'acousticness',
                  'instrumentalness',
                  'liveness',
                  'valence',
                  'tempo']
def remove_excess_info (x):
    error_words_with_feature=["(with","(feat."]
    error_words_with_feature_2=["[with","[feat."]
    error_words_with_hyphen =["remaster","bonus track","recorded at"," feat. ","- edit", "- video mix","remix"]

    for word in error_words_with_feature_2:
        if word in x.lower():
            x = x.rsplit(" [")[0]
            break
    for word in error_words_with_feature:
        if word in x.lower():
            x = x.rsplit(" (")[0]
            break
    for word in error_words_with_hyphen:
        if word in x.lower():
            x = x.rsplit(" -")[0]

    return x


# Define a function to retrieve the lyrics for a track
def get_lyrics(track_name, artist_name, genius):
    try:
        if "feat" in track_name:
            track_name = track_name.split("feat")[0]
            track_name = track_name[:-2]
        elif "with" in track_name:
            track_name = track_name.split("with")[0]
            track_name = track_name[:-2]
        song = genius.search_song(track_name, artist_name)

        public_song=lyricsgenius.PublicAPI().song(song.id)["song"]
        writers=public_song["writer_artists"]
        producers=public_song["producer_artists"]

        song_writers=[]
        for writer in writers:
            song_writers.append(writer["name"])

        song_producers=[]
        for producer in producers:
            song_producers.append(producer["name"])

        lyrics = song.lyrics
        return {"lyrics":lyrics,"writers":song_writers,"producers":song_producers,"genius_id":public_song["id"]}
    except Exception as error:
        print("greska: ",error)
        return {"lyrics":'No lyric found',"writers":'No writer found',"producers":'No producer found',"genius_id":'No id found'}

def clean_lyrics(artist_name, track_name, lyric, song_id):
    track_name = remove_excess_info(track_name)

    #lyric that is retrieved contains some unnecessary information that has to be removed

    if "lyrics[" in lyric:
        lyric = lyric.split("]", maxsplit=1)
    else:
        lyric = lyric.split(track_name + " lyrics", maxsplit=1)

    #if the lyric is retrieved, it would be split into two parts so that the second part represents the real lyric, and the excess info;
    # if it's not, the retrieved text is either "No lyric found" or it is some unuseful text, and it wasn't split in the previous code.
    # Because of that, web scrapping is needed to retrieve the lyric

    if len(lyric) == 2:
        lyric = lyric[1]
    else:
        return get_lyrics_selenium(track_name,artist_name)
    lyric = lyric.replace("’", "'").replace("embed", "").replace("\'", "'")

    result = re.sub('[[^]]*W+[^]]*]', '', lyric)

    if result[len(result) - 1].isnumeric():
        result = result.replace(result[len(result) - 1], "")
    return remove_rows(result)

def get_soup_selenium(url: str) -> BeautifulSoup:
    driver = webdriver.Chrome()
    driver.get(url)
    return BeautifulSoup(driver.page_source, 'html.parser')

def remove_rows(text):
    text = text.replace("\n", " ")
    return text

def replace_signs(x):
    signs = punctuation + " " + "’"
    x = [c for c in x.lower() if not c in signs]
    x = "".join(x)
    return x.strip()


def remove_the(x):
    if "The" in x[0:3]:
        return x.replace("The", "").strip()
    if "the" in x[0:3]:
        return x.replace("the", "").strip()
    return x


def get_lyrics_selenium(track_name, artist_name):
    try:
        track_name = remove_excess_info(track_name)

        track_name_lc = [w.strip() for w in track_name.lower().split("(") if (not "feat" in w)]

        if len(track_name_lc) >= 1:
            track_name_lc = "".join(track_name_lc)
        elif len(track_name_lc) == 0:
            track_name_lc = track_name.lower().split(" feat.", maxsplit=1)[0]

        artist_name = remove_the(artist_name)

        url = "https://www.azlyrics.com/lyrics/" + replace_signs(artist_name) + "/" + replace_signs(
            track_name_lc) + ".html"
        print(url)
        soup = get_soup_selenium(url)

        excess = "<!-- Usage of azlyrics.com content by any third-party lyrics provider is prohibited by our licensing agreement. Sorry about that. -->"
        text = str(
            soup.find("div", {"class": "col-xs-12 col-lg-8 text-center"}).find("div", {"class": None, "style": None}))
        text = text.replace("<br/>", "").replace(excess, "").replace("<div>", "").replace("</div>", "").replace("’",
                                                                                                                "'").strip()

        return remove_rows(text.lower())
    except:
        return 'No lyric found (selenium)'

def producers(row,shared_producers_df):

    shared_contributors = []
    match = shared_producers_df[shared_producers_df['track_ids']==row]
    for index,rows in match.iterrows():
        shared_contributors.append(rows['producer'])
    return shared_contributors

def writers(row,shared_writers_df):

    shared_contributors = []
    match = shared_writers_df[shared_writers_df['track_ids']==row]
    for index,rows in match.iterrows():
        shared_contributors.append(rows['writer'])
    return shared_contributors

def group_track_by_writer_producer(df, writer_csv_name, producer_csv_name, added_tracks_name ,track_name):
    track_name = track_name.replace(" ","_").lower()
    shared_writers = []
    shared_producers = []
    added_tracks=pd.DataFrame(columns=df.columns)

    if len(df)>0:
        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            track_id_1 = row['id']
            # Find shared writers
            for writer in row['Writers']:
                writer=writer.replace("'","").strip()
                for other_index, other_row in df.iterrows():
                    track_id_2 = other_row['id']
                    new_writer=[tuple(sorted([track_id_1, track_id_2])), writer, tuple(sorted([row['Track Name'], other_row['Track Name']]))]
                    if track_id_1 != track_id_2 and writer in other_row['Writers'] and new_writer not in shared_writers:
                        shared_writers.append(new_writer)
                        if track_id_1 not in added_tracks['id'].unique():
                            added_tracks.loc[len(added_tracks)]=row
                        if track_id_2 not in added_tracks['id'].unique():
                            added_tracks.loc[len(added_tracks)]=other_row



            # Find shared producers
            for producer in list(row['Producers']):
                producer=producer.replace("'","").strip()
                for other_index, other_row in df.iterrows():
                    track_id_2 = other_row['id']
                    new_producer=[tuple(sorted([track_id_1, track_id_2])), producer, tuple(sorted([row['Track Name'], other_row['Track Name']]))]
                    if track_id_1 != track_id_2 and producer in other_row['Producers'] and new_producer not in shared_producers:
                        shared_producers.append(new_producer)
                        if track_id_1 not in added_tracks['id'].unique():
                            added_tracks.loc[len(added_tracks)]=row
                        if track_id_2 not in added_tracks['id'].unique():
                            added_tracks.loc[len(added_tracks)]=other_row

    shared_writers_df = pd.DataFrame(shared_writers, columns=['track_ids', 'writer', 'track_names'])
    shared_producers_df = pd.DataFrame(shared_producers,columns=['track_ids', 'producer', 'track_names'])

    # Remove duplicates
    shared_writers_df = shared_writers_df.drop_duplicates()
    shared_producers_df = shared_producers_df.drop_duplicates()

    grouped_producers_df = shared_producers_df.groupby('track_ids').size().reset_index(name='count')
    grouped_producers_df['producers'] = grouped_producers_df.apply(lambda x: producers(x['track_ids'],shared_producers_df),axis = 1)

    grouped_writers_df = shared_writers_df.groupby('track_ids').size().reset_index(name='count')
    grouped_writers_df['writers']=grouped_writers_df.apply(lambda x: writers(x['track_ids'],shared_writers_df), axis = 1)

    writers_name = writer_csv_name+"_"+str(track_name)+".csv"
    producers_name = producer_csv_name+"_"+str(track_name)+".csv"
    all_name = added_tracks_name+"_"+str(track_name)+".csv"

    # Save the datasets to CSV files
    added_tracks.to_csv(all_name,index=False)
    grouped_writers_df.to_csv(writers_name, index=False)
    grouped_producers_df.to_csv(producers_name, index=False)

    return grouped_writers_df, grouped_producers_df, added_tracks

def calculate_euclidian_distance(all_nodes, node, one_song=False, least_similar=False, radio_nodes=False):
    if not radio_nodes:
        # Drop the current node from all_nodes if it exists
        if 'id' in all_nodes.columns and 'id' in node.columns:
            all_nodes = all_nodes[all_nodes['id'] != node.loc[0, 'id']].reset_index(drop=True)

    # Normalize the features
    scaler = StandardScaler()
    normed = scaler.fit_transform(all_nodes[radius_columns[1:]])  # Exclude 'node radius' from normalization

    # Create a normalized DataFrame
    df_normalized = pd.DataFrame(normed, columns=radius_columns[1:])  # Match with original columns except 'node radius'
    df_normalized['id'] = all_nodes['id'].values if not radio_nodes else [ i for i in range(1, len(all_nodes)+1)]

    # Normalize the node to compare
    new_song_df = pd.DataFrame([node[radius_columns[1:]].values[0]], columns=radius_columns[1:])
    new_song_normalized = scaler.transform(new_song_df)

    # Calculate Euclidean distance between the new song and each row in the dataset
    distances = df_normalized.apply(lambda row: euclidean(row[radius_columns[1:]].values, new_song_normalized[0]), axis=1)

    # Add distances to the DataFrame
    df_normalized['distance'] = distances

    # Find the most or least similar row
    if least_similar:
        most_similar_row_index = df_normalized['distance'].idxmax()
    else:
        most_similar_row_index = df_normalized['distance'].idxmin()

    most_similar_row = all_nodes.loc[most_similar_row_index]

    if one_song:
        return most_similar_row
    return df_normalized[['id', 'distance']]

# Example usage:
# most_similar = calculate_euclidian_distance(community_added_tracks, new_tracks, most_similar=True)

def create_graph(added_tracks, grouped_writers_df, grouped_producers_df, song_id):

    G = nx.Graph()

    for index, row in added_tracks.iterrows():
        color = '#41b6c4' if row['id'] == song_id else '#e6550d'
        G.add_node(row['id'], label = row['Track Name'], title=row['Track Name'], artist=row['Artist Name'], popularity=row['Popularity'], year='2023',topic=row['topic'], color = color)

    for index, row in grouped_writers_df.iterrows():
        G.add_edge(row['track_ids'][0], row['track_ids'][1], relation = 'writer',label = 'writer',weight=2*int(row['count']), type='writer', color = 'blue',count=row['count'], collaborators = row['writers'])

    for _, row in grouped_producers_df.iterrows():
        if G.has_edge(row['track_ids'][0], row['track_ids'][1]):
            G[row['track_ids'][0]][row['track_ids'][1]]['relation'] = 'writer & producer'
            G[row['track_ids'][0]][row['track_ids'][1]]['label'] = 'writer & producer'
            G[row['track_ids'][0]][row['track_ids'][1]]['type'] = 'writer & producer'
            G[row['track_ids'][0]][row['track_ids'][1]]['color'] = 'green'
            G[row['track_ids'][0]][row['track_ids'][1]]['count'] = 2*(int(row['count']) + G[row['track_ids'][0]][row['track_ids'][1]]['count'])
            
            G[row['track_ids'][0]][row['track_ids'][1]]['collaborators'] = list(set(G[row['track_ids'][0]][row['track_ids'][1]]['collaborators'] + row['producers']))
        else:
            G.add_edge(row['track_ids'][0], row['track_ids'][1], relation = 'producer',label = 'producer',weight=2*int(row['count']), type='producer', color = 'brown',count=row['count'], collaborators = row['producers'])

    return G

def plot_graph(G,topic=False, layout = None):
    rgb_colors = ['#F6511D','#FFB400','#00A6ED','#7FB800','#0D2C54']
    plt.figure(figsize=(20, 12))
    edges = G.edges()
    nodes = G.nodes()
    edge_colors = [G[u][v]['color'] for u,v in edges]
    if topic:
        node_colors = [rgb_colors[G.nodes[node]['topic']] for node in nodes]
    else:
        node_colors = [G.nodes[node]['color'] for node in nodes]
    size = [float(G.nodes[u]['popularity'])*10 for u in nodes]
    legend_labels = ['Writer', 'Producer', 'Writer & Producer']
    legend_colors = ['blue', 'brown', 'green']
    legend_markers = [plt.Line2D([0], [0], marker='_', color=color, markerfacecolor=color, markersize=20) for color in
                      legend_colors]
    plt.legend(legend_markers, legend_labels, loc='lower right', fontsize='large', frameon=True, borderpad=1,
               borderaxespad=1)
    if layout:
        nx.draw(G, labels=nx.get_node_attributes(G, 'title'), width=2, edge_color=edge_colors, alpha=0.5,
            with_labels=False, node_color=node_colors, node_size=size, font_weight='bold',pos=layout)
    else:
        nx.draw(G, labels=nx.get_node_attributes(G, 'title'), width=2, edge_color=edge_colors, alpha=0.5,
                with_labels=False, node_color=node_colors, node_size=size, font_weight='bold')

def centrality_measuers(df,G):

    sub_tracks=pd.DataFrame(df[['id','Track Name','Artist Name', 'Producers', 'Writers', 'Popularity']])

    song_degrees = dict(G.degree())
    sub_tracks['Degree'] = sub_tracks['id'].apply(lambda x: song_degrees[x])

    song_closeness = dict(nx.closeness_centrality(G))
    sub_tracks['Closeness'] = sub_tracks['id'].apply(lambda x: song_closeness[x])

    song_betweenness = dict(nx.betweenness_centrality(G))
    sub_tracks['Betweenness'] = sub_tracks['id'].apply(lambda x: song_betweenness[x])

    song_eigen = dict(nx.eigenvector_centrality(G))
    sub_tracks['Eigen'] = sub_tracks['id'].apply(lambda x: song_eigen[x])

    return sub_tracks

def furthest_nodes_paths(G,start_node):
    shortest_paths = nx.shortest_path_length(G, source=start_node)
    furthest_nodes = [k for k,v in shortest_paths.items() if max(shortest_paths.values()) == v]
    return furthest_nodes, max(shortest_paths.values())

def noramlize_feature(x):
    if sum(x)==0:
        return x
    else:
        return (x - min(x))/(max(x) - min(x))

def print_shortest_path(G,start_node,end_node):
    path_to_furthest_node = nx.shortest_path(G, source=start_node, target=end_node)

    print(f"The furthest distance from {G.nodes[start_node]['title']} is: {G.nodes[end_node]['title']}")
    print()
    for i, node in enumerate(path_to_furthest_node[:-1]):
        print(f"{G.nodes[node]['title']} -> {G[node][path_to_furthest_node[i+1]]['collaborators']} -> {G.nodes[path_to_furthest_node[i+1]]['title']}")
    print("_______________________________________________________")

def plot_graph_with_shortest_path(G, start_node, end_node):
    # Calculate the shortest path between the start and end nodes
    shortest_path = nx.shortest_path(G, source=start_node, target=end_node)

    # Get edges in the shortest path
    path_edges = list(zip(shortest_path, shortest_path[1:]))

    # Position nodes using a spring layout
    pos = nx.spring_layout(G)
    plt.figure(figsize=(20, 12))

    # Draw all nodes without labels
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, alpha=0.5)

    # Draw all edges without labels
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)

    # Highlight the shortest path nodes
    nx.draw_networkx_nodes(G, pos, nodelist=shortest_path, node_color='#e6550d', node_size=600)

    # Highlight the shortest path edges
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='#e6550d', width=2, alpha=0.5)

    # Draw labels only for nodes in the shortest path
    nx.draw_networkx_labels(G, pos, labels={node: G.nodes[node]['title'] for node in shortest_path}, font_size=10)

    # Draw edge labels only for edges in the shortest path with adjusted label position
    edge_labels = {(u, v): f"{G[u][v]['collaborators']}" for u, v in path_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', label_pos=0.3, rotate=False)  # Adjust label_pos

    plt.title(f'Shortest Path from {G.nodes[start_node]['title']} to {G.nodes[end_node]['title']}', fontsize=15)
    plt.show()





