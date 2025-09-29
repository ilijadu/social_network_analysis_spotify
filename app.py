import networkx as nx
import pandas as pd
import ast
from st_link_analysis import st_link_analysis, NodeStyle, EdgeStyle

grouped_producers_df = pd.read_csv('all_producers_please_please_please.csv',converters={"track_ids": ast.literal_eval,"producers":ast.literal_eval})
grouped_writers_df = pd.read_csv('all_writers_please_please_please.csv',converters={"track_ids": ast.literal_eval,"writers":ast.literal_eval})
added_tracks = pd.read_csv('all_tracks_please_please_please.csv')

G = nx.Graph()

for index, row in added_tracks.iterrows():
    G.add_node(row['id'], label=row['Track Name'], title=row['Track Name'], artist=row['Artist Name'],
               popularity=row['Popularity'],producers = row["Producers"], writers = row['Writers'], topic = row['topic'])

for index, row in grouped_writers_df.iterrows():
    G.add_edge(row['track_ids'][0], row['track_ids'][1], relation='writer', label='writer',
               weight=2 * int(row['count']), type='writer', color='blue', count=row['count'], collaborators = row['writers'] )

for _, row in grouped_producers_df.iterrows():
    if G.has_edge(row['track_ids'][0], row['track_ids'][1]):
        G[row['track_ids'][0]][row['track_ids'][1]]['relation'] = 'writer & producer'
        G[row['track_ids'][0]][row['track_ids'][1]]['label'] = 'writer & producer'
        G[row['track_ids'][0]][row['track_ids'][1]]['type'] = 'writer & producer'
        G[row['track_ids'][0]][row['track_ids'][1]]['color'] = 'green'
        G[row['track_ids'][0]][row['track_ids'][1]]['collaborators'] = list(set(G[row['track_ids'][0]][row['track_ids'][1]]['collaborators'] + row['producers']))
    else:
        G.add_edge(row['track_ids'][0], row['track_ids'][1], relation='producer', label='producer',
                   weight=2 * int(row['count']), type='producer', color='brown', count=row['count'], collaborators = row['producers'])

elements ={'nodes':[], 'edges':[]}

for node in G.nodes():
    if len(node)>1:
        data = {'id':node,"label":str(int(G.nodes[node]['topic'])+1),'title':G.nodes[node]['title'],'artist':G.nodes[node]['artist'],'popularity':G.nodes[node]['popularity'],'topic':str(int(G.nodes[node]['topic'])+1)}
        elements['nodes'].append({"data": data})

for u,v in G.edges():
    data = {'id':str(u)+"-"+str(v), 'label': G[u][v]['label'], 'weight':G[u][v]['count'], 'collaborators':G[u][v]['collaborators'], 'source':u, 'target':v }
    elements['edges'].append({"data":data})
hex_colors = [
    "#FF5733", "#33FF57", "#3357FF", "#FF33A6", "#33FFF6",
    "#FFBD33", "#C70039", "#900C3F", "#DAF7A6", "#581845",
    "#FFC300", "#FF5733", "#C70039", "#900C3F", "#DAF7A6",
    "#581845", "#3498DB", "#2ECC71"
]
#node_styles = [
#     NodeStyle(label='1', color='#F6511D', caption='title'),
#     NodeStyle(label='2', color='#FFB400', caption='title'),
#     NodeStyle(label='3', color='#00A6ED', caption='title'),
#     NodeStyle(label='4', color='#7FB800', caption='title'),
#     NodeStyle(label='5', color='#0D2C54', caption='title'),
# ]
node_styles = []
for i,c in enumerate(hex_colors):
    node_styles.append(NodeStyle(label=str(int(i)+1), color=c, caption='title'))

edge_style = [EdgeStyle( label = 'writer & producer', labeled=True,color = '#91cf60'),
              EdgeStyle( label = 'writer', labeled=True,color = '#fc8d59'),
              EdgeStyle( label = 'producer', labeled=True,color = '#4575b4')]

layout = {"name": "fcose", "animate": "end", "nodeDimensionsIncludeLabels": False}
st_link_analysis(elements, layout, node_styles, edge_style)

