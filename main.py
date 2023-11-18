# %% [markdown]
# In this project we use [NetworkX](https://nbviewer.org/github/massimo-nocentini/APAD-course/blob/master/ipynbs/networkx.ipynb) to show the graphs.
# 

# %%
import sys
sys.stdout = open('output/output.txt','wt')
print('Initiating...\n')

file_dir = ''

# %%
import networkx as nx
import pandas as pd
from collections import defaultdict
import csv

# Define a dictionary to hold and define all publications.
graphs = {
   'out-dblp_article': None,
   'out-dblp_book': None,
   'out-dblp_incollection': None,
   'out-dblp_inproceedings': None,
   'out-dblp_mastersthesis': None,
   'out-dblp_phdthesis': None,
   'out-dblp_proceedings': None,
}

# Define attributes for each type of publication to generate the venue.
publication_attr = {
   'out-dblp_article': ['year', 'journal'],
   'out-dblp_book': ['year', 'title'],
   'out-dblp_incollection': ['year', 'booktitle'],
   'out-dblp_inproceedings': ['year', 'booktitle'],
   'out-dblp_mastersthesis': ['year', 'title'],
   'out-dblp_phdthesis': ['year', 'title'],
   'out-dblp_proceedings': ['year', 'title'],
}

# Define the list of years.
years = [1960,1970,1980,1990,2000,2010,2020,2023]

# Define a dictionary to hold the publications.
pub_dict = {}

# %%
# Function to create a graph from a CSV file.
# It uses the separator || to split the publication attributes.
def get_graph(filename):
   B = nx.Graph()

   with open(f'{file_dir}DATA/{filename}.csv') as file:
      header = file.readline().replace('\n', '').split(';')
      authorIndex = header.index('author')
      csv_reader = csv.reader(file, delimiter=';', quotechar='"')
      
      for row in csv_reader:
         authors = [author.strip() for author in str(row[authorIndex]).split('|')]

         attributes = [row[header.index(attr)] if attr != '-' and str(row[header.index(attr)]) not in ('-', None, 'nan') else '' for attr in publication_attr[filename]]
         # Add row only if there is at least the year.
         if attributes[0] != '':
            if '|' in str(attributes[0]):
               attributes[0] = str(attributes[0]).split('|')[0]
            attributes[0] = int(attributes[0])
            
            publication = ' || '.join([str(attr) for attr in attributes])

            pub_dict[row[0]] = publication
            # Add nodes and edges to the graph for every row.
            B.add_nodes_from(authors, bipartite=0)
            B.add_nodes_from([row[0]], bipartite=1)
            B.add_edges_from([(author, row[0]) for author in authors])

   # Define the position of the nodes in the graph in case of visualization.
   pos = nx.bipartite_layout(B, list({n for n, d in B.nodes(data=True) if d["bipartite"] == 0}))
   return B, pos

# Returns the list of publications up to a given year (included).
def get_publications_up_to_year(publications, year):
   filtered_publications = []
   for publication in publications:
      pub_year = pub_dict[publication].split(' || ')[0]
      if pub_year.isdigit() and int(pub_year) <= year:
         filtered_publications.append(publication)
   return filtered_publications

# Returns the list of publications after a given year (excluded).
def get_publications_after_year(publications, year):
   filtered_publications = []
   for publication in publications:
      pub_year = pub_dict[publication].split(' || ')[0]
      if pub_year.isdigit() and int(pub_year) > year:
         filtered_publications.append(publication)
   return filtered_publications

# Returns the venue with the most publications and the number of publications.
def get_venue_with_more_publications(publications, year):
   publications = get_publications_up_to_year(publications, year)
   venues = [pub_dict[publication].split(' || ')[-1].strip() for publication in publications if publication.split(' || ')[-1].strip() != '']
   if not venues:
      return None
   # Create a series to count the number of publications for each venue.
   venues_series = pd.Series(venues)
   return venues_series.value_counts().idxmax(), venues_series.value_counts()[0]

# BFS visit of a graph
# Returns the nodes divided by layers and the eccentricity of the source (max distance from source)
def bfs_visit(graph, source):
    to_be_visited = {}  # dictionary to store for each node the distance from source
    layers = defaultdict(list)  # each entry contains a list of nodes at distance key

    # initialize dictionaries with source values
    c_distance = 0
    queue = [source]
    to_be_visited[source] = c_distance
    layers[c_distance] = [source]
    # classic BFS implementation
    while len(queue) > 0:
        node = queue.pop(0)
        c_distance = to_be_visited[node]
        for neighbour in list(graph.adj[node]):
            if neighbour not in to_be_visited:
                queue.append(neighbour)
                to_be_visited[neighbour] = c_distance + 1
                layers[c_distance + 1].append(neighbour)

    ecc = c_distance
    return layers, ecc

# Returns the diameter of a graph, using the optimized BFS visit
def get_graph_diameter(graph):
    if len(graph.nodes) == 1:
       return 0
    # find the highest degree node in the graph
    max_connections = 0
    source = None
    for node in graph.nodes:
        if len(graph.adj[node]) > max_connections:
            max_connections = len(graph.adj[node])
            source = node

    layers, ecc = bfs_visit(graph, source)
    i = ecc
    l_bound = ecc
    u_bound = 2 * ecc
    while u_bound > l_bound:
        # compute max_ecc_i(source)
        i_nodes = layers[i]
        max_ecc_i = 0
        for node in i_nodes:
            ecc_i = max(nx.single_source_shortest_path_length(graph, node).values())  # eccentricity
            if ecc_i > max_ecc_i:
                max_ecc_i = ecc_i

        if max(l_bound, max_ecc_i) > 2 * (i - 1):   # lower bound > upper bound
            return max(l_bound, max_ecc_i)
        else:
            l_bound = max(l_bound, max_ecc_i)
            u_bound = 2 * (i - 1)

        i -= 1

    return l_bound

# Returns the author with the most collaborations and the number of collaborations.
def get_author_with_more_collaborations(graph, authors, year):
   max_edges = 0
   selected_node = None

   for author in authors:
      author_publications = get_publications_up_to_year([edge[1] for edge in graph.edges(author)], year)
      multi_author_publications = [pub for pub in author_publications if len(graph[pub]) > 1]

      if len(multi_author_publications) > max_edges:
         max_edges = len(multi_author_publications)
         selected_node = author

   return selected_node, max_edges

#QUESTION 1: Which is the venue having more publications?
def question1(publications):
  print("\nWhich is the venue having more publications by year?")
  for year in years:
    print(f'{year}: {get_venue_with_more_publications(publications, year)}')

#QUESTION 2: Compute exactly the diameter of G
def question2(graph, publications):
   print("\nWhich is the exact diameter of the biggest CC by year?")

   for year in years:
      most_connected_graph = graph.copy()
      most_connected_graph.remove_nodes_from(get_publications_after_year(publications, year))
      GCCs = sorted(nx.connected_components(most_connected_graph), key=len, reverse=True)
      most_connected_graph = most_connected_graph.subgraph(GCCs[0])

      print(f'{year}: ({most_connected_graph.number_of_nodes()}, {get_graph_diameter(most_connected_graph)})')

#QUESTION 3: Who is the author who had the largest number of collaborations? If author A and B collaborated twice, this count 2
def question3(graph, authors):
  print("\nWho is the author who had the largest number of collaborations by year?")
  for year in years:
    print(f'{year}: {get_author_with_more_collaborations(graph, authors, year)}')


# %%
# First three questions for each file
for filename in graphs.keys():
   print(f'\nFile: {filename}')
   graphs[filename], pos = get_graph(filename)
   authors = {n for n, d in graphs[filename].nodes(data=True) if d["bipartite"] == 0}
   publications = set(graphs[filename]) - authors

   print(f'Authors: {len(authors)}')
   print(f'Publications: {len(publications)}')

   question1(publications)
   question2(graphs[filename], publications)
   question3(graphs[filename], authors)

   print('\n__________________________________________________\n')

# %%
# QUESTION 4: Merge all graphs and answer the same questions.
print(f'\nMerged graphs')

for filename in graphs.keys():
   graphs[filename], pos = get_graph(filename)

# merge all graphs
merged_graph = nx.compose_all(graphs.values())
authors = {n for n, d in merged_graph.nodes(data=True) if d["bipartite"] == 0}
publications = set(merged_graph) - authors

question1(publications)
question2(merged_graph, publications)
question3(merged_graph, authors)


# %%

print("\nWhich is the pair of authors who collaborated the most between themselves?")
# Which is the pair of authors who collaborated the most between themselves?
authors_graph = nx.Graph()
for publication in publications:
   authors = [edge[1] for edge in merged_graph.edges(publication)]
   for i in range(len(authors)):
      for j in range(i + 1, len(authors)):
         if authors_graph.has_edge(authors[i], authors[j]):
            authors_graph[authors[i]][authors[j]]['weight'] += 1
         else:
            authors_graph.add_edge(authors[i], authors[j], weight=1)
   
# Get edges sorted by weight
sorted_edges = sorted(authors_graph.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
if len(sorted_edges) > 0:
   print(sorted_edges[0])
else:
   print('None')


