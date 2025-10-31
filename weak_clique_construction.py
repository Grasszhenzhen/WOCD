import numpy as np
from itertools import combinations


def calculate_extend_centrality(G):
    centrality = {}
    edges_num_of_node = {node: 0 for node in G.nodes}
    total_connections = {node: 0 for node in G.nodes}

    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        edges_num_of_node[node] += len(neighbors)

        for nei1, nei2 in combinations(neighbors, 2):
            if G.has_edge(nei1, nei2):
                total_connections[node] += 1

        total_connections[node] += edges_num_of_node[node]

    for node in G.nodes:
        centrality[node] = (total_connections[node]) / (len(list(G.neighbors(node))) + 1)

    return sorted(centrality.items(), key=lambda item: item[1], reverse=True)


def calculate_similarity(G, max_id, neighbors):
    similarity_intersect = []
    neighbors_set = set(neighbors)

    for neighbor in neighbors:
        nei = set(G.neighbors(neighbor))
        intersect = neighbors_set.intersection(nei)
        intersect_num = len(intersect)

        simi = (intersect_num + 2) / np.sqrt(len(nei) * len(neighbors))

        similarity_intersect.append((simi, neighbor, list(intersect)))

    return similarity_intersect


def get_subgraph(G, centrality):
    centrality_copy = centrality[:]
    is_visited = {node: False for node in G.nodes()}
    density_subgraphs = []

    while centrality_copy:
        max_item = max((item for item in centrality_copy if not is_visited[item[0]]), key=lambda item: item[1], default=None)
        if not max_item:
            break
        max_id = max_item[0]
        # neighbors = [n for n in G.neighbors(max_id) if not is_visited[n]]
        neighbors = [n for n in G.neighbors(max_id)]
        if not neighbors:
            is_visited[max_id] = True
            centrality_copy = [item for item in centrality_copy if not is_visited[item[0]]]
            continue

        similarities = calculate_similarity(G, max_id, neighbors)
        # similarities = [s for s in similarities if not is_visited[s[1]]]
        similarities.sort(reverse=True, key=lambda x: x[0])

        if similarities:
            max_similarity = similarities[0]
            most_similar_neighbor = max_similarity[1]
            subgraph_nodes = max_similarity[2]
            subgraph_nodes.append(most_similar_neighbor)
            subgraph_nodes.append(max_id)

            used_pair_nodes = [most_similar_neighbor, max_id]

            used_pair_nodes = list(set(used_pair_nodes))

            # for node in subgraph_nodes:
            for node in used_pair_nodes:
                is_visited[node] = True

            subgraph_set = set(subgraph_nodes)
            # if len(subgraph_nodes) > 2 and not any(set(sg) == subgraph_set for sg in density_subgraphs):
            #     density_subgraphs.append(subgraph_nodes)
            should_add = True

            for existing_subgraph in density_subgraphs:
                if subgraph_set.issubset(set(existing_subgraph)):
                    should_add = False
                    break

            if should_add and len(subgraph_set) > 2:
                density_subgraphs.append(subgraph_nodes)

        centrality_copy = [item for item in centrality_copy if not is_visited[item[0]]]

    return density_subgraphs

