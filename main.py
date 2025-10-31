import time
import copy
import argparse
import random
import torch
import math
import networkx as nx
from sklearn.preprocessing import normalize
from scipy.sparse import coo_matrix
from load_dataset import *
# from dnmf.DNMF import DNMF
from parse import *
from torch_geometric.data import Data
from weak_clique_construction import *
from sklearn.metrics import f1_score as F1

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    total_start_time = time.time()

    dataset_path = 'data/facebook_ego/fb_1684.npz'
    # dataset_path = 'data/mag_ego/mag_med.npz'

    loader = load_dataset(dataset_path)

    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser_add_main_args(parser)
    args = parser.parse_args()
    change_input_args(args, dataset_path)

    seed_everything(args.seed)

    A, X, Z_gt = loader['A'], loader['X'], loader['Z']
    N, K = Z_gt.shape

    A_coo = coo_matrix(A)
    rows = A_coo.row
    cols = A_coo.col
    edge_index = torch.tensor([rows, cols], dtype=torch.long)

    device = args.device

    x_norm = normalize(X)  # node features
    # x_norm = normalize(A)  # adjacency matrix
    # x_norm = sp.hstack([normalize(X), normalize(A)])  # concatenate A and X
    x_norm = to_sparse_tensor(x_norm)
    y_tensor = torch.tensor(Z_gt, dtype=torch.float32)
    data = Data(x=x_norm.to_dense(), edge_index=edge_index, y=y_tensor, num_nodes=N)

    extra_column = torch.zeros((N, 1), dtype=y_tensor.dtype, device=y_tensor.device)
    y_extended = torch.cat([y_tensor, extra_column], dim=1)
    no_community_mask = (y_tensor.sum(dim=1) == 0)
    y_extended[no_community_mask, K] = 1
    y_np = y_extended.cpu().numpy()

    # split true-labeled nodes
    num_nodes = data.x.size(0)
    indices = torch.randperm(num_nodes)
    split_idx = int(num_nodes * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    test_indices = indices

    sample_size = int(train_indices.size(0) * args.sample_ratio * 1.25)
    num_samples_per_community = math.ceil(sample_size / (K+1))
    min_samples = max(1, int(num_samples_per_community * args.sample_ratio))
    unique_indices = set()
    train_indices_set = set(train_indices.tolist())

    for community_id in range(K+1):
        community_indices = torch.nonzero(y_extended[:, community_id]).squeeze(1)

        community_train_indices = [idx for idx in community_indices.tolist() if idx in train_indices_set]

        if len(community_train_indices) < num_samples_per_community:
            sampled_indices = torch.tensor(community_train_indices)[
                torch.randperm(len(community_train_indices))[:min(min_samples, len(community_train_indices))]]
        else:
            sampled_indices = torch.tensor(community_train_indices)[
            torch.randperm(len(community_train_indices))[:num_samples_per_community]]

        for idx in sampled_indices.tolist():
            unique_indices.add(idx)

    sampled_node_indices = torch.tensor(list(unique_indices))
    sampled_train_indices = [train_indices.tolist().index(i) for i in sampled_node_indices.tolist()]
    use_nodes_num = len(sampled_node_indices)
    print(f'Total Nodes number: {num_nodes}, Used Nodes number: {use_nodes_num}')

    if args.use_pseudo:
        HybridBlock1 = parse_method(args, K, x_norm.shape[1], device)
        HybridBlock2 = parse_method(args, K, x_norm.shape[1], device)
        opt1 = torch.optim.Adam(HybridBlock1.parameters(), lr=args.lr)
        opt2 = torch.optim.Adam(HybridBlock2.parameters(), lr=args.lr)
    else:
        HybridBlock1 = parse_method(args, K, x_norm.shape[1], device)
        opt1 = torch.optim.Adam(HybridBlock1.parameters(), lr=args.lr)

    # find weak cliques
    G = nx.Graph()
    G_indices = list(range(num_nodes))
    G.add_nodes_from(G_indices)

    for i in range(edge_index.shape[1]):
        edges = [(edge_index[0, i].item(), edge_index[1, i].item())]
        G.add_edges_from(edges)

    start_time = time.time()
    extend_centrality = calculate_extend_centrality(G)
    subgraphs = get_subgraph(G, extend_centrality)

    subgraph_labels_matrix = torch.zeros(len(subgraphs), K, dtype=torch.float)
    sampled_node_set = set(sampled_node_indices.tolist())
    for i, subgraph in enumerate(subgraphs):
        community_count = torch.zeros(K, dtype=torch.float)
        node_count = 0
        total_nodes = len(subgraph)

        for node in subgraph:
            if node in sampled_node_set:
                node_count += 1
                node_labels = data.y[node]
                community_count += node_labels

        if node_count >= args.real_label_threshold * total_nodes:
            subgraph_labels_matrix[i] = community_count

    max_values, _ = torch.max(subgraph_labels_matrix, dim=1, keepdim=True)
    subgraph_labels_matrix = torch.where(subgraph_labels_matrix == max_values, subgraph_labels_matrix, torch.tensor(0.0))
    subgraph_labels_matrix = torch.where(subgraph_labels_matrix > 0, torch.tensor(1.0), torch.tensor(0.0))
    node_community_matrix = torch.zeros(num_nodes, K, dtype=torch.float)

    for subgraph, labels in zip(subgraphs, subgraph_labels_matrix):
        for node in subgraph:
            node_community_matrix[node] += labels

    binary_matrix = torch.zeros_like(node_community_matrix)

    for i in range(node_community_matrix.size(0)):
        vals = node_community_matrix[i]
        total = vals.sum()
        if total <= 0:
            binary_matrix[i].zero_()
            continue

        sorted_vals, sorted_idx = torch.sort(vals, descending=True)
        cum = torch.cumsum(sorted_vals, dim=0)
        thresh = total * args.threshold_ratio

        ge_mask = (cum >= thresh)
        if ge_mask.any():
            k = ge_mask.nonzero(as_tuple=False)[0].item()
            selected = sorted_idx[:k + 1]
            binary_matrix[i].zero_()
            binary_matrix[i, selected] = 1.0
        else:
            binary_matrix[i].zero_()

    node_community_matrix = binary_matrix.to('cpu')
    real_y = data.y.cpu().detach().numpy()
    nmi = overlapping_nmi(node_community_matrix.numpy().copy(), real_y)
    print(f'Org ONMI: {nmi:.3f}')

    end_time = time.time()
    print(f'Pseudo-label Initialization Time: {end_time-start_time:.2f} ')

    max_nmi = 0.0
    max_jaccard = 0.0
    max_f1 = 0.0

    if args.val_best == 'loss':
        best_val_value = float('inf')
    elif args.val_best == 'onmi':
        best_val_value = float('-inf')
    else:
        raise ValueError(f'Unknown val_best option')

    best_epoch = 0
    patience = 500

    pseudo_labels = node_community_matrix[train_indices].to(torch.float32)
    mask = torch.ones(pseudo_labels.size(0), dtype=bool)
    mask[sampled_train_indices] = False
    mask_pseudo_labels = pseudo_labels[mask]

    train_pseudo_labels = mask_pseudo_labels.to(device)

    non_zero_mask = mask_pseudo_labels.sum(dim=1) > 0
    num_non_zero_rows = non_zero_mask.sum().item()
    train_pseudo_labels = train_pseudo_labels[non_zero_mask]
    train_pseudo_labels_narry = mask_pseudo_labels[non_zero_mask].cpu().detach().numpy()
    print(f"Number of non-zero rows: {num_non_zero_rows}")

    graph = data
    graph = graph.to(device)

    for epoch in range(args.max_epochs):

        train_mask = train_indices

        opt1.zero_grad()
        HybridBlock1.train()
        embeddings, train_pred_communities = HybridBlock1(graph.x, graph.edge_index, train_mask)

        train_pred_communities = torch.sigmoid(train_pred_communities)

        sampled_pred = train_pred_communities[sampled_train_indices]
        true_labels = data.y[sampled_node_indices].float()

        train_pred_labels = train_pred_communities[mask]
        train_pred_labels = train_pred_labels[non_zero_mask]

        loss = args.true_label_weight * F.binary_cross_entropy(sampled_pred, true_labels) + args.pseudo_label_weight * F.binary_cross_entropy(train_pred_labels, train_pseudo_labels)
        loss += l2_reg_loss(HybridBlock1, scale=args.weight_decay)
        train_loss = loss
        train_loss.backward()
        opt1.step()

        with torch.no_grad():
            opt1.zero_grad()
            HybridBlock1.eval()

            whether_best = False
            if args.val_best == 'loss':
                val_mask = val_indices
                embeddings, val_pred_communities = HybridBlock1(graph.x, graph.edge_index, val_mask)
                val_pred_communities = torch.sigmoid(val_pred_communities)
                val_true_labels = graph.y[val_mask]
                val_value = F.binary_cross_entropy(val_pred_communities, val_true_labels)
                if val_value <= best_val_value:
                    whether_best = True
            elif args.val_best == 'onmi':
                val_mask = val_indices
                embeddings, val_pred_communities = HybridBlock1(graph.x, graph.edge_index, val_mask)
                val_pred_communities = torch.sigmoid(val_pred_communities)
                Z_pred_val = val_pred_communities.cpu().detach().numpy() > 0.5
                real_y_val = graph.y[val_mask].cpu().detach().numpy()
                val_value = overlapping_nmi(Z_pred_val, real_y_val)
                if val_value >= best_val_value:
                    whether_best = True
            else:
                raise ValueError(f'Unknown val_best option')

            test_mask = test_indices
            embeddings, test_pred_communities = HybridBlock1(graph.x, graph.edge_index, test_mask)
            test_pred_communities = torch.sigmoid(test_pred_communities)
            Z_pred = test_pred_communities.cpu().detach().numpy() > 0.5

            real_y = data.y[test_indices].cpu().detach().numpy()
            nmi = overlapping_nmi(Z_pred, real_y)
            jaccard = symmetric_jaccard(Z_pred, real_y)
            f1 = symmetric_f1(Z_pred, real_y)

            if whether_best:
                best_val_value = val_value
                max_nmi = nmi
                max_jaccard = jaccard
                max_f1 = f1
                best_epoch = epoch
                best_model_state = copy.deepcopy(HybridBlock1.state_dict())
            if epoch % 500 == 0 or epoch == args.max_epochs - 1:
                print(f'Epoch: {epoch:3d} Training loss: {train_loss.item():5.4f} Val loss: {val_value.item():5.4f} ONMI: {nmi:5.3f}  Jaccard:{jaccard:5.3f}  F1:{f1:5.3f}')
                print(f'==========Best Epoch so far: {best_epoch} Best Val loss: {best_val_value.item():5.4f} Best ONMI: {max_nmi:5.3f} Best Jaccard: {max_jaccard:5.3f} Best F1: {max_f1:5.3f}==========')

    if args.use_pseudo:
        middle_nmi = max_nmi
        # print(f'Best Epoch: {best_epoch} Middle Best NMI: {middle_nmi:.3f}')
        HybridBlock1.load_state_dict(best_model_state)
        HybridBlock1.eval()
        with torch.no_grad():
            test_mask = torch.arange(N, dtype=torch.long).to(device)
            embeddings, updated_train_pred_communities = HybridBlock1(graph.x, graph.edge_index, test_mask)
            updated_train_pred_communities = torch.sigmoid(updated_train_pred_communities)

            updated_train_pred_communities = updated_train_pred_communities.cpu().detach().numpy() > 0.5
            real_y = data.y[test_mask].cpu().detach().numpy()
            round1_nmi = overlapping_nmi(updated_train_pred_communities, real_y)
            round1_jaccard = symmetric_jaccard(updated_train_pred_communities, real_y)
            round1_f1 = symmetric_f1(updated_train_pred_communities, real_y)
            print(f'Middle ONMI: {round1_nmi:.3f}, Middle Jaccard: {round1_jaccard:.3f}, Middle F1: {round1_f1:.3f}')

            node_community_matrix = torch.tensor(updated_train_pred_communities.astype(int), dtype=torch.float).to(device)
            pseudo_labels = node_community_matrix[train_indices]
            mask = torch.ones(pseudo_labels.size(0), dtype=bool)
            mask[sampled_train_indices] = False
            mask_pseudo_labels = pseudo_labels[mask]
            train_pseudo_labels = mask_pseudo_labels.to(device)

            non_zero_mask = mask_pseudo_labels.sum(dim=1) > 0
            num_non_zero_rows = non_zero_mask.sum().item()
            train_pseudo_labels = train_pseudo_labels[non_zero_mask]
            print(f"Number of non-zero rows: {num_non_zero_rows}")

        max_nmi = 0.0
        max_jaccard = 0.0
        max_f1 = 0.0
        if args.val_best == 'loss':
            best_val_value = float('inf')
        elif args.val_best == 'onmi':
            best_val_value = float('-inf')
        else:
            raise ValueError(f'Unknown val_best option')
        best_epoch = 0

        for epoch in range(args.max_epochs):

            graph = data
            graph = graph.to(device)
            train_mask = train_indices

            opt2.zero_grad()
            HybridBlock2.train()
            embeddings, train_pred_communities = HybridBlock2(graph.x, graph.edge_index, train_mask)
            train_pred_communities = torch.sigmoid(train_pred_communities)
            sampled_pred = train_pred_communities[sampled_train_indices]
            true_labels = data.y[sampled_node_indices].float()

            train_pred_labels = train_pred_communities[mask]
            train_pred_labels = train_pred_labels[non_zero_mask]

            loss = args.true_label_weight * F.binary_cross_entropy(sampled_pred, true_labels) + args.pseudo_label_weight * F.binary_cross_entropy(train_pred_labels, train_pseudo_labels)

            loss += l2_reg_loss(HybridBlock2, scale=args.weight_decay)
            train_loss = loss
            train_loss.backward()
            opt2.step()

            with torch.no_grad():
                opt2.zero_grad()
                HybridBlock2.eval()

                graph = data
                graph = graph.to(device)

                whether_best = False
                if args.val_best == 'loss':
                    val_mask = val_indices
                    embeddings, val_pred_communities = HybridBlock2(graph.x, graph.edge_index, val_mask)
                    val_pred_communities = torch.sigmoid(val_pred_communities)
                    val_true_labels = graph.y[val_mask]
                    val_value = F.binary_cross_entropy(val_pred_communities, val_true_labels)
                    if val_value <= best_val_value:
                        whether_best = True
                elif args.val_best == 'onmi':
                    val_mask = val_indices
                    embeddings, val_pred_communities = HybridBlock2(graph.x, graph.edge_index, val_mask)
                    val_pred_communities = torch.sigmoid(val_pred_communities)
                    Z_pred_val = val_pred_communities.cpu().detach().numpy() > 0.5
                    real_y_val = graph.y[val_mask].cpu().detach().numpy()
                    val_value = overlapping_nmi(Z_pred_val, real_y_val)
                    if val_value >= best_val_value:
                        whether_best = True
                else:
                    raise ValueError(f'Unknown val_best option')

                test_mask = test_indices
                embeddings, test_pred_communities = HybridBlock2(graph.x, graph.edge_index, test_mask)
                test_pred_communities_sigmoid = torch.sigmoid(test_pred_communities)
                Z_pred = test_pred_communities_sigmoid.cpu().detach().numpy() > 0.5

                real_y = data.y[test_indices].cpu().detach().numpy()
                nmi = overlapping_nmi(Z_pred, real_y)
                jaccard = symmetric_jaccard(Z_pred, real_y)
                f1 = symmetric_f1(Z_pred, real_y)
                if whether_best:
                    best_val_value = val_value
                    max_nmi = nmi
                    max_jaccard = jaccard
                    max_f1 = f1
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(HybridBlock2.state_dict())
                    best_test_pred_communities = test_pred_communities.cpu().detach().numpy()

                if epoch % 500 == 0 or epoch == args.max_epochs - 1:
                    print(f'Epoch: {epoch:3d} Training loss: {train_loss.item():5.4f} Val loss: {val_value.item():5.4f} ONMI: {nmi:5.3f} Jaccard:{jaccard:5.3f} F1:{f1:5.3f}')
                    print(f'==========Best Epoch so far: {best_epoch} Best Val loss: {best_val_value.item():5.4f} Best ONMI: {max_nmi:5.3f} Best Jaccard: {max_jaccard:5.3f} Best F1: {max_f1:5.3f}==========')

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    if best_model_state is not None:
        training_epoch_time = total_time / (2 * args.max_epochs)

        if args.use_pseudo:
            print(
                f"Best model in {best_epoch} with ONMI: {max_nmi:.4f} with Jaccard: {max_jaccard:.4f} with F1: {max_f1:.4f} "
                f"First Round ONMI:{round1_nmi:.4f} First Round Jaccard: {round1_jaccard:.4f} First Round F1: {round1_f1:.4f} "
                f"Training Time Per Epoch: {training_epoch_time:.4f} Training time: {total_time:.4f}")
        else:
            print(f"Best model in {best_epoch} with ONMI: {max_nmi:.4f} Training Time Per Epoch: {training_epoch_time:.4f} Training time: {total_time:.4f}")


if __name__ == '__main__':
    main()
