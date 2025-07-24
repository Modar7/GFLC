

import numpy as np
import torch
import pandas as pd
import networkx as nx
from scipy import sparse
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
import time

class GFLC:
    def __init__(self, k=10, ricci_iter=2, alpha=0.2, beta=0.6, gamma=0.2, n_jobs=-1, 
                 approx_knn=True, sample_size=None, verbose=True, 
                 pos_threshold=0.3, neg_threshold=0.7, max_fpr=0.05):
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.max_fpr = max_fpr
        self.k = k
        self.ricci_iter = ricci_iter
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_jobs = n_jobs
        self.approx_knn = approx_knn
        self.sample_size = sample_size
        self.verbose = verbose
        self.preprocessor = None
        self.ensemble = RandomForestClassifier(
            n_estimators=50, 
            class_weight={0: 0.7, 1: 0.3},
            n_jobs=self.n_jobs
        )
        self.graph = None
        self.pos_counts = defaultdict(int)
        self.total_counts = defaultdict(int)
        self.node_index_map = None
        self.subsample_indices = None

    def fit(self, X, y, s):
        if self.verbose:
            print("Starting GFLC fit process...")
            start_time = time.time()
        
        self.X_full = X
        self.y_full = y
        self.s_full = s
        
        if self.sample_size and X.shape[0] > self.sample_size:
            if self.verbose:
                print(f"Subsampling {self.sample_size} points from {X.shape[0]}")
            self.subsample_indices = np.random.choice(X.shape[0], self.sample_size, replace=False)
            X_processed = np.array(X.iloc[self.subsample_indices].values if hasattr(X, 'iloc') 
                                  else X[self.subsample_indices], dtype=np.float32)
            y_np = y.iloc[self.subsample_indices].values if hasattr(y, 'iloc') else y[self.subsample_indices]
            s_np = s.iloc[self.subsample_indices].values if hasattr(s, 'iloc') else s[self.subsample_indices]
            self.node_index_map = {i: self.subsample_indices[i] for i in range(len(self.subsample_indices))}
        else:
            X_processed = np.array(X.values if hasattr(X, 'values') else X, dtype=np.float32)
            y_np = y.values if hasattr(y, 'values') else y
            s_np = s.values if hasattr(s, 'values') else s
            self.node_index_map = {i: i for i in range(X_processed.shape[0])}
            self.subsample_indices = np.arange(X_processed.shape[0])

        if self.verbose:
            print("Building k-NN graph...")
            knn_start = time.time()
            
        if self.approx_knn and X_processed.shape[0] > 10000:
            try:
                import pynndescent
                index = pynndescent.NNDescent(X_processed, n_neighbors=self.k, 
                                             metric='euclidean', n_jobs=self.n_jobs)
                indices, distances = index.neighbor_graph
            except ImportError:
                if self.verbose:
                    print("pynndescent not available, falling back to exact kNN")
                nbrs = NearestNeighbors(n_neighbors=self.k+1, 
                                       algorithm='auto', 
                                       n_jobs=self.n_jobs).fit(X_processed)
                distances, indices = nbrs.kneighbors(X_processed)
                distances = distances[:, 1:]
                indices = indices[:, 1:]
        else:
            nbrs = NearestNeighbors(n_neighbors=self.k+1, 
                                   algorithm='auto', 
                                   n_jobs=self.n_jobs).fit(X_processed)
            distances, indices = nbrs.kneighbors(X_processed)
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        
        edges = []
        weights = []
        for i in range(X_processed.shape[0]):
            for j_idx, j in enumerate(indices[i]):
                if i != j:
                    weight = 1.0 / (max(distances[i][j_idx], 1e-8) + 1e-8)
                    edges.append((i, j))
                    weights.append(weight)
                    
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(X_processed.shape[0]))
        self.graph.add_weighted_edges_from([(u, v, w) for (u, v), w in zip(edges, weights)])
        
        if self.verbose:
            print(f"k-NN graph built in {time.time() - knn_start:.2f} seconds")
            print(f"Graph size: {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")
            ricci_start = time.time()
            print("Starting Ricci flow...")

        if self.ricci_iter > 0:
            edge_list = list(self.graph.edges())
            batch_size = min(10000, len(edge_list))
            for iter_idx in range(self.ricci_iter):
                if self.verbose and iter_idx % max(1, self.ricci_iter // 5) == 0:
                    print(f"Ricci iteration {iter_idx+1}/{self.ricci_iter}")
                for i in range(0, len(edge_list), batch_size):
                    batch_edges = edge_list[i:i+batch_size]
                    curvature_batch = {}
                    for u, v in batch_edges:
                        w_uv = max(self.graph[u][v]['weight'], 1e-8)
                        u_weights = np.array([w_uv/max(self.graph[u][x]['weight'], 1e-8) 
                                          for x in self.graph.neighbors(u) if x != v])
                        v_weights = np.array([w_uv/max(self.graph[v][x]['weight'], 1e-8) 
                                          for x in self.graph.neighbors(v) if x != u])
                        sum_u = np.sum(np.sqrt(u_weights)) if len(u_weights) > 0 else 0
                        sum_v = np.sum(np.sqrt(v_weights)) if len(v_weights) > 0 else 0
                        curvature_batch[(u, v)] = w_uv * (1 - (sum_u + sum_v)/2)
                    for (u, v), c in curvature_batch.items():
                        if self.graph.has_edge(u, v):
                            self.graph[u][v]['weight'] = max(
                                self.graph[u][v]['weight'] + 0.1 * c, 1e-8
                            )

        if self.verbose:
            if self.ricci_iter > 0:
                print(f"Ricci flow completed in {time.time() - ricci_start:.2f} seconds")
            model_start = time.time()
            print("Training classifier model...")

        self.ensemble.fit(X_processed, y_np)
        self.probs = self.ensemble.predict_proba(X_processed)[:, 1]

        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_np, self.probs)
        optimal_idx = np.argmax(tpr - 5*fpr)
        self.pos_threshold = thresholds[optimal_idx]
        self.neg_threshold = 1 - self.pos_threshold
        
        pos_mask = (self.probs >= self.pos_threshold)
        neg_mask = (self.probs <= self.neg_threshold)
        self.margins = np.zeros_like(self.probs)
        self.margins[pos_mask] = self.probs[pos_mask] - self.pos_threshold
        self.margins[neg_mask] = self.neg_threshold - self.probs[neg_mask]

        if self.verbose:
            print(f"Model training completed in {time.time() - model_start:.2f} seconds")
            counts_start = time.time()
            print("Computing group statistics...")

        unique_groups = np.unique(s_np)
        for group in unique_groups:
            group_mask = (s_np == group)
            self.pos_counts[group] = np.sum(y_np[group_mask])
            self.total_counts[group] = np.sum(group_mask)

        if self.verbose:
            print(f"Group statistics computed in {time.time() - counts_start:.2f} seconds")
            preproc_start = time.time()
            print("Preprocessing graph for GPU...")

        self._preprocess_graph_for_gpu()

        if self.verbose:
            print(f"Total fit time: {time.time() - start_time:.2f} seconds")

    def _preprocess_graph_for_gpu(self):
        """Convert graph to COO format components on GPU"""
        adj_matrix = nx.adjacency_matrix(self.graph, weight='weight').tocoo()
        
        # Debugging: Print matrix info
        print("adj_matrix shape:", adj_matrix.shape)
        print("adj_matrix.row min:", adj_matrix.row.min(), "max:", adj_matrix.row.max())
        print("adj_matrix.col min:", adj_matrix.col.min(), "max:", adj_matrix.col.max())
        
        # Validation: Ensure indices are within bounds
        num_nodes = adj_matrix.shape[0]
        assert adj_matrix.row.min() >= 0 and adj_matrix.row.max() < num_nodes, \
            f"Invalid row indices: min={adj_matrix.row.min()}, max={adj_matrix.row.max()}, expected [0, {num_nodes-1}]"
        assert adj_matrix.col.min() >= 0 and adj_matrix.col.max() < num_nodes, \
            f"Invalid col indices: min={adj_matrix.col.min()}, max={adj_matrix.col.max()}, expected [0, {num_nodes-1}]"
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move data to device
        self.rows = torch.tensor(adj_matrix.row, dtype=torch.long, device=self.device)
        self.cols = torch.tensor(adj_matrix.col, dtype=torch.long, device=self.device)
        self.weights = torch.tensor(adj_matrix.data, dtype=torch.float32, device=self.device)
        self.num_nodes = num_nodes
        
        # Store subsample indices tensor
        self.subsample_indices_tensor = torch.tensor(self.subsample_indices, device=self.device)
        
        if self.verbose:
            print(f"Graph preprocessing completed. Using device: {self.device}")


    def estimate_flips(self, noise_rates):
        """
        Estimate optimal pos_flips and neg_flips based on noise rates.
        
        Parameters:
        - noise_rates (dict): Noise rates per group, e.g., {0: 0.0, 1: 0.2}.
        
        Returns:
        - tuple: (pos_flips, neg_flips)
        """
        # Assume group 0 has zero noise to estimate true positive rate
        if 0 in noise_rates and noise_rates[0] == 0:
            p = self.pos_counts[0] / self.total_counts[0]
        else:
            raise ValueError("Need a group with zero noise to estimate true positive rate")
        
        pos_flips = 0
        neg_flips = 0
        for group, rate in noise_rates.items():
            if rate > 0:
                group_size = self.total_counts[group]
                pos_flips += int((1 - p) * rate * group_size)  # False positives
                neg_flips += int(p * rate * group_size)        # False negatives
        
        return pos_flips, neg_flips
    
    
    def compute_flips(self, D=0.05):
        if D < 0 or D > 1:
            raise ValueError("Disparity target D must be between 0 and 1.")
        
        P_y = np.mean(self.y_full)  # Overall prevalence
        P_y_l = P_y * (1 - D)      # Lower bound
        P_y_h = P_y * (1 + D)      # Upper bound
        
        pos_flips = 0
        neg_flips = 0
        
        unique_groups = np.unique(self.s_full)
        for group in unique_groups:
            group_mask = (self.s_full == group)
            group_y = self.y_full[group_mask]
            group_size = np.sum(group_mask)
            if group_size == 0:
                continue
            current_pos = np.sum(group_y)
            P_y_s = current_pos / group_size
            
            if P_y_s < P_y_l:
                target_pos_l = np.ceil(P_y_l * group_size)
                flips = target_pos_l - current_pos
                pos_flips += max(int(flips), 0)
            elif P_y_s > P_y_h:
                target_pos_h = np.floor(P_y_h * group_size)
                flips = current_pos - target_pos_h
                neg_flips += max(int(flips), 0)
        
        return pos_flips, neg_flips


    def correct_labels(self, X, y, s, pos_flips=None, neg_flips=None, noise_rates=None, disparity_target=None, max_flips=None):
        """
        Correct labels with optional flip estimation methods.
        
        Parameters:
        - X, y, s: Input data, labels, and sensitive attributes.
        - pos_flips, neg_flips: Number of flips; if None, estimated via noise_rates or disparity_target.
        - noise_rates (dict): Optional noise rates to estimate flips.
        - disparity_target (float): Optional disparity target for Fair-OBNC flip computation.
        - max_flips (int): Optional total flips to override pos_flips and neg_flips.
        """
        if pos_flips is None and neg_flips is None:
            if disparity_target is not None:
                pos_flips, neg_flips = self.compute_flips(D=disparity_target)
                if self.verbose:
                    print(f"Fair-OBNC estimated pos_flips={pos_flips}, neg_flips={neg_flips}")
            elif noise_rates is not None:
                pos_flips, neg_flips = self.estimate_flips(noise_rates)
                if self.verbose:
                    print(f"Noise-based estimated pos_flips={pos_flips}, neg_flips={neg_flips}")
            else:
                raise ValueError("Must provide pos_flips and neg_flips, noise_rates, or disparity_target.")
        
        if max_flips is not None:
            pos_flips = int(max_flips * 0.7)
            neg_flips = max_flips - pos_flips
        
        # Rest of your existing correct_labels code follows...
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        y_tensor = torch.from_numpy(np.array(y.values if hasattr(y, 'values') else y)).float().to(device)
        y_np = y_tensor.cpu().numpy()
        class_weights = torch.where(y_tensor == 1, 1/0.01, 1/0.99)
        
        if self.verbose:
            print(f"Starting label correction for {X.shape[0]} samples...")
            start_time = time.time()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = self.device
        
        if self.sample_size and X.shape[0] > self.sample_size:
            if self.verbose:
                print("Generating margins for full dataset using trained model...")
            X_processed = np.array(X.values if hasattr(X, 'values') else X, dtype=np.float32)
            full_probs = self.ensemble.predict_proba(X_processed)[:, 1]
            full_margins = np.abs(full_probs - 0.5)
            margins_tensor = torch.from_numpy(full_margins).float().to(device)
        else:
            margins_tensor = torch.from_numpy(self.margins).float().to(device)
            
        y_tensor = torch.from_numpy(np.array(y.values if hasattr(y, 'values') else y)).float().to(device)
        s_tensor = torch.from_numpy(np.array(s.values if hasattr(s, 'values') else s)).long().to(device)
        
        group_ids = list(self.pos_counts.keys())
        group_pos = torch.tensor([self.pos_counts[g] for g in group_ids], device=device).float()
        group_total = torch.tensor([self.total_counts[g] for g in group_ids], device=device).float()
    
        batch_size = 10000
        if self.verbose:
            print(f"Using batch size of {batch_size}")
        
        n_samples = y_tensor.size(0)
        scores = torch.zeros(n_samples, device=device)
    
        if self.sample_size and X.shape[0] > self.sample_size:
            subsample_indices_set = set(self.subsample_indices.tolist())
            idx_to_subsample_pos = {idx: i for i, idx in enumerate(self.subsample_indices.tolist())}
        
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_indices = torch.arange(batch_start, batch_end, device=device)
            batch_indices_cpu = batch_indices.cpu().numpy()
    
            batch_margins = margins_tensor[batch_start:batch_end]
            batch_y = y_tensor[batch_indices]
            batch_s = s_tensor[batch_indices]
    
            margin_term = 1 - batch_margins
            lap_term = torch.zeros_like(batch_margins)
            
            if self.sample_size and X.shape[0] > self.sample_size:
                batch_to_subsample = {}
                for i, idx in enumerate(batch_indices_cpu):
                    if idx in subsample_indices_set:
                        batch_to_subsample[i] = idx_to_subsample_pos[idx]
                
                if batch_to_subsample:
                    batch_idx_in_subsample = torch.tensor(list(batch_to_subsample.keys()), device=device)
                    subsample_pos = torch.tensor(list(batch_to_subsample.values()), device=device)
                    mask = torch.isin(self.rows, subsample_pos)
                    if torch.any(mask):
                        sub_rows = self.rows[mask]
                        sub_cols = self.cols[mask]
                        sub_weights = self.weights[mask]
                        sub_pos_to_batch_idx = {sub.item(): batch_idx for batch_idx, sub in zip(batch_idx_in_subsample, subsample_pos)}
                        valid_rows_mask = torch.tensor([r.item() in sub_pos_to_batch_idx for r in sub_rows], device=device)
                        if torch.any(valid_rows_mask):
                            valid_rows = sub_rows[valid_rows_mask]
                            valid_cols = sub_cols[valid_rows_mask]
                            valid_weights = sub_weights[valid_rows_mask]
                            batch_row_indices = torch.tensor([sub_pos_to_batch_idx[r.item()] for r in valid_rows], device=device)
                            col_bounds_mask = (valid_cols >= 0) & (valid_cols < len(self.subsample_indices_tensor))
                            if torch.any(col_bounds_mask):
                                batch_row_indices = batch_row_indices[col_bounds_mask]
                                valid_cols = valid_cols[col_bounds_mask]
                                valid_weights = valid_weights[col_bounds_mask]
                                row_bounds_mask = (batch_row_indices >= 0) & (batch_row_indices < len(batch_y))
                                if torch.any(row_bounds_mask):
                                    batch_row_indices = batch_row_indices[row_bounds_mask]
                                    valid_cols = valid_cols[row_bounds_mask]
                                    valid_weights = valid_weights[row_bounds_mask]
                                    batch_y_i = batch_y[batch_row_indices]
                                    original_col_indices = self.subsample_indices_tensor[valid_cols]
                                    valid_original_mask = (original_col_indices >= 0) & (original_col_indices < len(y_tensor))
                                    if torch.any(valid_original_mask):
                                        batch_row_indices = batch_row_indices[valid_original_mask]
                                        valid_weights = valid_weights[valid_original_mask]
                                        original_col_indices = original_col_indices[valid_original_mask]
                                        batch_y_i = batch_y_i[valid_original_mask]
                                        full_y_j = y_tensor[original_col_indices]
                                        y_diff = batch_y_i - full_y_j
                                        lap_values = (valid_weights * y_diff).pow(2)
                                        for i, r in enumerate(batch_row_indices):
                                            lap_term[r] += lap_values[i]
            else:
                orig_indices = batch_indices
                mask = torch.isin(self.rows, orig_indices)
                if torch.any(mask):
                    batch_rows = self.rows[mask]
                    batch_cols = self.cols[mask]
                    batch_weights = self.weights[mask]
                    batch_row_indices = batch_rows - batch_start
                    valid_mask = (batch_row_indices >= 0) & (batch_row_indices < len(batch_y))
                    if torch.any(valid_mask):
                        valid_batch_rows = batch_row_indices[valid_mask]
                        valid_batch_cols = batch_cols[valid_mask]
                        valid_batch_weights = batch_weights[valid_mask]
                        col_valid_mask = (valid_batch_cols >= 0) & (valid_batch_cols < len(y_tensor))
                        if torch.any(col_valid_mask):
                            valid_batch_rows = valid_batch_rows[col_valid_mask]
                            valid_batch_cols = valid_batch_cols[col_valid_mask]
                            valid_batch_weights = valid_batch_weights[col_valid_mask]
                            y_diff = batch_y[valid_batch_rows] - y_tensor[valid_batch_cols]
                            lap_values = (valid_batch_weights * y_diff).pow(2)
                            for i, r in enumerate(valid_batch_rows):
                                lap_term[r] += lap_values[i]
    
            group_ids_tensor = torch.tensor(group_ids, device=device)
            group_mask = batch_s.unsqueeze(1) == group_ids_tensor.unsqueeze(0)
            delta = (1 - 2 * batch_y).unsqueeze(1)
            group_impact = (delta * group_mask).sum(dim=0)
            new_pos = group_pos + group_impact
            new_rates = new_pos / group_total.clamp(min=1e-8)
            original_min = group_pos.min() / group_total.max().clamp(min=1e-8)
            new_min = new_rates.min() / new_rates.max().clamp(min=1e-8)
            fairness_impact = new_min - original_min
    
            scores[batch_indices] = class_weights[batch_indices] * (
                self.alpha * margin_term + 
                self.beta * lap_term + 
                self.gamma * fairness_impact
            )
    
        pos_mask = y_tensor == 1
        neg_mask = y_tensor == 0
        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]
        
        print('-------------------------------------')
        print(f'scores shape: {scores.shape}')
        print(f'pos_scores shape: {pos_scores.shape}')
        print(f'neg_scores shape: {neg_scores.shape}')
        print('-------------------------------------')
        
        if len(pos_scores) > 0:
            pos_indices = torch.topk(pos_scores, min(pos_flips, len(pos_scores)))[1]
            pos_global = torch.nonzero(pos_mask).squeeze()[pos_indices]
        else:
            pos_global = torch.tensor([], device=device, dtype=torch.long)
            
        if len(neg_scores) > 0:
            neg_indices = torch.topk(neg_scores, min(neg_flips, len(neg_scores)))[1]
            neg_global = torch.nonzero(neg_mask).squeeze()[neg_indices]
        else:
            neg_global = torch.tensor([], device=device, dtype=torch.long)
        
        if len(pos_global.shape) == 0 and len(pos_global) > 0:
            pos_global = pos_global.unsqueeze(0)
        if len(neg_global.shape) == 0 and len(neg_global) > 0:
            neg_global = neg_global.unsqueeze(0)
        
        all_indices = torch.cat([pos_global, neg_global]) if (len(pos_global) > 0 or len(neg_global) > 0) else torch.tensor([], device=device, dtype=torch.long)
    
        corrected_y = y_tensor.clone()
        if len(all_indices) > 0:
            corrected_y[all_indices] = 1 - corrected_y[all_indices]
        
        if torch.any(neg_mask):
            current_fpr = (corrected_y[neg_mask] == 1).sum() / neg_mask.sum()
            if self.verbose:
                print(f"Current FPR: {current_fpr:.4f}, Max FPR: {self.max_fpr}")
            #if current_fpr > self.max_fpr:
            #    print(f"FPR {current_fpr:.2f} exceeds max {self.max_fpr}, rolling back")
            #    return y_tensor.cpu().numpy()
    
        return corrected_y.cpu().numpy()

    def save_model(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        if self.verbose:
            print(f"Model saved to {filename}")

    @classmethod
    def load_model(cls, filename):
        """Load a trained model from a file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)


