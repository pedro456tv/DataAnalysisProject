import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import os
import torch
from torch.utils.data import Dataset


def generate_pair_fixed_norm(N, dim, s_target, sigma, rng, max_norm, hausdorf=False):
    """
    Generate a pair of point sets (A,B) each with N points in `dim` dimensions,
    such that the average similarity between matched points is around s_target,
    with per-point similarity noise of `sigma`. Similarity is defined as
    1 - (distance / max_norm), where max_norm is a fixed normalization constant."""
    A = rng.random((N, dim))
    # per-point similarities around s_target
    s_i = rng.normal(loc=s_target, scale=sigma, size=N)
    s_i = np.clip(s_i, 0.0, 1.0)
    
    # distances so that if normalized by max_norm similarity = s_i
    d_i = (1.0 - s_i) * max_norm
    # random directions
    dirs = rng.normal(size=(N, dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    if hausdorf:
        h_target = (1.0 - s_target) * max_norm
        
        r_i = np.minimum(d_i, h_target)
        k = rng.integers(0, N)
        r_i[k] = h_target
        B = A + dirs * r_i[:, None]
        D = cdist(A, B)
        d_ab = np.max(np.min(D, axis=1))  # for each A find nearest B, then worst
        d_ba = np.max(np.min(D, axis=0))  # for each B find nearest A, then worst
        haus = max(d_ab, d_ba)
        set_similarity = 1.0 - (haus / max_norm)
        set_similarity = float(np.clip(set_similarity, 0.0, 1.0))

        nearest_dist_A = np.min(D, axis=1)
        assigned_similarities = 1.0 - (nearest_dist_A / max_norm)
        assigned_similarities = np.clip(assigned_similarities, 0.0, 1.0)

        return A, B, s_i, set_similarity, assigned_similarities
    
    B = A + (dirs * d_i[:, None])
    # compute pairwise distances and similarity matrix using fixed max_norm Euclidean distance
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    S = 1 - (D / max_norm)
    S = np.clip(S, 0.0, 1.0)
    cost = 1 - S
    row_ind, col_ind = linear_sum_assignment(cost)
    assigned_similarities = S[row_ind, col_ind]
    set_similarity = assigned_similarities.mean()
    return A, B, s_i, set_similarity, assigned_similarities

# ---------------- dataset generator + train/test saver ----------------
def generate_and_save(
    out_train_path,
    out_test_path,
    M=8000,
    N=20,
    dim=5,
    sigma=0.03,
    rng_seed=42,
    test_fraction=0.2,
    max_norm_multiplier=1.0,
    save_label_as='similarity',   # 'similarity' or 'cost'
    shuffle_before_split=True,
    hausdorf=False,
    random_set_sizes=False
):
    """
    Generate M pairs (A,B) with N points in `dim` and save train/test .npz files
    with keys exactly: 'A', 'B', 'y' (so PairSetDataset works directly).

    Parameters:
    - out_train_path, out_test_path : target filenames
    - M : number of pairs total
    - N : points per set
    - dim : dimension
    - sigma : per-point similarity noise (small keeps actual close to target)
    - test_fraction : fraction reserved for test set (default 0.2)
    - max_norm_multiplier : multiplier for sqrt(dim) used as fixed normalization (default 2.0)
    - save_label_as : 'similarity' (default) or 'cost' (1 - similarity)
    - shuffle_before_split : shuffle dataset randomly before splitting
    """
    rng = np.random.default_rng(rng_seed)
    max_norm = max_norm_multiplier * np.sqrt(dim)

    

    A_all = []
    B_all = []
    y_all = []

    s_targets = np.linspace(0, 1, M)

    for m in range(M):
        if random_set_sizes:
            N_i = rng.integers(low=1, high=N)
        else:
            N_i = N
        s_target = np.clip(s_targets[m] + rng.normal(0, sigma), 0, 1)
        A, B, s_i, set_sim, assigned_s = generate_pair_fixed_norm(N_i, dim, s_target, sigma, rng, max_norm, hausdorf=hausdorf)
        A_all.append(A)
        B_all.append(B)
        y_all.append(set_sim)

        if (m + 1) % 500 == 0 or (m+1) == M:
            print(f"Generated {m+1}/{M} pairs...")

    y_all = np.array(y_all, dtype=np.float32)
    # Optionally shuffle dataset before splitting
    idx = np.arange(M)
    if shuffle_before_split:
        rng.shuffle(idx)
    A_all = [A_all[i] for i in idx]
    B_all = [B_all[i] for i in idx]
    y_all = y_all[idx]

    # Convert labels if user wants cost instead of similarity
    if save_label_as == 'cost':
        y_saved = 1.0 - y_all
    elif save_label_as == 'similarity':
        y_saved = y_all.copy()
    else:
        raise ValueError("save_label_as must be 'similarity' or 'cost'")

    # Split into train/test
    n_test = int(np.ceil(M * test_fraction))
    n_train = M - n_test
    A_train = A_all[:n_train]
    B_train = B_all[:n_train]
    y_train = y_saved[:n_train]
    A_test = A_all[n_train:]
    B_test = B_all[n_train:]
    y_test = y_saved[n_train:]

    # Ensure output directories exist
    os.makedirs(os.path.dirname(out_train_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_test_path) or ".", exist_ok=True)

    # Save in the format your PairSetDataset expects: keys 'A','B','y'
    np.savez_compressed(out_train_path, A=np.array(A_train, dtype=object), B=np.array(B_train, dtype=object), y=y_train)
    np.savez_compressed(out_test_path,  A=np.array(A_test, dtype=object),  B=np.array(B_test, dtype=object),  y=y_test)

    print("Saved train/test datasets:")
    print("  train ->", out_train_path, ", shape A:", len(A_train), "y:", len(y_train))
    print("  test  ->", out_test_path,  ", shape A:", len(A_test),  "y:", len(y_test))
    print("Label saved as:", save_label_as)
    print("Label stats (train): mean =", float(y_train.mean()), "std =", float(y_train.std()))

    # plot histogram of actual similarities and requested targets
    fig, axes = plt.subplots(1,1, figsize=(12,4))
    axes.hist(y_all, bins=40)
    axes.set_title("Actual set similarity")
    axes.set_xlabel("actual similarity (0..1)")
    axes.set_ylabel("count")
    plt.tight_layout()
    plt.show()

    # also return metadata in case you want to inspect immediately
    return {
        'out_train_path': out_train_path,
        'out_test_path': out_test_path,
        'A_train_shape': len(A_train),
        'A_test_shape': len(A_test),
        'y_train_mean': float(y_train.mean()),
        'y_train_std': float(y_train.std()),
        'max_norm_used': max_norm
    }

# ---------- small dataset wrapper ----------
class PairSetDataset(Dataset):
    """Dataset wrapper for pairs of sets stored in .npz files"""
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.A = data['A']   # shape (Npairs, n_points, dim)
        self.B = data['B']
        self.y = data['y']   # shape (Npairs,)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.A[idx], self.B[idx], self.y[idx]

def collate_fn(batch, random_set_sizes=False):
    """Collate function to pad variable-sized sets in a batch with zeros"""
    batch_size = len(batch)
    dim = batch[0][0].shape[1]
    # find max size in this batch
    max_n = max(b[0].shape[0] for b in batch)
    
    # allocate padded tensors
    As = np.zeros((batch_size, max_n, dim), dtype=np.float32)
    Bs = np.zeros((batch_size, max_n, dim), dtype=np.float32)
    ys = np.zeros((batch_size,), dtype=np.float32)
    sizes = np.zeros((batch_size,), dtype=np.int32)
    
    for i, (A_i, B_i, y_i) in enumerate(batch):
        n_i = A_i.shape[0]
        As[i, :n_i, :] = A_i
        Bs[i, :n_i, :] = B_i
        ys[i] = y_i
        sizes[i] = n_i
    
    # convert to torch
    A_t = torch.from_numpy(As).float()
    B_t = torch.from_numpy(Bs).float()
    y_t = torch.from_numpy(ys).float()
    sizes_t = torch.from_numpy(sizes).long()
    
    # create mask: (B, N) with True for real points, False for padding
    mask = torch.arange(max_n).unsqueeze(0) < sizes_t.unsqueeze(1)  # (B, N)
    if random_set_sizes:
        return A_t, B_t, y_t, mask
    return A_t, B_t, y_t , None