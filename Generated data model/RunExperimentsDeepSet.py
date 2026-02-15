import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from DeepSets import DeepSets, Head
from functools import partial
from sklearn.metrics import mean_squared_error
from Data_utils import generate_and_save, PairSetDataset, collate_fn
import os

# --- Hyperparameters ---
BATCH = 128
INPUT_DIM = 10
HIDDEN_DIM = 64
EMB_DIM = 64
AGG = 'sum'
DROPOUT = 0.0
HEAD_HIDDEN = 128
LR = 1e-3
EPOCHS = 100  # Same as Sinkhorn experiment
FIXED_M = 10000 
FIXED_DIM = 10
FIXED_N = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def relative_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred) / (torch.abs(y_true) + 1))

def run_training_session(M, N, dim, random_set_sizes, hausdorf, epochs=EPOCHS):
    print(f"--- Running Config: M={M}, N={N}, Dim={dim}, RandomSz={random_set_sizes}, Hausdorf={hausdorf} ---")
    
    # Temporary filenames to avoid overwriting main data
    TRAIN_PATH = f'temp_train_ds_M{M}_N{N}_D{dim}_R{int(random_set_sizes)}_H{int(hausdorf)}.npz'
    TEST_PATH  = f'temp_test_ds_M{M}_N{N}_D{dim}_R{int(random_set_sizes)}_H{int(hausdorf)}.npz'
    
    try:
        # 1. Generate Data
        generate_and_save(
            out_train_path=TRAIN_PATH,
            out_test_path=TEST_PATH,
            M=M,
            N=N,
            dim=dim,
            sigma=0.03,
            rng_seed=2025, 
            test_fraction=0.2,
            max_norm_multiplier=1.0,
            save_label_as='similarity', # DeepSets trains on similarity directly
            shuffle_before_split=True,
            hausdorf=hausdorf,
            random_set_sizes=random_set_sizes
        )
        
        # 2. Load Data
        train_ds = PairSetDataset(TRAIN_PATH)
        test_ds  = PairSetDataset(TEST_PATH)
        
        val_ratio = 0.2
        n_train_full = len(train_ds)
        n_train = int((1 - val_ratio) * n_train_full)
        n_val = n_train_full - n_train
        train_subset, val_subset = random_split(train_ds, [n_train, n_val])
        
        collate_fn_with_params = partial(collate_fn, random_set_sizes=random_set_sizes)
        train_loader = DataLoader(train_subset, batch_size=BATCH, shuffle=True, collate_fn=collate_fn_with_params)
        test_loader  = DataLoader(test_ds,      batch_size=BATCH, shuffle=False, collate_fn=collate_fn_with_params)
        
        # 3. Initialize Model
        encoder = DeepSets(input_dim=dim, hidden_dim=HIDDEN_DIM, output_dim=EMB_DIM,
                           aggregator=AGG, dropout=DROPOUT).to(device)
        head = Head(EMB_DIM, HEAD_HIDDEN).to(device)
        
        params = list(encoder.parameters()) + list(head.parameters())
        opt = optim.Adam(params, lr=LR)
        loss_fn = nn.MSELoss()
        
        # 4. Training Loop
        for epoch in range(1, epochs + 1):
            encoder.train()
            head.train()
            for A_batch, B_batch, y_batch, mask_batch in train_loader: 
                A_batch = A_batch.to(device)
                B_batch = B_batch.to(device)
                y_batch = y_batch.to(device) # Similarity [0,1]

                if mask_batch is not None:
                     mask_batch = mask_batch.to(device)

                # Forward pass
                emb_A = encoder(A_batch, mask=mask_batch)
                emb_B = encoder(B_batch, mask=mask_batch)
                pred = head(emb_A, emb_B) # Output is unbounded, but ideally learns to map to [0,1]
                
                loss = loss_fn(pred, y_batch)

                opt.zero_grad()
                loss.backward()
                opt.step()
                
        # 5. Evaluation
        encoder.eval()
        head.eval()
        y_true_all, y_pred_all = [], []
        with torch.no_grad():
            for A_batch, B_batch, y_batch, mask_batch in test_loader:
                A_batch = A_batch.to(device)
                B_batch = B_batch.to(device)
                y_batch = y_batch.to(device)
                if mask_batch is not None: mask_batch = mask_batch.to(device)

                emb_A = encoder(A_batch, mask=mask_batch)
                emb_B = encoder(B_batch, mask=mask_batch)
                pred = head(emb_A, emb_B)

                y_true_all.append(y_batch.cpu().numpy())
                y_pred_all.append(pred.cpu().numpy())
                
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        
        mse = mean_squared_error(y_true_all, y_pred_all)
        rel_err = relative_error(torch.tensor(y_true_all), torch.tensor(y_pred_all)).item()
        
        print(f"   -> Result: MSE={mse:.5f}, RelErr={rel_err:.5f}")
        return mse, rel_err

    finally:
        # Cleanup
        if os.path.exists(TRAIN_PATH): os.remove(TRAIN_PATH)
        if os.path.exists(TEST_PATH): os.remove(TEST_PATH)

def plot_benchmark(x_values, results_dict, x_label, title, filename):
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-', '--']
    
    for i, (label, y_values) in enumerate(results_dict.items()):
        plt.plot(x_values, y_values, 
                 label=label, 
                 color=colors[i % len(colors)], 
                 marker=markers[i % len(markers)],
                 linestyle=linestyles[i % len(linestyles)],
                 linewidth=2,
                 markersize=6)
        
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Test MSE (Similarity)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    configs = [
        ("Euclidean (Fixed Set Size)", False, False),
        ("Euclidean (Random Set Size)", True, False),
        ("Hausdorff (Fixed Set Size)", False, True),
        ("Hausdorff (Random Set Size)", True, True)
    ]
    
    # --- Experiment 1: Varying Set Size N ---
    """ print("\n" + "="*50)
    print("STARTING DEEPSETS EXPERIMENT 1: VARYING N (set size)")
    print("="*50)
    
    N_values = [10, 20, 50, 100]
    
    results_N = {label: [] for label, _, _ in configs}
    
    for N in N_values:
         for label, rnd, haus in configs:
            mse, _ = run_training_session(M=FIXED_M, N=N, dim=FIXED_DIM, random_set_sizes=rnd, hausdorf=haus)
            results_N[label].append(mse)
            
    plot_benchmark(N_values, results_N, "Set Cardinality (N)", 
                   f"DeepSets Test MSE vs Set Size (M={FIXED_M}, Dim={FIXED_DIM})", "benchmark_N_DeepSet.png")
     """
    # --- Experiment 2: Varying Dimension ---
    print("\n" + "="*50)
    print("STARTING DEEPSETS EXPERIMENT 2: VARYING DIMENSION")
    print("="*50)
    
    dim_values = [2, 10, 30, 100]
    
    results_dim = {label: [] for label, _, _ in configs}
    
    for dim in dim_values:
        for label, rnd, haus in configs:
            mse, _ = run_training_session(M=FIXED_M, N=FIXED_N, dim=dim, random_set_sizes=rnd, hausdorf=haus)
            results_dim[label].append(mse)
            
    plot_benchmark(dim_values, results_dim, "Input Dimension (d)", 
                   f"DeepSets Test MSE vs Input Dimension (M={FIXED_M}, N={FIXED_N})", "benchmark_dim_DeepSet.png")
    
    print("\nDeepSets Experiments Completed. Plots saved as 'benchmark_N_DeepSet.png' and 'benchmark_dim_DeepSet.png'.")
