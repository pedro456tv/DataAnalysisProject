
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from torch.utils.data import DataLoader, random_split
from DeepSets import DeepSets, Head
from Data_utils import generate_and_save, PairSetDataset, collate_fn
from functools import partial
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_and_validate(config, train_loader, val_loader, device, epochs=20):
    # Unpack config
    input_dim = config['input_dim']
    hidden_dim = config['hidden_dim']
    emb_dim = config['emb_dim']
    dropout = config.get('dropout', 0.0)
    head_hidden = config['head_hidden']
    lr = config['lr']
    
    # Initialize model
    encoder = DeepSets(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=emb_dim, dropout=dropout).to(device)
    head = Head(emb_dim, head_hidden).to(device)
    
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = optim.Adam(params, lr=lr)
    loss_fn = nn.MSELoss()
    
    # Training Loop
    for epoch in range(epochs):
        encoder.train()
        head.train()
        for A_batch, B_batch, y_batch, mask_batch in train_loader:
            A_batch, B_batch = A_batch.to(device), B_batch.to(device)
            y_batch = y_batch.to(device)
            if mask_batch is not None:
                mask_batch = mask_batch.to(device)
            
            optimizer.zero_grad()
            embA = encoder(A_batch, mask=mask_batch)
            embB = encoder(B_batch, mask=mask_batch)
            pred = head(embA, embB)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            
    # Validation Loop
    encoder.eval()
    head.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for A_batch, B_batch, y_batch, mask_batch in val_loader:
            A_batch, B_batch = A_batch.to(device), B_batch.to(device)
            y_batch = y_batch.to(device)
            if mask_batch is not None:
                mask_batch = mask_batch.to(device)
                
            embA = encoder(A_batch, mask=mask_batch)
            embB = encoder(B_batch, mask=mask_batch)
            pred = head(embA, embB)
            loss = loss_fn(pred, y_batch)
            total_val_loss += loss.item() * A_batch.size(0)
            
    avg_val_loss = total_val_loss / len(val_loader.dataset)
    return avg_val_loss, encoder, head

def main():
    # --- Data Setup ---
    TRAIN_PATH = 'C:/Users/PC1/Downloads/train_similarity_final.npz'
    TEST_PATH  = 'C:/Users/PC1/Downloads/test_similarity_final.npz'
    random_set_sizes = False
    
    # Ensure data exists (similar to DeepSet.py)
    print("Checking/Generating Data...")
    generate_and_save(
        out_train_path=TRAIN_PATH,
        out_test_path=TEST_PATH,
        M=10000,
        N=20,
        dim=10,
        sigma=0.03,
        rng_seed=20251031,
        test_fraction=0.2,
        max_norm_multiplier=1.0,
        save_label_as='similarity',
        shuffle_before_split=True,
        hausdorf=False,
        random_set_sizes=random_set_sizes
    )
    
    # Load Main Datasets
    full_train_ds = PairSetDataset(TRAIN_PATH)
    test_ds = PairSetDataset(TEST_PATH)
    
    # --- Train/Validation Split ---
    # User requested split: partitioning the training file into Train and Validation
    val_ratio = 0.2
    train_size = int((1 - val_ratio) * len(full_train_ds))
    val_size = len(full_train_ds) - train_size
    
    train_subset, val_subset = random_split(full_train_ds, [train_size, val_size])
    print(f"Data Split: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_ds)}")

    # Dataloaders
    batch_size = 128
    collate = partial(collate_fn, random_set_sizes=random_set_sizes)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Grid Search Configuration ---
    param_grid = {
        'hidden_dim': [64, 128],
        'lr': [1e-3, 1e-2, 1e-4],
        'dropout': [0.0, 0.2, 0.5]
    }
    
    # Fixed parameters
    fixed_params = {
        'input_dim': 10,
        'emb_dim': 64,
        'head_hidden': 128,
        'epochs': 20  # Reduced epochs for grid search speed
    }

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Starting Grid Search with {len(combinations)} configurations...")
    
    best_val_loss = float('inf')
    best_config = None
    results = []

    for i, config in enumerate(combinations):
        # Merge with fixed params
        full_config = {**config, **fixed_params}
        print(f"Run {i+1}/{len(combinations)}: {config}", end=" ... ")
        
        val_loss, _, _ = train_and_validate(full_config, train_loader, val_loader, device, epochs=full_config['epochs'])
        
        print(f"Val MSE: {val_loss:.6f}")
        results.append((config, val_loss))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = full_config

    print("\n--- Grid Search Complete ---")
    print(f"Best Validation MSE: {best_val_loss:.6f}")
    print(f"Best Configuration: {best_config}")
    
    # --- Final Evaluation with Best Model ---
    print("\nRetraining with Best Configuration on Data...")
    # Optionally, you could retrain on train+val here, but often just checking the best model found is good enough for valid/test comparison
    # or retrain fresh on train_subset for longer epochs.
    
    best_config['epochs'] = 100 # Train longer for final result
    final_loss, best_encoder, best_head = train_and_validate(best_config, train_loader, val_loader, device, epochs=best_config['epochs'])
    
    print("Evaluating on Test Set...")
    best_encoder.eval()
    best_head.eval()
    y_true_all, y_pred_all = [], []
    
    with torch.no_grad():
        for A_batch, B_batch, y_batch, mask_batch in test_loader:
            A_batch, B_batch = A_batch.to(device), B_batch.to(device)
            y_batch = y_batch.to(device)
            if mask_batch is not None:
                mask_batch = mask_batch.to(device)
                
            pred = best_head(best_encoder(A_batch, mask=mask_batch), best_encoder(B_batch, mask=mask_batch))
            
            y_true_all.append(y_batch.cpu().numpy())
            y_pred_all.append(pred.cpu().numpy())
            
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    
    mse = mean_squared_error(y_true_all, y_pred_all)
    mae = mean_absolute_error(y_true_all, y_pred_all)
    
    print(f"Final Test MSE: {mse:.6f}")
    print(f"Final Test MAE: {mae:.6f}")

if __name__ == "__main__":
    main()
