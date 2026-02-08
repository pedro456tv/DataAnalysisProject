import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from DeepSets import DeepSets, Head
from functools import partial
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Data_utils import generate_and_save, PairSetDataset, collate_fn

def relative_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred) / (torch.abs(y_true) + 1))

#Change paths as needed
TRAIN_PATH = 'C:/Users/PC1/Downloads/train_similarity_final.npz'
TEST_PATH  = 'C:/Users/PC1/Downloads/test_similarity_final.npz'
#Change paths as needed

BATCH = 128
random_set_sizes = False #Change to True to test variable set sizes

meta = generate_and_save(
        out_train_path=TRAIN_PATH,
        out_test_path=TEST_PATH,
        M=10000,
        N=20,
        dim=10,
        sigma=0.03,
        rng_seed=20251031,
        test_fraction=0.2,
        max_norm_multiplier=1.0,        #norm changer
        save_label_as='similarity',   # change to 'cost' if you prefer
        shuffle_before_split=True,
        hausdorf=False, #Change to True to use Hausdorff distance based similarity else Euclidean
        random_set_sizes=random_set_sizes
    )
print("Done. Metadata:", meta)

train_ds = PairSetDataset(TRAIN_PATH)
test_ds  = PairSetDataset(TEST_PATH)

# --- Create Train/Validation Split ---
val_ratio = 0.2
n_train_full = len(train_ds)
n_train = int((1 - val_ratio) * n_train_full)
n_val = n_train_full - n_train
train_subset, val_subset = random_split(train_ds, [n_train, n_val])

print(f"Data Split: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_ds)}")

#normalization of label
y_train_all = train_ds.y
#y_train_all = 1.0-y_train_all  # convert back to cost if needed
y_mean = float(y_train_all.mean())
y_std  = float(y_train_all.std()) if float(y_train_all.std())>0 else 1.0
print("Label mean/std:", y_mean, y_std)

from functools import partial
collate_fn_with_params = partial(collate_fn, random_set_sizes=random_set_sizes)

# Create loaders for Train, Val, and Test
train_loader = DataLoader(train_subset, batch_size=BATCH, shuffle=True, collate_fn=collate_fn_with_params)
val_loader   = DataLoader(val_subset,   batch_size=BATCH, shuffle=False, collate_fn=collate_fn_with_params)
test_loader  = DataLoader(test_ds,      batch_size=BATCH, shuffle=False, collate_fn=collate_fn_with_params)

# ---------- hyperparameters ----------
INPUT_DIM = 10
HIDDEN_DIM = 128
EMB_DIM = 64
AGG = 'sum'
DROPOUT = 0.0

HEAD_HIDDEN = 128
LR = 1e-3
EPOCHS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

encoder = DeepSets(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=EMB_DIM,
                   aggregator=AGG, dropout=DROPOUT).to(device)

head = Head(EMB_DIM, HEAD_HIDDEN).to(device)

params = list(encoder.parameters()) + list(head.parameters())

opt = optim.Adam(params, lr=LR)
loss_fn = nn.MSELoss()
val_loss_epochs = []

train_loss_epochs = []

# ---------- training loop ----------
for epoch in range(1, EPOCHS + 1):
    encoder.train()
    head.train()
    running_loss = 0.0
    for A_batch, B_batch, y_batch, mask_batch in train_loader: #added mask_batch if random sizes
        A_batch = A_batch.to(device)
        B_batch = B_batch.to(device)
        y_batch = y_batch.to(device)
        if mask_batch is not None:
            mask_batch = mask_batch.to(device)

        # original pooled-head training (unchanged)
        embA = encoder(A_batch, mask=mask_batch)   # (B, EMB_DIM)
        embB = encoder(B_batch, mask=mask_batch)
        
        pred = head(embA, embB)
        loss = loss_fn(pred, y_batch)
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        running_loss += float(loss.item()) * A_batch.size(0)

    train_loss_epoch = running_loss / len(train_loader.dataset)
    train_loss_epochs.append(train_loss_epoch)

    encoder.eval(); head.eval()
    val_loss = 0.0
    total_mse = 0.0
    relative = 0.0
    with torch.no_grad():
        for A_batch, B_batch, y_batch, mask_batch in val_loader:
            A_batch = A_batch.to(device)
            B_batch = B_batch.to(device)
            y_batch = y_batch.to(device)
            if mask_batch is not None:
                mask_batch = mask_batch.to(device)
            
            embA = encoder(A_batch, mask=mask_batch)
            embB = encoder(B_batch, mask=mask_batch)
            pred = head(embA, embB)
            
            mse_batch = ((pred - y_batch) ** 2).sum().item()
            relative += relative_error(y_batch, pred).item() * A_batch.size(0)
            loss = loss_fn(pred, y_batch)
            val_loss += float(loss.item()) * A_batch.size(0)
            total_mse += mse_batch

    val_mse = total_mse / len(val_loader.dataset)
    relative = relative / len(val_loader.dataset)
         
    val_loss_epoch = val_loss / len(val_loader.dataset)
    val_loss_epochs.append(val_loss_epoch)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d}  train_loss={train_loss_epoch:.6f}  val_loss={val_loss_epoch:.6f}  val_MSE={val_mse:.6f} val_rel={relative:.6f}") #

# ---------- final evaluation ----------
y_true_all, y_pred_all = [], []
with torch.no_grad():
    for A_batch, B_batch, y_batch, mask_batch in test_loader:
        A_batch, B_batch = A_batch.to(device), B_batch.to(device)
        y_batch = y_batch.to(device)
        if mask_batch is not None:
            mask_batch = mask_batch.to(device)

        pred = head(encoder(A_batch, mask=mask_batch), encoder(B_batch, mask=mask_batch))
        
        y_true_all.append(y_batch.cpu().numpy())
        y_pred_all.append(pred.cpu().numpy())

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)

mse = mean_squared_error(y_true_all, y_pred_all)
mae = mean_absolute_error(y_true_all, y_pred_all)
relative_err = relative_error(torch.tensor(y_true_all), torch.tensor(y_pred_all)).item()
r2 = r2_score(y_true_all, y_pred_all)
corr = np.corrcoef(y_true_all, y_pred_all)[0,1]

print(f"Final Test Results:\nMSE={mse:.4f}, RMSE={np.sqrt(mse):.4f}, MAE={mae:.4f}, R2={r2:.4f}, Corr={corr:.4f}, Relative Error={relative_err:.4f}")

# Plot predicted vs true
plt.scatter(y_true_all, y_pred_all, s=10, alpha=0.5)
plt.plot([0,1],[0,1],'r--')
plt.xlabel("True similarity")
plt.ylabel("Predicted similarity")
plt.title(f"Predicted vs True similarity different set sizes={random_set_sizes}")
plt.show()

# ---------- plot training loss ----------
plt.figure(figsize=(7,4))
plt.plot(range(1, EPOCHS+1), train_loss_epochs, label='Train Loss')
plt.plot(range(1, EPOCHS+1), val_loss_epochs, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()