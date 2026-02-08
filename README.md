# SAMPL GED Predictor & Generated Data Model

This project implements Deep Learning models (DeepSets and Sinkhorn Networks) to predict similarity and Graph Edit Distance (GED) between sets of points or molecular graphs.

## Project Structure

### 1. Generated Data Model (Python Scripts & Notebook)
These scripts generate synthetic point set data and train models to predict set similarity.

- **`Similarity_predictor_generated_data.ipynb`**: A comprehensive notebook that combines data generation, model definition (DeepSets & Sinkhorn), and training into a single interactive file.
- **`Data_utils.py`**: Utilities for generating synthetic datasets (pairs of point sets with known similarity) and PyTorch Dataset classes.
- **`DeepSets.py`**: Defines the DeepSets architecture (Invariant deep neural network for sets).
- **`SinkhornSetNet.py`**: Defines the Sinkhorn Network architecture for learning soft matching between sets.
- **`DeepSet.py`**: Training script for the DeepSets model on synthetic data.
- **`Sinkhorn.py`**: Training script for the Sinkhorn model on synthetic data.

### 2. Molecular Graph Predictor (Jupyter Notebook)
- **`SAMPL_Ged_predictor.ipynb`**: A notebook that processes the SAMPL molecule dataset. It:
  - Converts SMILES to NetworkX graphs.
  - Computes approximate GED using bipartite matching.
  - Trains Sinkhorn and DeepSets models to predict GED between molecular graphs.

## Requirements

To run these files, you need Python installed with the following libraries:

```bash
pip install torch numpy matplotlib scipy scikit-learn pandas networkx rdkit
```

*Note: `rdkit` is required for the notebook to parse SMILES strings.*

## Configuration & Changes Needed

Before running the Python scripts (`DeepSet.py` and `Sinkhorn.py`), you **must** update the data paths.

1. Open `DeepSet.py` and `Sinkhorn.py`.
2. Locate the following lines (approx. lines 15-18):
   ```python
   #Change paths as needed
   TRAIN_PATH = 'C:/Users/PC1/Downloads/train_similarity_final.npz'
   TEST_PATH  = 'C:/Users/PC1/Downloads/test_similarity_final.npz'
   #Change paths as needed
   ```
3. Change these paths to a valid location on your machine, or use a relative path like:
   ```python
   TRAIN_PATH = 'train_similarity_final.npz'
   TEST_PATH  = 'test_similarity_final.npz'
   ```

## Usage

### Running the Synthetic Data Models

To train the DeepSets model:
```bash
python DeepSet.py
```

To train the Sinkhorn model:
```bash
python Sinkhorn.py
```
*These scripts will automatically generate the synthetic data if it doesn't exist at the specified path.*

### Running the Notebooks

**Molecular Graph Predictor:**
1. Ensure the dataset file `SAMPL.csv` is located in the same directory as the notebook.
2. Open `SAMPL_Ged_predictor.ipynb` in VS Code or Jupyter.
3. Run the cells sequentially to preprocess data, train the models, and visualize results.

**Generated Data Predictor:**
1. Open `Similarity_predictor_generated_data.ipynb`.
2. This notebook is self-contained. It generates its own synthetic data.
3. Run the cells to generate data, define models, and train/evaluate both DeepSets and Sinkhorn architectures.
