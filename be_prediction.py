import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import math
import time
import psutil

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

def parse_mass_mas20(path):
    """
    Parse AME mass.mas20 file into DataFrame with Z, N, A, BE_exp columns.
    Returns DataFrame with binding energy in keV.
    """
    data = []
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines until we find the data section
    start_idx = 0
    for i, line in enumerate(lines):
        if '1N-Z    N    Z   A  EL    O     MASS EXCESS           BINDING ENERGY/A        BETA-DECAY ENERGY               ATOMIC MASS' in line:
            start_idx = i + 3 
            break
    
    print(f"Starting to parse from line {start_idx}")
    
    # Parse each line
    for i in range(start_idx, len(lines)):
        line = lines[i]
        
        # Skip short lines or obvious header/comment lines
        if len(line) < 100 or line.strip().startswith('*') or 'A=   0 TO' in line:
            continue
            
        try:
            # Parse AME format more carefully
            # The format is: cc NZ N Z A el o mass unc binding unc ...
            
            # Extract control character and position info
            cc = line[0] if len(line) > 0 else ''
            
            # Extract N, Z, A from fixed positions
            nz_str = line[1:4].strip() if len(line) > 4 else ''
            n_str = line[4:9].strip() if len(line) > 9 else ''
            z_str = line[9:14].strip() if len(line) > 14 else ''
            a_str = line[14:19].strip() if len(line) > 19 else ''
            
            # Skip if basic info is missing
            if not n_str or not z_str or not a_str:
                continue
                
            # Try to parse numbers
            try:
                n = int(n_str)
                z = int(z_str)
                a = int(a_str)
            except ValueError:
                continue
            
            # Extract binding energy per nucleon (around position 54-67)
            be_per_a_str = line[54:67].strip() if len(line) > 67 else ''
            
            # Clean the binding energy string and handle estimated values
            if be_per_a_str:
                # Replace # with decimal point for estimated values
                be_per_a_clean = be_per_a_str.replace('#', '')
                
                # Skip if completely missing or just asterisk
                if not be_per_a_clean or be_per_a_clean == '*' or be_per_a_clean == '':
                    continue
                
                try:
                    be_per_a = float(be_per_a_clean)
                    # Total binding energy = BE/A * A
                    # be_total = be_per_a * a
                    
                    data.append({
                        'Z': z,
                        'N': n,
                        'A': a,
                        'BE_exp': be_per_a
                    })
                    
                except ValueError:
                    # Skip if we can't parse the binding energy
                    continue
                    
        except (IndexError, ValueError) as e:
            continue
    
    df = pd.DataFrame(data)
    print(f"Parsed {len(df)} nuclei from {path}")
    
    # Show some statistics
    if len(df) > 0:
        print(f"Z range: {df['Z'].min()} to {df['Z'].max()}")
        print(f"A range: {df['A'].min()} to {df['A'].max()}")
    
    return df

def make_features(df):
    """
    Build simple features from the DataFrame using only Z, N, and A.
    Returns X (features DataFrame) and y (target array in MeV for better scaling).
    """
    X = df.copy()
    
    # Keep only basic nuclear features: Z, N, A
    feature_cols = ['Z', 'N', 'A']
    
    X = X[feature_cols]
    y = df['BE_exp'].values / 1000.0  # Convert to MeV 
    
    print(f"Created {len(feature_cols)} features for {len(X)} nuclei")
    print(f"Features: {feature_cols}")
    print(f"Target range: {y.min():.1f} to {y.max():.1f} MeV")
    return X, y

class BaseNN(nn.Module):
    """Simple Feed Forward Neural Network for binding energy prediction."""
    
    def __init__(self, input_dim, hidden_sizes=[64, 48], dropout=0.1, use_batch_norm=True):
        super(BaseNN, self).__init__()
        
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Use LayerNorm instead of BatchNorm for small datasets
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            else:
                layers.append(nn.LayerNorm(hidden_size))
                
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Final output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x).squeeze()

def train_nn(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=38, lr=0.001):
    """
    Train neural network with early stopping and learning rate scheduling.
    Returns trained model and training history.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Adjust batch size if dataset is too small
    min_batch_size = 8  # Minimum for BatchNorm to work reliably
    actual_batch_size = min(batch_size, max(min_batch_size, len(X_train) // 4))
    
    if actual_batch_size < min_batch_size and len(X_train) < min_batch_size * 2:
        print(f"Warning: Small dataset ({len(X_train)} samples). Using batch size {actual_batch_size}")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Create data loaders with drop_last=True to avoid single-sample batches
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True, drop_last=True)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training history
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()  
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor).item()
        
        train_loss = np.mean(train_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, history

def predict_nn(model, X):
    """Make predictions with trained neural network."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        pred = model(X_tensor).cpu().numpy()
    
    return pred

def create_train_val_test_split(X, y, test_size=0.1, random_state=42):
    """
    Creates a proper train/val/test split.
    - First split: remove test set (10% of data)
    - Second split: from remaining 90%, create 80%/20% train/val split
    
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: separate test set (10% of total data)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: from remaining 90%, create 80%/20% train/val split
    # This gives us: 72% train, 18% val, 10% test of original data
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=random_state
    )
    
    print(f"Data split:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_nn_ensemble(X_train, X_val, y_train, y_val, n_models=50, epochs=200, batch_size=38, lr=0.001):
    """
    Train ensemble of neural networks using Monte Carlo cross validation.
    Each model is trained on a different random subset of the train/val data.
    Test data is never seen during training.
    
    Monte Carlo approach: For each model, randomly reassign the train+val data into 
    new 80/20 train/val splits, maintaining the same proportions as the original split.
    
    Returns ensemble predictions for train and val sets, models, scaler, and training histories.
    """
    print(f"\nTraining ensemble of {n_models} neural networks with Monte Carlo CV...")
    
    # Combine train and val for Monte Carlo sampling (but keep original val for final evaluation)
    X_trainval_combined = np.vstack([X_train, X_val])
    y_trainval_combined = np.hstack([y_train, y_val])
    
    # Scale the combined train+val dataset
    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval_combined)
    
    # Calculate the size for Monte Carlo train/val splits
    # We want to maintain approximately 80/20 split of the train+val data
    total_trainval_size = len(X_trainval_combined)
    mc_train_size = int(0.8 * total_trainval_size)  # 80% of train+val for MC training
    mc_val_size = total_trainval_size - mc_train_size  # 20% of train+val for MC validation
    
    # Determine if we should use BatchNorm or LayerNorm based on MC training set size
    use_batch_norm = mc_train_size > 100  # Use LayerNorm for small training sets
    norm_type = "BatchNorm" if use_batch_norm else "LayerNorm"
    print(f"  Using {norm_type} (MC training set size: {mc_train_size})")
    print(f"  Total train+val: {total_trainval_size}, MC train: {mc_train_size}, MC val: {mc_val_size}")
    
    models = []
    ensemble_predictions_train = np.zeros((len(X_train), n_models))
    ensemble_predictions_val = np.zeros((len(X_val), n_models))
    model_scores = []
    training_histories = []
    
    # Scale original train and val sets for predictions
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    for model_idx in range(n_models):
        print(f"  Training model {model_idx + 1}/{n_models}...")
        
        # Monte Carlo: create random train/val split from the combined train+val data
        np.random.seed(42 + model_idx)  # Different seed for each model
        
        # Use 80% for training, 20% for validation (of the train+val combined data)
        X_train_mc, X_val_mc, y_train_mc, y_val_mc = train_test_split(
            X_trainval_scaled, y_trainval_combined, 
            train_size=mc_train_size,  # Use exact size instead of ratio
            random_state=42 + model_idx)
        
        # Create and train model with appropriate normalization
        model = BaseNN(input_dim=X_trainval_scaled.shape[1], use_batch_norm=use_batch_norm)
        model, history = train_nn(model, X_train_mc, y_train_mc, X_val_mc, y_val_mc, 
                                 epochs=epochs, batch_size=batch_size, lr=lr)
        
        # Store training history
        training_histories.append(history)
        
        # Make predictions on original train and val sets
        train_pred = predict_nn(model, X_train_scaled)
        val_pred = predict_nn(model, X_val_scaled)
        
        ensemble_predictions_train[:, model_idx] = train_pred
        ensemble_predictions_val[:, model_idx] = val_pred
        
        # Calculate validation score for this model (on the MC validation set)
        mc_val_pred = predict_nn(model, X_val_mc)
        val_rmse = math.sqrt(mean_squared_error(y_val_mc, mc_val_pred))
        model_scores.append(val_rmse)
        
        models.append(model)
        if model_idx < 5 or (model_idx + 1) % 10 == 0:  # Reduce output for large ensembles
            print(f"    Model {model_idx + 1} MC validation RMSE: {val_rmse * 1000:.2f} keV ({val_rmse:.3f} MeV)")
            print(f"    Trained on {len(X_train_mc)} samples, validated on {len(X_val_mc)} samples")
    
    # Calculate ensemble predictions (average of all models)
    ensemble_pred_train = np.mean(ensemble_predictions_train, axis=1)
    ensemble_pred_val = np.mean(ensemble_predictions_val, axis=1)
    
    # Calculate ensemble performance on original train and val sets
    ensemble_rmse_train = math.sqrt(mean_squared_error(y_train, ensemble_pred_train))
    ensemble_rmse_val = math.sqrt(mean_squared_error(y_val, ensemble_pred_val))
    
    print(f"\n  Ensemble Results:")
    print(f"    Individual model MC val RMSEs: {[f'{score*1000:.1f}' for score in model_scores[:5]]}... keV (showing first 5)")
    print(f"    Mean individual MC val RMSE: {np.mean(model_scores) * 1000:.2f} ± {np.std(model_scores) * 1000:.2f} keV")
    print(f"    Ensemble train RMSE: {ensemble_rmse_train * 1000:.2f} keV ({ensemble_rmse_train:.3f} MeV)")
    print(f"    Ensemble val RMSE: {ensemble_rmse_val * 1000:.2f} keV ({ensemble_rmse_val:.3f} MeV)")
    
    return ensemble_pred_train, ensemble_pred_val, models, scaler, training_histories

def predict_ensemble_on_test(models, scaler, X_test):
    """
    Make ensemble predictions on test set using trained models.
    """
    X_test_scaled = scaler.transform(X_test)
    test_predictions = np.zeros((len(X_test), len(models)))
    
    for i, model in enumerate(models):
        test_predictions[:, i] = predict_nn(model, X_test_scaled)
    
    # Average predictions from all models
    ensemble_pred_test = np.mean(test_predictions, axis=1)
    return ensemble_pred_test

def measure_training_time(func, *args, **kwargs):
    """
    Measure different types of timing for a training function.
    
    Returns:
        result: The function's return value
        timing_info: Dictionary with timing measurements
    """
    # Start measurements
    start_wall = time.time()
    start_cpu = time.process_time()
    
    # Get process info for memory tracking
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run the function
    result = func(*args, **kwargs)
    
    # End measurements
    end_wall = time.time()
    end_cpu = time.process_time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate times
    wall_time = end_wall - start_wall
    cpu_time = end_cpu - start_cpu
    memory_used = end_memory - start_memory
    
    timing_info = {
        'wall_time': wall_time,
        'cpu_time': cpu_time,
        'cpu_efficiency': (cpu_time / wall_time * 100) if wall_time > 0 else 0,
        'memory_used_mb': memory_used,
        'wall_time_formatted': f"{wall_time:.1f}s ({wall_time/60:.1f}min)" if wall_time < 3600 else f"{wall_time/3600:.1f}h",
        'cpu_time_formatted': f"{cpu_time:.1f}s ({cpu_time/60:.1f}min)" if cpu_time < 3600 else f"{cpu_time/3600:.1f}h"
    }
    
    return result, timing_info

def train_xgb_on_residuals(X_train, resid_train, X_val, resid_val):
    """
    Train XGBoost on residuals with early stopping.
    
    Args:
        X_train, X_val: SCALED feature arrays (already transformed by StandardScaler)
        resid_train, resid_val: Residuals in MeV (y_true - y_pred_ensemble)
    """
    
    # Validate inputs
    assert X_train.shape[1] == X_val.shape[1], "Feature dimensions must match"
    assert len(X_train) == len(resid_train), "X_train and resid_train must have same length"
    assert len(X_val) == len(resid_val), "X_val and resid_val must have same length"
    
    print(f"  Training XGBoost on {len(X_train)} train samples, {len(X_val)} val samples")
    print(f"  Residual ranges - Train: [{resid_train.min():.3f}, {resid_train.max():.3f}] MeV")
    print(f"                   Val: [{resid_val.min():.3f}, {resid_val.max():.3f}] MeV")
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    xgb_model.fit(
        X_train, resid_train,
        eval_set=[(X_val, resid_val)],
        verbose=False
    )
    
    return xgb_model

def evaluate_predictions(y_true, y_pred, model_name):
    """Compute and print evaluation metrics (input assumed to be in MeV)."""
    rmse_mev = math.sqrt(mean_squared_error(y_true, y_pred))
    rmse_kev = rmse_mev * 1000.0
    mae_mev = mean_absolute_error(y_true, y_pred)
    mae_kev = mae_mev * 1000.0
    
    print(f"{model_name} RMSE: {rmse_kev:.2f} keV ({rmse_mev:.3f} MeV)")
    print(f"{model_name} MAE:  {mae_kev:.2f} keV ({mae_mev:.3f} MeV)")
    
    return rmse_kev, mae_kev

def evaluate_final_model(y_train, y_val, y_test, 
                        ensemble_pred_train, ensemble_pred_val, y_test_pred_nn,
                        y_test_pred_stacked):
    """
    Comprehensive evaluation of the final stacked model showing all stages.
    This ensures we only evaluate the test set once at the very end.
    """
    print("COMPREHENSIVE MODEL EVALUATION")
    
    # Ensemble NN Performance
    print("\n1. NEURAL NETWORK ENSEMBLE PERFORMANCE:")
    ensemble_rmse_train = math.sqrt(mean_squared_error(y_train, ensemble_pred_train))
    ensemble_rmse_val = math.sqrt(mean_squared_error(y_val, ensemble_pred_val))
    ensemble_rmse_test = math.sqrt(mean_squared_error(y_test, y_test_pred_nn))
    
    print(f"   Train RMSE: {ensemble_rmse_train * 1000:.2f} keV ({ensemble_rmse_train:.3f} MeV)")
    print(f"   Val RMSE:   {ensemble_rmse_val * 1000:.2f} keV ({ensemble_rmse_val:.3f} MeV)")
    print(f"   Test RMSE:  {ensemble_rmse_test * 1000:.2f} keV ({ensemble_rmse_test:.3f} MeV)")
    
    # Final Stacked Model Performance  
    print("\n2. FINAL STACKED MODEL PERFORMANCE (NN + XGBoost):")
    stacked_rmse_test = math.sqrt(mean_squared_error(y_test, y_test_pred_stacked))
    stacked_mae_test = mean_absolute_error(y_test, y_test_pred_stacked)
    
    print(f"   Test RMSE: {stacked_rmse_test * 1000:.2f} keV ({stacked_rmse_test:.3f} MeV)")
    print(f"   Test MAE:  {stacked_mae_test * 1000:.2f} keV ({stacked_mae_test:.3f} MeV)")
    
    # Improvement Analysis
    print("\n3. STACKING IMPROVEMENT:")
    improvement_rmse = (ensemble_rmse_test - stacked_rmse_test) / ensemble_rmse_test * 100
    print(f"   RMSE improvement: {improvement_rmse:.2f}%")
    print(f"   ({ensemble_rmse_test * 1000:.2f} → {stacked_rmse_test * 1000:.2f} keV)")
    
    # Data Usage Summary
    print(f"\n4. DATA USAGE SUMMARY:")
    total_samples = len(y_train) + len(y_val) + len(y_test)
    print(f"   Total dataset: {total_samples} nuclei")
    print(f"   Train: {len(y_train)} ({len(y_train)/total_samples*100:.1f}%) - Used for ensemble training")
    print(f"   Val:   {len(y_val)} ({len(y_val)/total_samples*100:.1f}%) - Used for ensemble training")
    print(f"   Test:  {len(y_test)} ({len(y_test)/total_samples*100:.1f}%) - HELD OUT (never seen during training)")
    print("TEST SET EVALUATION COMPLETE")
    
    return stacked_rmse_test * 1000, stacked_mae_test * 1000

