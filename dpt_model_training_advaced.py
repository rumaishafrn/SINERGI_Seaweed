"""
TRAINING MODEL ADVANCED UNTUK ESTIMASI BIOMASSA RUMPUT LAUT
============================================================
Model yang digunakan:
1. Random Forest (baseline)
2. XGBoost (baseline)
3. GBDT (Gradient Boosting Decision Tree)
4. CatBoost
5. CNN (Convolutional Neural Network - untuk tabular data)
6. LSTM (Long Short-Term Memory)
7. Transformer (untuk sequence/tabular data)
8. Ensemble (kombinasi semua model)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import joblib
import warnings
warnings.filterwarnings('ignore')

# ======================= KONFIGURASI =======================
FEATURES_CSV = "features_extracted.csv"
GROUND_TRUTH_FILE = "ground_truth_berat_kering.xlsx"
OUTPUT_MODEL_DIR = "trained_models_advanced"
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters untuk Deep Learning
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 30  # Early stopping patience

# ===========================================================

# ==================== DEEP LEARNING MODELS ====================

class CNNRegressor(nn.Module):
    """CNN untuk tabular data dengan reshape ke 2D"""
    def __init__(self, input_size):
        super(CNNRegressor, self).__init__()
        self.input_size = input_size
        
        # Reshape input menjadi "image-like" (1 channel, height, width)
        # Untuk 10 fitur, bisa reshape ke (1, 2, 5) atau (1, 10, 1)
        self.conv_height = 2
        self.conv_width = 5
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # Reshape input: (batch, 10) -> (batch, 1, 2, 5)
        x = x.view(-1, 1, self.conv_height, self.conv_width)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x.squeeze()


class LSTMRegressor(nn.Module):
    """LSTM untuk sequence data"""
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=1,  # Treat each feature as a time step
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # Reshape: (batch, features) -> (batch, features, 1) untuk LSTM
        x = x.unsqueeze(-1)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Ambil output dari last time step
        x = lstm_out[:, -1, :]
        x = self.fc_layers(x)
        return x.squeeze()


class TransformerRegressor(nn.Module):
    """Transformer untuk tabular data"""
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2):
        super(TransformerRegressor, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        
        # Embedding layer untuk project input ke d_model
        self.embedding = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, input_size, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc_layers = nn.Sequential(
            nn.Linear(d_model * input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # Reshape: (batch, features) -> (batch, features, 1)
        x = x.unsqueeze(-1)
        
        # Embedding
        x = self.embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Transformer
        x = self.transformer(x)
        
        # Flatten and pass through FC
        x = x.flatten(start_dim=1)
        x = self.fc_layers(x)
        return x.squeeze()


class DeepLearningTrainer:
    """Helper class untuk training deep learning models"""
    
    def __init__(self, model, model_name):
        self.model = model.to(DEVICE)
        self.model_name = model_name
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def fit(self, train_loader, val_loader):
        print(f"\n[INFO] Training {self.model_name}...")
        print(f"Device: {DEVICE}")
        
        for epoch in range(EPOCHS):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Learning rate scheduler
            self.scheduler.step(val_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict()
            else:
                self.patience_counter += 1
                if self.patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()


# ==================== MAIN TRAINER CLASS ====================

class AdvancedBiomassTrainer:
    """Main class untuk training semua model"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.dl_trainers = {}
        self.best_model = None
        self.feature_names = [
            'area_cm2', 'panjang_cm', 'lebar_cm', 'jarak_kamera_cm',
            'ketebalan_cm', 'volume_cm3', 'aspect_ratio', 'perimeter_cm',
            'solidity', 'compactness'
        ]
    
    def load_and_merge_data(self, features_csv, ground_truth_file):
        print("[INFO] Memuat data...")
        
        # Load features
        try:
            df_features = pd.read_csv(features_csv, sep=',')
            if len(df_features.columns) == 1:
                df_features = pd.read_csv(features_csv, sep=';')
        except Exception as e:
            raise
        
        print(f"[INFO] Fitur dimuat: {len(df_features)} sampel")
        
        # Load ground truth
        import os
        file_ext = os.path.splitext(ground_truth_file)[1].lower()
        if file_ext in ['.xlsx', '.xls']:
            df_gt = pd.read_excel(ground_truth_file)
        else:
            df_gt = pd.read_csv(ground_truth_file)
        
        print(f"[INFO] Ground truth dimuat: {len(df_gt)} sampel")
        
        if 'berat_kering' in df_gt.columns and 'berat_kering_gram' not in df_gt.columns:
            df_gt.rename(columns={'berat_kering': 'berat_kering_gram'}, inplace=True)
        
        df_merged = pd.merge(df_features, df_gt, on=['image_file', 'object_id'], how='inner')
        print(f"[INFO] Data setelah merge: {len(df_merged)} sampel\n")
        
        return df_merged
    
    def prepare_data(self, df):
        print("[INFO] Menyiapkan data...")
        
        df = df.dropna()
        X = df[self.feature_names].values
        y = df['berat_kering_gram'].values
        
        # Split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(TEST_SIZE + VALIDATION_SIZE), random_state=RANDOM_STATE
        )
        val_ratio = VALIDATION_SIZE / (TEST_SIZE + VALIDATION_SIZE)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), random_state=RANDOM_STATE
        )
        
        print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        print("\n[1/8] Training Random Forest...")
        rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        print(f"  Validation MAE: {mae:.3f} gram")
        self.models['Random_Forest'] = rf
        return rf
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        print("\n[2/8] Training XGBoost...")
        xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        print(f"  Validation MAE: {mae:.3f} gram")
        self.models['XGBoost'] = xgb_model
        return xgb_model
    
    def train_gbdt(self, X_train, y_train, X_val, y_val):
        print("\n[3/8] Training GBDT (Gradient Boosting)...")
        gbdt = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, 
            subsample=0.8, random_state=RANDOM_STATE
        )
        gbdt.fit(X_train, y_train)
        y_pred = gbdt.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        print(f"  Validation MAE: {mae:.3f} gram")
        self.models['GBDT'] = gbdt
        return gbdt
    
    def train_catboost(self, X_train, y_train, X_val, y_val):
        print("\n[4/8] Training CatBoost...")
        catboost = CatBoostRegressor(
            iterations=300, depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, verbose=False
        )
        catboost.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=False)
        y_pred = catboost.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        print(f"  Validation MAE: {mae:.3f} gram")
        self.models['CatBoost'] = catboost
        return catboost
    
    def train_cnn(self, X_train, y_train, X_val, y_val):
        print("\n[5/8] Training CNN...")
        
        # Prepare DataLoaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        # Train
        cnn_model = CNNRegressor(input_size=len(self.feature_names))
        trainer = DeepLearningTrainer(cnn_model, "CNN")
        trainer.fit(train_loader, val_loader)
        
        self.dl_trainers['CNN'] = trainer
        return trainer
    
    def train_lstm(self, X_train, y_train, X_val, y_val):
        print("\n[6/8] Training LSTM...")
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        lstm_model = LSTMRegressor(input_size=len(self.feature_names))
        trainer = DeepLearningTrainer(lstm_model, "LSTM")
        trainer.fit(train_loader, val_loader)
        
        self.dl_trainers['LSTM'] = trainer
        return trainer
    
    def train_transformer(self, X_train, y_train, X_val, y_val):
        print("\n[7/8] Training Transformer...")
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        transformer_model = TransformerRegressor(input_size=len(self.feature_names))
        trainer = DeepLearningTrainer(transformer_model, "Transformer")
        trainer.fit(train_loader, val_loader)
        
        self.dl_trainers['Transformer'] = trainer
        return trainer
    
    def evaluate_all(self, X_test, y_test):
        print("\n" + "="*70)
        print("EVALUASI PADA TEST SET")
        print("="*70)
        
        results = {}
        
        # Evaluate traditional ML models
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            
            results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
            print(f"\n{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.4f}, MAPE={mape:.2f}%")
        
        # Evaluate DL models
        for name, trainer in self.dl_trainers.items():
            y_pred = trainer.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            
            results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
            print(f"\n{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.4f}, MAPE={mape:.2f}%")
        
        # Best model
        best_name = min(results, key=lambda x: results[x]['MAE'])
        print(f"\n{'='*70}")
        print(f"🏆 BEST MODEL: {best_name} (MAE: {results[best_name]['MAE']:.3f} gram)")
        print("="*70)
        
        return results
    
    def save_models(self):
        import os
        os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
        
        joblib.dump(self.scaler, f'{OUTPUT_MODEL_DIR}/scaler.pkl')
        
        for name, model in self.models.items():
            joblib.dump(model, f'{OUTPUT_MODEL_DIR}/{name.lower()}.pkl')
        
        for name, trainer in self.dl_trainers.items():
            torch.save(trainer.model.state_dict(), f'{OUTPUT_MODEL_DIR}/{name.lower()}_model.pth')
        
        print(f"\n✅ Semua model disimpan di: {OUTPUT_MODEL_DIR}/")


def main():
    print("="*70)
    print("TRAINING MODEL ADVANCED - BIOMASSA RUMPUT LAUT")
    print("="*70)
    
    trainer = AdvancedBiomassTrainer()
    
    # Load data
    df = trainer.load_and_merge_data(FEATURES_CSV, GROUND_TRUTH_FILE)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df)
    
    # Train all models
    trainer.train_random_forest(X_train, y_train, X_val, y_val)
    trainer.train_xgboost(X_train, y_train, X_val, y_val)
    trainer.train_gbdt(X_train, y_train, X_val, y_val)
    trainer.train_catboost(X_train, y_train, X_val, y_val)
    trainer.train_cnn(X_train, y_train, X_val, y_val)
    trainer.train_lstm(X_train, y_train, X_val, y_val)
    trainer.train_transformer(X_train, y_train, X_val, y_val)
    
    # Evaluate
    results = trainer.evaluate_all(X_test, y_test)
    
    # Save
    trainer.save_models()
    
    print("\n✅ TRAINING SELESAI!")


if __name__ == "__main__":
    main()