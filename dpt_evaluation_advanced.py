"""
EVALUASI SEMUA MODEL ADVANCED TERHADAP SEMUA DATA
==================================================
Evaluasi 7 model: RF, XGBoost, GBDT, CatBoost, CNN, LSTM, Transformer
"""

import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# ======================= KONFIGURASI =======================
FEATURES_CSV = "features_extracted.csv"
GROUND_TRUTH_FILE = "ground_truth_berat_kering.xlsx"
MODEL_DIR = "trained_models_advanced"
OUTPUT_CSV = "trained_models_advanced/evaluation_results_advanced_all_data.csv"
OUTPUT_METRICS_CSV = "trained_models_advanced/evaluation_metrics_advanced_summary.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================================

# ==================== DEEP LEARNING MODEL DEFINITIONS ====================
# (Copy exact same architecture from training)

class CNNRegressor(nn.Module):
    def __init__(self, input_size):
        super(CNNRegressor, self).__init__()
        self.input_size = input_size
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
        x = x.view(-1, 1, self.conv_height, self.conv_width)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x.squeeze()


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=1,
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
        x = x.unsqueeze(-1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc_layers(x)
        return x.squeeze()


class TransformerRegressor(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2):
        super(TransformerRegressor, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, input_size, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
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
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        x = self.fc_layers(x)
        return x.squeeze()


# ==================== EVALUATOR CLASS ====================

class AdvancedModelEvaluator:
    def __init__(self, model_dir):
        print("[INFO] Memuat model dan scaler...")
        
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        
        # Load traditional ML models
        self.ml_models = {}
        ml_model_files = ['random_forest.pkl', 'xgboost.pkl', 'gbdt.pkl', 'catboost.pkl']
        
        for model_file in ml_model_files:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                model_name = model_file.replace('.pkl', '').replace('_', ' ').title().replace(' ', '_')
                self.ml_models[model_name] = joblib.load(model_path)
                print(f"  ✓ {model_name} loaded")
        
        # Load DL models
        self.dl_models = {}
        self.feature_names = [
            'area_cm2', 'panjang_cm', 'lebar_cm', 'jarak_kamera_cm',
            'ketebalan_cm', 'volume_cm3', 'aspect_ratio', 'perimeter_cm',
            'solidity', 'compactness'
        ]
        input_size = len(self.feature_names)
        
        # CNN
        cnn_path = os.path.join(model_dir, 'cnn_model.pth')
        if os.path.exists(cnn_path):
            cnn_model = CNNRegressor(input_size).to(DEVICE)
            cnn_model.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
            cnn_model.eval()
            self.dl_models['CNN'] = cnn_model
            print(f"  ✓ CNN loaded")
        
        # LSTM
        lstm_path = os.path.join(model_dir, 'lstm_model.pth')
        if os.path.exists(lstm_path):
            lstm_model = LSTMRegressor(input_size).to(DEVICE)
            lstm_model.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
            lstm_model.eval()
            self.dl_models['LSTM'] = lstm_model
            print(f"  ✓ LSTM loaded")
        
        # Transformer
        transformer_path = os.path.join(model_dir, 'transformer_model.pth')
        if os.path.exists(transformer_path):
            transformer_model = TransformerRegressor(input_size).to(DEVICE)
            transformer_model.load_state_dict(torch.load(transformer_path, map_location=DEVICE))
            transformer_model.eval()
            self.dl_models['Transformer'] = transformer_model
            print(f"  ✓ Transformer loaded")
        
        print(f"[INFO] Total {len(self.ml_models) + len(self.dl_models)} model berhasil dimuat!")
    
    def load_data(self, features_csv, ground_truth_file):
        print("\n[INFO] Memuat data...")
        
        try:
            df_features = pd.read_csv(features_csv, sep=',')
            if len(df_features.columns) == 1:
                df_features = pd.read_csv(features_csv, sep=';')
        except Exception as e:
            raise
        
        print(f"[INFO] Fitur dimuat: {len(df_features)} sampel")
        
        file_ext = os.path.splitext(ground_truth_file)[1].lower()
        if file_ext in ['.xlsx', '.xls']:
            df_gt = pd.read_excel(ground_truth_file)
        else:
            df_gt = pd.read_csv(ground_truth_file)
        
        print(f"[INFO] Ground truth dimuat: {len(df_gt)} sampel")
        
        if 'berat_kering' in df_gt.columns and 'berat_kering_gram' not in df_gt.columns:
            df_gt.rename(columns={'berat_kering': 'berat_kering_gram'}, inplace=True)
        
        df_merged = pd.merge(df_features, df_gt, on=['image_file', 'object_id'], how='inner')
        print(f"[INFO] Data setelah merge: {len(df_merged)} sampel")
        
        return df_merged
    
    def predict_all_models(self, df):
        print("\n[INFO] Melakukan prediksi dengan semua model...")
        
        X = df[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        y_true = df['berat_kering_gram'].values
        
        predictions = {}
        
        # ML models
        for name, model in self.ml_models.items():
            print(f"  - Prediksi dengan {name}...")
            predictions[name] = model.predict(X_scaled)
        
        # DL models
        X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)
        for name, model in self.dl_models.items():
            print(f"  - Prediksi dengan {name}...")
            with torch.no_grad():
                y_pred = model(X_tensor).cpu().numpy()
            predictions[name] = y_pred
        
        return predictions, y_true
    
    def calculate_errors(self, y_true, y_pred):
        ae = np.abs(y_true - y_pred)
        se = (y_true - y_pred) ** 2
        pe = np.abs((y_true - y_pred) / (y_true + 1e-8)) * 100
        return ae, se, pe
    
    def create_results_dataframe(self, df_original, predictions, y_true):
        print("\n[INFO] Menyusun hasil evaluasi...")
        
        df_results = df_original.copy()
        df_results['Ground_Truth_gram'] = y_true
        
        # Add predictions
        for name, y_pred in predictions.items():
            df_results[f'Pred_{name}_gram'] = np.round(y_pred, 2)
        
        # Add errors
        for name, y_pred in predictions.items():
            ae, se, pe = self.calculate_errors(y_true, y_pred)
            df_results[f'AE_{name}_gram'] = np.round(ae, 2)
            df_results[f'SE_{name}_gram2'] = np.round(se, 2)
            df_results[f'APE_{name}_percent'] = np.round(pe, 2)
        
        return df_results
    
    def calculate_metrics_summary(self, predictions, y_true):
        print("\n[INFO] Menghitung metrik evaluasi...")
        
        metrics_summary = []
        
        for name, y_pred in predictions.items():
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            
            ae = np.abs(y_true - y_pred)
            
            metrics_summary.append({
                'Model': name,
                'MAE (gram)': round(mae, 3),
                'MSE (gram²)': round(mse, 3),
                'RMSE (gram)': round(rmse, 3),
                'R² Score': round(r2, 4),
                'MAPE (%)': round(mape, 2),
                'Min_Error (gram)': round(np.min(ae), 2),
                'Max_Error (gram)': round(np.max(ae), 2),
                'Median_Error (gram)': round(np.median(ae), 2),
                'Std_Error (gram)': round(np.std(ae), 2),
                'Samples': len(y_true)
            })
            
            print(f"\n{name}:")
            print(f"  MAE:  {mae:.3f} gram")
            print(f"  RMSE: {rmse:.3f} gram")
            print(f"  R²:   {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")
        
        return pd.DataFrame(metrics_summary)
    
    def save_results(self, df_results, df_metrics):
        print(f"\n[INFO] Menyimpan hasil...")
        
        df_results.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Hasil lengkap: {OUTPUT_CSV}")
        print(f"   Kolom: {len(df_results.columns)}, Baris: {len(df_results)}")
        
        df_metrics.to_csv(OUTPUT_METRICS_CSV, index=False)
        print(f"✅ Metrik summary: {OUTPUT_METRICS_CSV}")


def main():
    print("="*70)
    print("EVALUASI MODEL ADVANCED TERHADAP SEMUA DATA")
    print("="*70)
    
    evaluator = AdvancedModelEvaluator(MODEL_DIR)
    df = evaluator.load_data(FEATURES_CSV, GROUND_TRUTH_FILE)
    predictions, y_true = evaluator.predict_all_models(df)
    df_results = evaluator.create_results_dataframe(df, predictions, y_true)
    df_metrics = evaluator.calculate_metrics_summary(predictions, y_true)
    evaluator.save_results(df_results, df_metrics)
    
    print("\n" + "="*70)
    print("📊 METRIK EVALUASI SUMMARY:")
    print("="*70)
    print(df_metrics.to_string(index=False))
    
    print("\n" + "="*70)
    print("🏆 RANKING MODEL (berdasarkan MAE):")
    print("="*70)
    df_ranked = df_metrics.sort_values('MAE (gram)')
    for idx, (_, row) in enumerate(df_ranked.iterrows(), 1):
        print(f"{idx}. {row['Model']:<20} MAE={row['MAE (gram)']:.3f}g  R²={row['R² Score']:.4f}")
    
    print("\n✅ EVALUASI SELESAI!")


if __name__ == "__main__":
    main()