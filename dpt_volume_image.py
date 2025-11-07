import cv2
import torch
import numpy as np
import open3d as o3d
import json
from ultralytics import YOLO
import os
from datetime import datetime
import csv
from PIL import Image # <--- [BAGIAN 1] TAMBAHAN IMPORT
from transformers import DPTImageProcessor, DPTForDepthEstimation # <--- [BAGIAN 1] TAMBAHAN IMPORT

# --- KONFIGURASI ---
# Ganti dengan nama file gambar input Anda
INPUT_IMAGE_PATH = "pear_seaweed/ZoneA_175.jpg" 
CALIBRATION_FILE = "calibration.json"
OUTPUT_DIR = "hasil_deteksi_volume_image"

# --- FUNGSI-FUNGSI PEMBANTU ---
# (Fungsi setup_logging, estimate_distance, load_calibration_data, 
# create_point_cloud SAMA PERSIS seperti sebelumnya)

def setup_logging():
    """Menyiapkan folder dan file CSV untuk logging data debug."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(OUTPUT_DIR, f"log_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "depth_maps_relative"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "depth_maps_scaled"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "point_clouds_raw"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "point_clouds_cleaned"), exist_ok=True)
    
    csv_filepath = os.path.join(log_dir, "summary.csv")
    csv_file = open(csv_filepath, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    
    header = [
        "log_prefix", "timestamp", "image_file",
        "obb_width_px", "obb_height_px", "pixel_area",
        "distance_anchor_cm", "avg_relative_depth", "scale_constant_k", # Ganti nama kolom
        "real_width_cm", "real_height_cm", "real_area_cm2",
        "pcd_points_before_clean", "pcd_points_after_clean",
        "raw_volume_cm3", "correction_factor", "corrected_volume_cm3",
        "relative_depth_map_file", "scaled_depth_map_file", 
        "point_cloud_raw_file", "point_cloud_cleaned_file"
    ]
    csv_writer.writerow(header)
    print(f"[INFO] Logging diaktifkan. Data akan disimpan di folder: {log_dir}")
    return csv_writer, csv_file, log_dir

def estimate_distance(w, h):
    size = w + h
    if size == 0: return float('inf')
    a = 21550; b = 16.159
    distance = a / size + b
    return round(distance, 2)

def load_calibration_data(filepath):
    try:
        with open(filepath, 'r') as f: return np.array(json.load(f)['camera_matrix'])
    except FileNotFoundError:
        print(f"ERROR: File kalibrasi '{filepath}' tidak ditemukan."); return None

# --- [BAGIAN 2] FUNGSI load_models DIUBAH TOTAL ---
def load_models():
    """Memuat model YOLO dan DPT (pengganti MiDaS)."""
    print("[INFO] Memuat model...")
    
    # 1. Muat YOLO
    yolo_model = YOLO("pear_seaweed.pt")
    
    # 2. Muat DPT (Pengganti MiDaS)
    print("[INFO] Memuat model DPT (Intel/dpt-large)... Ini mungkin butuh waktu saat pertama kali.")
    dpt_model_name = "Intel/dpt-large"
    try:
        dpt_processor = DPTImageProcessor.from_pretrained(dpt_model_name)
        dpt_model = DPTForDepthEstimation.from_pretrained(dpt_model_name)
    except Exception as e:
        print(f"[ERROR] Gagal memuat model DPT dari Hugging Face: {e}")
        print("Pastikan Anda memiliki koneksi internet saat pertama kali menjalankan.")
        return None, None, None, None

    # 3. Tentukan Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[INFO] Menggunakan device: {device}")
    
    # 4. Pindahkan model ke device
    yolo_model.to(device)
    dpt_model.to(device)
    
    # 5. Set model ke mode evaluasi
    dpt_model.eval() 
    
    print("[INFO] Semua model berhasil dimuat!")
    # Kembalikan model & processor DPT
    return yolo_model, dpt_model, dpt_processor, device
# --- AKHIR BAGIAN 2 ---

def create_point_cloud(mask, scaled_depth, camera_matrix):
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
    pixels_yx = np.argwhere(mask > 0)
    if len(pixels_yx) == 0: return None
    depths = scaled_depth[pixels_yx[:, 0], pixels_yx[:, 1]]
    valid_indices = depths > 0
    if len(np.where(valid_indices)[0]) == 0: return None
    pixels_yx, depths = pixels_yx[valid_indices], depths[valid_indices]
    z = depths; x = (pixels_yx[:, 1] - cx) * z / fx; y = (pixels_yx[:, 0] - cy) * z / fy
    points3d = np.stack((x, y, z), axis=-1)
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(points3d)
    return pcd

# --- ALUR KERJA UTAMA (MODIFIKASI UNTUK GAMBAR) ---
if __name__ == "__main__":
    # 1. Persiapan
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    camera_matrix = load_calibration_data(CALIBRATION_FILE)
    if camera_matrix is None: exit()
    
    # [BAGIAN 3.A] Ganti nama variabel saat memanggil load_models
    yolo, dpt_model, dpt_processor, device = load_models()
    if yolo is None or dpt_model is None: 
        print("[ERROR] Gagal memuat model, program berhenti.")
        exit()
    
    csv_writer, csv_file, log_dir = setup_logging()
    COLORS = np.random.randint(0, 255, size=(len(yolo.names), 3), dtype=np.uint8)

    # 2. Membaca Gambar
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"[ERROR] File gambar tidak ditemukan: {INPUT_IMAGE_PATH}")
        exit()
        
    frame = cv2.imread(INPUT_IMAGE_PATH)
    if frame is None:
        print(f"[ERROR] Gagal membaca gambar: {INPUT_IMAGE_PATH}")
        exit()
    print(f"[INFO] Memproses gambar: {INPUT_IMAGE_PATH}")

    # --- Mulai Proses Deteksi & Estimasi ---
    annotated_frame = frame.copy()
    
    # A. Deteksi YOLO
    yolo_results = yolo(frame, conf=0.5, verbose=False)
    
    if yolo_results and yolo_results[0].masks is not None:
        
        # --- [BAGIAN 3.B] Estimasi Kedalaman DPT (Pengganti MiDaS) ---
        print("[INFO] Menjalankan estimasi kedalaman DPT pada seluruh gambar...")
        # DPT processor lebih suka input PIL Image dalam format RGB
        img_pil_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Siapkan gambar untuk model
        inputs = dpt_processor(images=img_pil_rgb, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = dpt_model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolasi (resize) depth map ke ukuran gambar asli
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=img_pil_rgb.size[::-1], # (height, width)
            mode="bicubic",
            align_corners=False,
        )
        
        # Ambil output sebagai array numpy. Ini adalah PETA KEDALAMAN (DEPTH)
        relative_depth_map = prediction.squeeze().cpu().numpy()
        print("[INFO] Estimasi kedalaman DPT selesai.")
        # --- AKHIR BAGIAN 3.B ---

        # C. Loop untuk Setiap Objek yang Terdeteksi
        num_objects = len(yolo_results[0].masks.data)
        print(f"[INFO] Terdeteksi {num_objects} objek.")
        
        for i in range(num_objects):
            print(f"   - Memproses Objek {i+1}/{num_objects}...")
            real_width_cm, real_height_cm, real_area_cm2, distance_anchor = 'N/A', 'N/A', 'N/A', 'N/A'
            corrected_volume_cm3 = 0.0
            avg_relative_depth_for_log = 0.0 # Ganti nama var
            scale_constant_k = 0.0

            # C1. Ambil Masker Objek
            mask_tensor = yolo_results[0].masks.data[i]
            mask_np = cv2.resize(mask_tensor.cpu().numpy().astype(np.uint8), 
                                 (frame.shape[1], frame.shape[0]))
            
            # (Opsional) Warnai masker di gambar hasil
            cls_id = int(yolo_results[0].boxes.cls[i])
            color = COLORS[cls_id]
            annotated_frame[mask_np > 0] = cv2.addWeighted(
                annotated_frame[mask_np > 0], 0.5, 
                np.full(annotated_frame[mask_np > 0].shape, color, dtype=np.uint8), 0.5, 0)

            # C2. Analisis Kontur & OBB
            pixel_area = np.sum(mask_np > 0)
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            contour = max(contours, key=cv2.contourArea)
            if len(contour) < 5: continue

            rotated_rect = cv2.minAreaRect(contour)
            box_points = cv2.boxPoints(rotated_rect).astype(np.int32)

            try:
                # C3. Estimasi Jarak & Skala
                (w_obb, h_obb) = rotated_rect[1]
                distance_anchor = estimate_distance(w_obb, h_obb)
                if not (distance_anchor > 0 and distance_anchor < float('inf')): 
                    raise ValueError("Jarak estimasi tidak valid")

                fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
                real_width_cm = (w_obb * distance_anchor) / fx
                real_height_cm = (h_obb * distance_anchor) / fy
                real_area_cm2 = real_width_cm * real_height_cm

                # --- [BAGIAN 3.C] MODIFIKASI LOGIKA PENSKALAAN UNTUK DPT (DEPTH) ---
                
                # DPT menghasilkan 'depth', bukan 'disparity'
                depth_values = relative_depth_map[mask_np > 0]
                valid_depths = depth_values[depth_values > 0]
                if len(valid_depths) == 0: 
                    raise ValueError("Tidak ada data kedalaman (depth) valid pada masker")
                
                # Gunakan rata-rata kedalaman relatif
                avg_relative_depth = np.mean(valid_depths)
                avg_relative_depth_for_log = avg_relative_depth # Simpan untuk logging
                
                # Logika scaling DPT (depth): k = Z_nyata / Z_relatif
                scale_constant_k = distance_anchor / (avg_relative_depth + 1e-6) 
                
                # Logika scaling DPT (depth): Z_scaled = Z_relatif * k
                scaled_depth_map = relative_depth_map * scale_constant_k
                # --- AKHIR BAGIAN 3.C ---

                # C4. Buat Point Cloud & Hitung Volume
                pcd = create_point_cloud(mask_np, scaled_depth_map, camera_matrix)
                if not pcd or len(pcd.points) <= 100: 
                    raise ValueError(f"Point cloud terlalu sedikit ({len(pcd.points) if pcd else 0} poin)")
                
                pcd_before = len(pcd.points)
                cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
                cleaned_pcd = pcd.select_by_index(ind)
                pcd_after = len(cleaned_pcd.points)
                
                if pcd_after <= 100: 
                    raise ValueError(f"Point cloud bersih terlalu sedikit ({pcd_after} poin)")

                hull, _ = cleaned_pcd.compute_convex_hull()
                raw_volume_cm3 = hull.get_volume()
                correction_factor = 1.58 # Anda bisa sesuaikan ini
                corrected_volume_cm3 = raw_volume_cm3 / correction_factor

                # C5. Logging Data (Simpan ke Disk)
                log_prefix = f"img_{os.path.basename(INPUT_IMAGE_PATH).split('.')[0]}_obj{i}"
                rel_depth_file = f"{log_prefix}_rel_depth.npy"
                scaled_depth_file = f"{log_prefix}_scaled_depth.npy"
                raw_pcd_file = f"{log_prefix}_pcd_raw.npy"
                clean_pcd_file = f"{log_prefix}_pcd_cleaned.npy"

                np.save(os.path.join(log_dir, "depth_maps_relative", rel_depth_file), relative_depth_map)
                np.save(os.path.join(log_dir, "depth_maps_scaled", scaled_depth_file), scaled_depth_map)
                np.save(os.path.join(log_dir, "point_clouds_raw", raw_pcd_file), np.asarray(pcd.points))
                np.save(os.path.join(log_dir, "point_clouds_cleaned", clean_pcd_file), np.asarray(cleaned_pcd.points))

                log_row = [
                    log_prefix, datetime.now().isoformat(), os.path.basename(INPUT_IMAGE_PATH),
                    f"{w_obb:.2f}", f"{h_obb:.2f}", pixel_area,
                    f"{distance_anchor:.2f}", 
                    f"{avg_relative_depth_for_log:.4f}", # Gunakan var yg disimpan
                    f"{scale_constant_k:.4f}", # Tambah presisi
                    f"{real_width_cm:.2f}", f"{real_height_cm:.2f}", f"{real_area_cm2:.2f}",
                    pcd_before, pcd_after,
                    f"{raw_volume_cm3:.2f}", f"{correction_factor}", f"{corrected_volume_cm3:.2f}",
                    rel_depth_file, scaled_depth_file, raw_pcd_file, clean_pcd_file
                ]
                csv_writer.writerow(log_row)
                print(f"     -> Sukses! Volume: {corrected_volume_cm3:.2f} cm3")

            except Exception as e:
                print(f"     -> Gagal memproses objek {i}: {e}")
                corrected_volume_cm3 = 0.0 # Reset jika gagal

            # C6. Gambar Info di Frame
            cv2.polylines(annotated_frame, [box_points], isClosed=True, color=(0, 255, 0), thickness=3)

            # --- [PERBAIKAN] Parameter Teks Baru ---
            # Kita naikkan skala dan ketebalan font
            FONT_SCALE = 2.8  # (Ini 4x lebih besar dari 0.7)
            FONT_THICKNESS = 6 # (Ini 3x lebih tebal dari 2, agar proporsional)
            FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
            
            # Ambil tinggi teks untuk spasi dinamis
            (dummy_w, dummy_h), _ = cv2.getTextSize("Test", FONT_FACE, FONT_SCALE, FONT_THICKNESS)
            LINE_SPACING = dummy_h + 15 # Spasi = tinggi teks + 15px padding
            # --- [AKHIR PERBAIKAN] ---
            
            # Teks info yang akan ditampilkan
            info_text = [
                f"Vol: {corrected_volume_cm3:.2f} cm3",
                f"Area: {real_area_cm2:.1f} cm2" if isinstance(real_area_cm2, float) else "",
                f"Dist: {distance_anchor:.1f} cm" if isinstance(distance_anchor, float) else ""
            ]
            
            # Posisi teks (di dekat objek)
            text_x = box_points[:, 0].min()
            
            # Tentukan posisi Y awal (baseline baris pertama)
            text_y_start = box_points[:, 1].min() - 10 # 10px di atas box
            
            for j, line in enumerate(info_text):
                 if line: # Hanya gambar jika teks tidak kosong
                    
                    # [PERBAIKAN] Gunakan LINE_SPACING dinamis
                    y_pos = text_y_start - (j * LINE_SPACING)
                    
                    # [PERBAIKAN] Cek agar bagian ATAS teks tidak keluar frame
                    # (dummy_h adalah tinggi teks)
                    if (y_pos - dummy_h) < 30: 
                        # Jika keluar, pindah ke BAWAH box
                        text_y_start = box_points[:, 1].max() + 30 
                        y_pos = text_y_start + (j * LINE_SPACING)
                     
                    # [PERBAIKAN] Dapatkan ukuran teks yang *spesifik* untuk baris ini
                    (text_w, text_h), _ = cv2.getTextSize(line, FONT_FACE, FONT_SCALE, FONT_THICKNESS)
                    
                    # [PERBAIKAN] Gambar background hitam berdasarkan y_pos dan text_h
                    # y_pos adalah baseline, jadi kotak atasnya adalah y_pos - text_h
                    cv2.rectangle(annotated_frame, (text_x - 5, int(y_pos) - text_h - 10), 
                                  (text_x + text_w + 5, int(y_pos) + 10), (0,0,0), -1)
                                  
                    # [PERBAIKAN] Gambar teks dengan parameter baru
                    cv2.putText(annotated_frame, line, (text_x, int(y_pos)), 
                                FONT_FACE, FONT_SCALE, (0, 255, 255), FONT_THICKNESS)

    else:
        print("[INFO] Tidak ada objek yang terdeteksi oleh YOLO.")

    # 3. Simpan & Tampilkan Hasil Akhir
    output_image_path = os.path.join(OUTPUT_DIR, f"result_{os.path.basename(INPUT_IMAGE_PATH)}")
    cv2.imwrite(output_image_path, annotated_frame)
    print(f"[INFO] Selesai. Gambar hasil disimpan di: {output_image_path}")
    
    csv_file.close()
    
    # Tampilkan hasil (tekan sembarang tombol untuk menutup)
    # Resize jika gambar terlalu besar untuk layar
    h_display, w_display = annotated_frame.shape[:2]
    max_h = 800
    if h_display > max_h:
        scale_ratio = max_h / h_display
        w_display_new = int(w_display * scale_ratio)
        annotated_frame_display = cv2.resize(annotated_frame, (w_display_new, max_h))
    else:
        annotated_frame_display = annotated_frame

    cv2.imshow("Hasil Estimasi Volume (DPT)", annotated_frame_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()