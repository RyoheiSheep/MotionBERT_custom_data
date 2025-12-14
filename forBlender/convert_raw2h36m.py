import os
import pickle
import numpy as np
import glob
import random
from tqdm import tqdm

# ================= 設定エリア =================
# Rawデータの場所（Blenderで出力したフォルダ）
RAW_DATA_DIR = "D:/MotionBERT_Data/motion3d/raw_blender"

# 出力先（今回の実験用データセット）
OUTPUT_DATASET_DIR = "data/motion3d/experiment_v1_full_norm"

# 学習パラメータ
CLIP_LEN = 243    # MotionBERTの入力フレーム数
STRIDE = 81       # 切り出し間隔
TRAIN_RATIO = 0.9 # Train/Test分割比率

# ==============================================

def normalize_2d_3d_h36m(pose, res_w, res_h, is_3d=False):
    """
    元のH36Mデータローダーのロジックに基づいて3D/2D座標を正規化する。
    
    pose: (T, J, 3) (x, y, [z|conf])
    res_w, res_h: 画像解像度
    is_3d: Trueの場合、3D座標(z軸も含む)として処理。Falseの場合は2D+Confとして処理。
    """
    
    # 座標 (x, y)
    xy = pose[..., :2] 
    
    # 信頼度またはZ座標
    last_dim = pose[..., 2:] 
    
    # 放送(Broadcast)用の係数作成
    # x軸とy軸の正規化は共通
    scale_xy = np.array([res_w, res_w], dtype=np.float32)
    offset_xy = np.array([1.0, res_h / res_w], dtype=np.float32)
    
    # (x, y) 軸の正規化 (2Dと3Dで共通)
    xy_norm = (xy / scale_xy) * 2 - offset_xy

    if is_3d:
        # Z軸の正規化: z_norm = (z / res_w) * 2
        scale_z = np.array([res_w], dtype=np.float32)
        offset_z = np.array([0.0], dtype=np.float32) # 平行移動なし
        z_norm = (last_dim / scale_z) * 2 - offset_z
        
        # 3Dラベルは正規化された(x, y, z)を結合
        return np.concatenate([xy_norm, z_norm], axis=-1)
        
    else:
        # 2D入力は正規化された(x, y)と信頼度(conf)を結合
        return np.concatenate([xy_norm, last_dim], axis=-1)

def create_dataset():
    # ファイル検索と分割ロジックは変更なし
    raw_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.pkl")))
    if not raw_files:
        print("No raw files found.")
        return

    random.seed(42)
    random.shuffle(raw_files)
    split_idx = int(len(raw_files) * TRAIN_RATIO)
    
    subsets = {
        "train": raw_files[:split_idx],
        "test": raw_files[split_idx:]
    }

    for sub in subsets:
        os.makedirs(os.path.join(OUTPUT_DATASET_DIR, sub), exist_ok=True)

    print(f"Start processing: Train={len(subsets['train'])}, Test={len(subsets['test'])}")

    for subset_name, files in subsets.items():
        count = 0
        for pkl_path in tqdm(files, desc=f"Processing {subset_name}"):
            with open(pkl_path, "rb") as f:
                raw = pickle.load(f)
            
            # データの取り出し: 3D座標は World座標のmm、2D座標は Pixel値
            label_3d_mm = raw["data_label"] # (T_all, 17, 3)
            input_2d_px = raw["data_input"] # (T_all, 17, 3) [x, y, conf]
            res_w = raw["res_w"]
            res_h = raw["res_h"]
            
            # --- 変換処理 ---
            
            # 1. 2D正規化 (input_norm)
            input_norm = normalize_2d_3d_h36m(input_2d_px, res_w, res_h, is_3d=False)
            
            # 2. 3D正規化 (label_norm): ルート相対化は行わない
            #   注意: Blenderで出力した3Dはmm単位なので、ここではZ軸も正規化される。
            label_norm = normalize_2d_3d_h36m(label_3d_mm, res_w, res_h, is_3d=True)
            
            # 3. クリップ分割
            total_frames = label_norm.shape[0]
            num_clips = (total_frames - CLIP_LEN) // STRIDE + 1
            
            if num_clips < 1:
                continue

            base_name = os.path.splitext(os.path.basename(pkl_path))[0]

            for i in range(num_clips):
                start = i * STRIDE
                end = start + CLIP_LEN
                
                # 切り出し
                clip_label = label_norm[start:end] # (243, 17, 3) Fully Normalized
                clip_input = input_norm[start:end] # (243, 17, 3) Normalized + Conf
                
                # 保存データ作成
                save_dict = {
                    "data_label": clip_label.astype(np.float32), 
                    "data_input": clip_input.astype(np.float32)
                }
                
                # ファイル保存
                save_name = f"{base_name}_clip{i:03d}.pkl"
                save_path = os.path.join(OUTPUT_DATASET_DIR, subset_name, save_name)
                
                with open(save_path, "wb") as f:
                    pickle.dump(save_dict, f)
                count += 1
                
        print(f"[{subset_name}] Created {count} clips.")

    print(f"\nDone! Dataset ready at: {OUTPUT_DATASET_DIR}")

if __name__ == "__main__":
    create_dataset()