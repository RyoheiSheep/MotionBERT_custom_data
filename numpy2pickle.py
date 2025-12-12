import numpy as np
import pickle

# パラメータ
T = 100   # フレーム数
J = 7     # キーポイント数（本来は17だが今回は7）

# (T, J, 3) のランダム配列を作成
data = np.random.rand(T, J, 3)

# pickle で保存
with open("keypoints.pkl", "wb") as f:
    pickle.dump(data, f)

print("Saved keypoints.pkl")
