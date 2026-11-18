import os
import glob
import numpy as np

# 1. 路径设置（精准对应你截图中的路径）
input_dir = "./datasets/sleepedf-2013/npz/Fpz-Cz"
output_dir = "./datasets/sleepedf-2013/npz"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "sleep_edf_processed_Fpz.npz")

# 2. 读取所有的受试者 npz 文件并排序
file_list = sorted(glob.glob(os.path.join(input_dir, "*.npz")))
print(f"共找到 {len(file_list)} 个单受试者 npz 文件，准备打包...")

if len(file_list) == 0:
    raise FileNotFoundError(f"未在 {input_dir} 下找到任何 .npz 文件，请检查路径！")

# 3. 按受试者切分成 10 折（严格按受试者划分，防止数据泄露）
n_folds = 10
files_per_fold = np.array_split(file_list, n_folds)

fold_data = []
fold_label = []
fold_len = []

for k in range(n_folds):
    cur_x = []
    cur_y = []
    for f in files_per_fold[k]:
        d = np.load(f)
        x = d['x']  # 形状为 (N, 3000)
        y = d['y']  # 形状为 (N,)
        
        # 扩展维度为 (N, 3000, 1) 以符合卷积网络输入要求
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=-1)
            
        cur_x.append(x)
        cur_y.append(y)
    
    cur_x = np.concatenate(cur_x, axis=0)
    cur_y = np.concatenate(cur_y, axis=0)
    
    fold_data.append(cur_x)
    fold_label.append(cur_y)
    fold_len.append(len(cur_y))
    print(f"Fold {k}: 样本片段数 = {len(cur_y)}")

# 4. 打包保存（注意键名大小写严格匹配 MobileSleepNet.py 的读取要求）
np.savez(
    output_file,
    Fold_len=np.array(fold_len, dtype=object),
    Fold_data=np.array(fold_data, dtype=object),
    Fold_label=np.array(fold_label, dtype=object)
)

print(f"\n 打包完成！最终训练数据已成功保存至: {output_file}")
