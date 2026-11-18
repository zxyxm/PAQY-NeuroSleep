import os
import numpy as np
import tensorflow as tf
from sklearn import metrics
from scipy.signal import savgol_filter

# 1. 映射分期标签
STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']

# 2. 检查并加载 TFLite 解释器
tflite_path = "model.tflite"
if not os.path.exists(tflite_path):
    raise FileNotFoundError(f"未找到端侧模型文件: {tflite_path}")

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("端侧模型期望输入维度:", input_details[0]['shape'])  # 期望为 (1, 3000, 1)

# 3. 读取你刚刚打包好的大文件
npz_pkg_path = "./datasets/sleepedf-2013/npz/sleep_edf_processed_Fpz.npz"
print(f"正在加载打包数据: {npz_pkg_path} ...")
pkg = np.load(npz_pkg_path, allow_pickle=True)

# 取第 0 折作为本次评估的测试集（模拟留出一折测试）
test_x = pkg['Fold_data'][0]
test_y = pkg['Fold_label'][0]

# 统一输入维度为 (N, 3000, 1)
if len(test_x.shape) == 2:
    test_x = np.expand_dims(test_x, axis=-1)

# 统一标签为整数类别索引 (0~4)
if len(test_y.shape) > 1:
    y_true = np.argmax(test_y, axis=1)
else:
    y_true = test_y.astype(int)

print(f"测试集准备完毕，待推断片段数: {len(test_x)} (每个片段30秒)")

# 4. 逐片段执行 TFLite 端侧模型前向推理
# 将数据调整为 (N, 10, 3000) 格式
print("正在通过 TFLite 解释器进行端侧推断...")
raw_probs = []

for i in range(len(test_x)):
    # 原始单通道切片形状为 (3000,) 或 (3000, 1)
    single_eeg = test_x[i].squeeze()  # (3000,)
    
    # 沿通道维度重复 10 次，形状变为 (1, 10, 3000)
    sample_10ch = np.tile(single_eeg, (10, 1))           # (10, 3000)
    sample_10ch = np.expand_dims(sample_10ch, axis=0)    # (1, 10, 3000)
    sample_10ch = sample_10ch.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], sample_10ch)
    interpreter.invoke()
    prob = interpreter.get_tensor(output_details[0]['index'])[0]
    raw_probs.append(prob)

raw_probs = np.array(raw_probs)
# 5. Savitzky-Golay (S-G) 概率时序平滑（论文核心免参数后处理）
def savgol_smooth(probs, window_length=5, polyorder=2):
    smoothed = np.zeros_like(probs)
    for c in range(probs.shape[1]):
        smoothed[:, c] = savgol_filter(probs[:, c], window_length=window_length, polyorder=polyorder)
    smoothed = np.clip(smoothed, 0, 1)
    sums = smoothed.sum(axis=1, keepdims=True)
    return smoothed / np.where(sums == 0, 1, sums)

smoothed_probs = savgol_smooth(raw_probs, window_length=5, polyorder=2)

# 6. 计算并输出各项分期指标
preds_raw = np.argmax(raw_probs, axis=1)
preds_smooth = np.argmax(smoothed_probs, axis=1)

acc_raw = metrics.accuracy_score(y_true, preds_raw)
acc_smooth = metrics.accuracy_score(y_true, preds_smooth)
mf1_smooth = metrics.f1_score(y_true, preds_smooth, average='macro')
kappa_smooth = metrics.cohen_kappa_score(y_true, preds_smooth)

print("\n" + "="*50)
print(f"【纯 TFLite 骨干输出】  Accuracy: {acc_raw * 100:.2f}%")
print(f"【TFLite + S-G 平滑】 Accuracy: {acc_smooth * 100:.2f}% | Macro-F1: {mf1_smooth * 100:.2f}% | Cohen Kappa: {kappa_smooth:.4f}")
print("="*50)

print("\n各睡眠分期分类详细报告 (Wake, N1, N2, N3, REM):")
print(metrics.classification_report(y_true, preds_smooth, target_names=STAGE_NAMES, digits=4))
