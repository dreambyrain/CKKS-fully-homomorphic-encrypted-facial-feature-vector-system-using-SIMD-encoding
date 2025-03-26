import tkinter as tk
from tkinter import filedialog, messagebox
import os
import face_recognition
import pickle
import tenseal as ts
import numpy as np
import time
import logging

# 全局配置
DATASET_PATH = "/mnt/hgfs/casia_webface_100"
FEATURES_FILE = "features.pkl"
ENCRYPTED_FILE = "encrypted_features_bfv.pkl"
SCALE_FACTOR = 100  # 调整缩放因子为100，确保计算不溢出
POLY_MODULUS_DEGREE = 8192  # 增大多项式模数以支持批处理
PLAIN_MODULUS = 1032193  # 选择一个满足条件的素数

# 配置日志
logging.basicConfig(filename='performancecomparison.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def measure_time(func):
    """时间测量装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"{func.__name__} 耗时: {elapsed_time:.4f} 秒")
        messagebox.showinfo("耗时", f"{func.__name__} 耗时: {elapsed_time:.4f} 秒")
        return result
    return wrapper

def normalize_features(feature_vectors):
    """将特征向量归一化到 [-1, 1] 范围"""
    feature_vectors = np.array(feature_vectors)
    max_val = np.max(np.abs(feature_vectors))
    return feature_vectors / max_val

@measure_time
def extract_features():
    """提取CASIA-WebFace数据集的人脸特征（明文）"""
    if not os.path.exists(DATASET_PATH):
        messagebox.showerror("错误", f"数据集文件夹不存在: {DATASET_PATH}")
        return

    feature_vectors = []
    image_paths = []

    for filename in os.listdir(DATASET_PATH):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(DATASET_PATH, filename)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                feature_vectors.append(encodings[0])
                image_paths.append(img_path)

    if feature_vectors:
        # 归一化特征向量
        feature_vectors = normalize_features(feature_vectors)
        data = {"paths": image_paths, "features": feature_vectors}
        with open(FEATURES_FILE, "wb") as f:
            pickle.dump(data, f)
        messagebox.showinfo("成功", f"已提取 {len(feature_vectors)} 张人脸特征")
    else:
        messagebox.showerror("错误", "未提取到特征")

@measure_time
def encrypt_features():
    """使用BFV加密特征向量"""
    if not os.path.exists(FEATURES_FILE):
        messagebox.showerror("错误", "请先提取特征向量！")
        return

    # 加载明文特征向量
    with open(FEATURES_FILE, "rb") as f:
        data = pickle.load(f)
    features = data["features"]

    # 将浮点特征向量转换为整数（缩放并取整）
    scaled_features = [np.array(vec * SCALE_FACTOR, dtype=np.int64) for vec in features]

    # 配置BFV参数（支持批处理）
    context = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        plain_modulus=PLAIN_MODULUS
    )
    context.generate_galois_keys()  # 生成 Galois 密钥以支持旋转操作

    # 检查是否支持批处理
    if not context.is_private():
        messagebox.showerror("错误", "加密上下文不支持批处理")
        return

    # 加密特征向量
    encrypted_vectors = [ts.bfv_vector(context, vec).serialize() for vec in scaled_features]

    # 保存加密数据
    with open(ENCRYPTED_FILE, "wb") as f:
        pickle.dump({
            "context": context.serialize(save_secret_key=True, save_galois_keys=True),
            "encrypted_vectors": encrypted_vectors,
            "scale_factor": SCALE_FACTOR
        }, f)
    messagebox.showinfo("成功", "特征向量已加密（BFV）")

@measure_time
def verify_user():
    """用户验证（BFV加密域计算）"""
    if not os.path.exists(ENCRYPTED_FILE):
        messagebox.showerror("错误", "请先加密特征向量！")
        return

    # 加载加密数据
    with open(ENCRYPTED_FILE, "rb") as f:
        data = pickle.load(f)
    context = ts.context_from(data["context"])  # 加载上下文（包含 Galois 密钥）
    encrypted_vectors = [ts.bfv_vector_from(context, vec) for vec in data["encrypted_vectors"]]
    scale_factor = data["scale_factor"]

    # 用户上传图片
    file_path = filedialog.askopenfilename(
        title="选择上传图片",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        messagebox.showwarning("文件选择", "未选择图片文件。")
        return

    # 提取用户特征并缩放为整数
    user_image = face_recognition.load_image_file(file_path)
    user_encodings = face_recognition.face_encodings(user_image)
    if not user_encodings:
        messagebox.showerror("错误", "未检测到人脸")
        return
    user_feature = np.array(user_encodings[0] * SCALE_FACTOR, dtype=np.int64)

    # 加密用户特征
    user_encrypted = ts.bfv_vector(context, user_feature)

    # 计算平方欧式距离（整数域）
    distances = []
    for enc_vec in encrypted_vectors:
        diff = user_encrypted - enc_vec
        squared_diff = diff * diff
        sum_result = squared_diff.sum()  # 使用旋转操作实现跨位置求和
        distance = max(0, sum_result.decrypt()[0])  # 确保距离非负
        distances.append(distance)

    # 还原实际距离（除以缩放因子的平方）
    actual_distances = [d / (scale_factor ** 2) for d in distances]
    min_distance = min(actual_distances)

    # 判断结果（基于经验阈值）
    threshold_offset = 2.4  # 经验值
    threshold = 1.0  # 经验值
    if (min_distance - threshold_offset) < threshold:
        messagebox.showinfo("结果", f"验证通过！距离：{min_distance:.4f}")
    else:
        messagebox.showinfo("结果", f"验证失败！距离：{min_distance:.4f}")

# GUI界面（保持不变）
root = tk.Tk()
root.title("人脸识别与BFV加密系统")
root.geometry("400x300")

btn_extract = tk.Button(root, text="提取图库特征", command=extract_features, width=25)
btn_extract.pack(pady=10)

btn_encrypt = tk.Button(root, text="加密特征向量", command=encrypt_features, width=25)
btn_encrypt.pack(pady=10)

btn_verify = tk.Button(root, text="上传图片验证", command=verify_user, width=25)
btn_verify.pack(pady=10)

btn_exit = tk.Button(root, text="退出", command=root.quit, width=25)
btn_exit.pack(pady=10)

root.mainloop()