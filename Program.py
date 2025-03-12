import tkinter as tk
from tkinter import filedialog, messagebox
import os
import face_recognition
import pickle
import tenseal as ts  # 用于 CKKS 加密

# CASIA-WebFace 数据集路径
DATASET_PATH = "/mnt/hgfs/casia_webface_100"
FEATURES_FILE = "features.pkl"
ENCRYPTED_FILE = "encrypted_features.pkl"

def extract_features():
    """ 提取 CASIA-WebFace 数据集的人脸特征 """
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
        data = {"paths": image_paths, "features": feature_vectors}
        with open(FEATURES_FILE, "wb") as f:
            pickle.dump(data, f)
        messagebox.showinfo("成功", f"已提取 {len(feature_vectors)} 张人脸特征，并保存到 {FEATURES_FILE}")
    else:
        messagebox.showerror("错误", "没有提取到任何人脸特征，请检查数据集！")

def encrypt_features():
    """ 对特征向量进行 CKKS 加密 """
    if not os.path.exists(FEATURES_FILE):
        messagebox.showerror("错误", "请先提取特征向量！")
        return

    # 加载特征向量
    with open(FEATURES_FILE, "rb") as f:
        data = pickle.load(f)
    feature_vectors = data["features"]

    # 归一化特征向量
    import numpy as np
    feature_vectors = np.array(feature_vectors)
    max_val = np.max(np.abs(feature_vectors))
    feature_vectors = feature_vectors / max_val

    # 创建 CKKS 上下文
    poly_modulus_degree = 16384
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=[60, 50, 50, 60]
    )
    context.generate_galois_keys()  # 生成 Galois 密钥
    context.global_scale = 2**50  # 增大全局缩放因子

    # 加密特征向量，并转换为字节流
    encrypted_vectors = [ts.ckks_vector(context, vec).serialize() for vec in feature_vectors]

    # 保存加密后的特征向量和上下文（包含私钥和 Galois 密钥）
    with open(ENCRYPTED_FILE, "wb") as f:
        pickle.dump({
            "context": context.serialize(save_secret_key=True, save_galois_keys=True),
            "encrypted_vectors": encrypted_vectors,
            "poly_modulus_degree": poly_modulus_degree
        }, f)

    messagebox.showinfo("成功", "特征向量已加密并保存到 encrypted_features.pkl")

def verify_user():
    """ 用户上传图片并进行验证 """
    if not os.path.exists(ENCRYPTED_FILE):
        messagebox.showerror("错误", "请先加密特征向量！")
        return

    # 加载加密后的特征向量和上下文
    with open(ENCRYPTED_FILE, "rb") as f:
        data = pickle.load(f)
    context = ts.context_from(data["context"])
    encrypted_vectors = [ts.ckks_vector_from(context, vec) for vec in data["encrypted_vectors"]]
    poly_modulus_degree = data["poly_modulus_degree"]

    # 用户上传图片
    file_path = filedialog.askopenfilename(
        title="选择上传图片",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        messagebox.showwarning("文件选择", "未选择图片文件。")
        return

    # 提取用户图片的特征向量
    user_image = face_recognition.load_image_file(file_path)
    user_encodings = face_recognition.face_encodings(user_image)
    if len(user_encodings) == 0:
        messagebox.showerror("错误", "未检测到人脸，请上传有效图片！")
        return
    user_feature = user_encodings[0]

    # 加密用户特征向量
    user_encrypted = ts.ckks_vector(context, user_feature)

    # 计算与加密特征向量的平方欧式距离
    distances = []
    for enc_vec in encrypted_vectors:
        diff = user_encrypted - enc_vec
        squared_diff = diff * diff

        # 使用 sum() 方法直接求和
        sum_result = squared_diff.sum()
        distance = max(0, sum_result.decrypt()[0])  # 确保距离非负
        distances.append(distance)

    # 找到最小距离
    min_distance = min(distances)

    # 根据阈值规则判断是否为同一用户
    threshold = 2.4
    if min_distance - threshold < 1:
        messagebox.showinfo("验证结果", f"验证通过！最小距离: {min_distance}")
    else:
        messagebox.showinfo("验证结果", f"验证失败！最小距离: {min_distance}")

# 创建主窗口
root = tk.Tk()
root.title("人脸识别与同态加密系统")
root.geometry("400x300")

# 添加按钮
btn_extract = tk.Button(root, text="提取图库特征", command=extract_features, width=25)
btn_extract.pack(pady=10)

btn_encrypt = tk.Button(root, text="加密特征向量", command=encrypt_features, width=25)
btn_encrypt.pack(pady=10)

btn_verify = tk.Button(root, text="上传图片验证", command=verify_user, width=25)
btn_verify.pack(pady=10)

btn_exit = tk.Button(root, text="退出", command=root.quit, width=25)
btn_exit.pack(pady=10)

# 运行主循环
root.mainloop()
