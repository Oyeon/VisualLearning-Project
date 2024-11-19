import cv2
import numpy as np
import os
import shutil
# 定義來源資料夾
source_dir = '/Users/ccy/Documents/CMU/Fall2024/Visual Learning and Recognition/Group_project/VisualLearning-Project/data/test'  # 替換為你的來源資料夾路徑

# 遍歷所有子資料夾和文件
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # 檢查是否是我們需要移動的文件（排除 target.jpg 和 mask.jpg）
        if file.endswith('.jpg') and file not in ['target.jpg', 'mask.jpg']:
            # 提取文件名（不含副檔名）
            folder_name = os.path.splitext(file)[0]
            
            # 找到對應的資料夾
            folder_path = os.path.join(root, folder_name)
            if os.path.isdir(folder_path):  # 確保資料夾存在
                # 移動文件到對應的資料夾
                src_path = os.path.join(root, file)
                dst_path = os.path.join(folder_path, file)
                shutil.move(src_path, dst_path)
                print(f"Moved {src_path} to {dst_path}")