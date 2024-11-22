import cv2
import numpy as np
import os
import shutil

source_dir = '/Users/ccy/Documents/CMU/Fall2024/Visual Learning and Recognition/Group_project/VisualLearning-Project/data/test'  


for root, dirs, files in os.walk(source_dir):
    for file in files:
        
        if file.endswith('.jpg') and file not in ['target.jpg', 'mask.jpg']:
            
            folder_name = os.path.splitext(file)[0]
            
            
            folder_path = os.path.join(root, folder_name)
            if os.path.isdir(folder_path): 
                
                src_path = os.path.join(root, file)
                dst_path = os.path.join(folder_path, file)
                shutil.move(src_path, dst_path)
                print(f"Moved {src_path} to {dst_path}")