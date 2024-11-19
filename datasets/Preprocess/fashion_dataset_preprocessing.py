import cv2
import numpy as np
import os

from scipy.io import loadmat
import pickle

def check_image_shapes(data_path):
    image_shapes = {}

    for dirpath, _, filenames in os.walk(data_path):
        for filename in filenames:
            if filename == 'mask.jpg':
                file_path = os.path.join(dirpath, filename)
                image = cv2.imread(file_path)
                if image is not None:
                    image_shapes[file_path] = image.shape
                else:
                    print(f"Warning: Unable to read {file_path}")

    # Display results
    unique_shapes = set(image_shapes.values())
    print(f"Unique shapes found: {unique_shapes}")

def original_mask_value(data_pth):
    segmentation_mask = cv2.imread(data_pth)
    points_to_check = [(50, 50), (100, 100), (150, 150)]  # Replace with actual shallow blue region coordinates
    # Print the BGR values at these coordinates
    for point in points_to_check:
        b, g, r = segmentation_mask[point[1], point[0]]  # Note: OpenCV uses (y, x) format
        print(f"BGR value at {point}: ({b}, {g}, {r})")

def record_the_contour_and_the_coordinates(mask, save_coordinate_path):
    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours: 
        largest_contours = max(contours, key= cv2.contourArea)

        x, y, w ,h = cv2.boundingRect(largest_contours)

        #cv2.rectangle(mask,(x,y), (x+w,y+h), (0,0,0))

        with open(save_coordinate_path, 'w', newline='') as f:
            f.write(f"{x}, {y}, {w}, {h}")

def turn_mask_into_binary_mask(data_pth, save_path):
    segmentation_mask = cv2.imread(data_pth)

    '''Check the image'''
    #print(segmentation_mask.shape)
    #print(type(segmentation_mask))

    '''Threshold for the region of clothes'''
    lower_blue = np.array([200, 100, 0])  # Updated lower bound for blue in BGR
    upper_blue = np.array([255, 150, 50]) # Updated upper bound for blue in BGR

    lower_orange = np.array([0, 50, 150])   # Updated lower bound for orange in BGR
    upper_orange = np.array([80, 150, 255]) # Updated upper bound for orange in BGR

    '''Image preprocessing'''
    # Create binary masks for each color
    blue_mask = cv2.inRange(segmentation_mask, lower_blue, upper_blue)
    orange_mask = cv2.inRange(segmentation_mask, lower_orange, upper_orange)

    # Combine the masks to get a single mask for the clothing area
    clothing_mask = cv2.bitwise_or(blue_mask, orange_mask)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_OPEN, kernel)

    '''Save the Image'''
    cv2.imwrite(save_path, clothing_mask)

    return clothing_mask

def read_coordinates_file(coordinate_path, image_path ,reference_image_save_path):
    with open(coordinate_path,'r') as f:
        line = f.readline().strip()  # 讀取並去除換行符號
        x, y, w, h = map(int, line.split(','))

    # Load the original image
    image = cv2.imread(image_path)

    #print((int(x), int(y)), (int(x+w), int(y+h)))

    # Draw the rectangle on the image to segment the upper body
    upper_body_segmented = image.copy()
    cv2.rectangle(upper_body_segmented, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 0), thickness=-1)

    # Save or display the result
    cv2.imwrite(reference_image_save_path, upper_body_segmented)


def read_pose_file():
    # Load .pkl file
    with open('/Users/ccy/Documents/CMU/Fall2024/Visual Learning and Recognition/Group_project/VisualLearning-Project/data/train/0/1_target/pose.pkl', 'rb') as f:
        pose_data = pickle.load(f)

    candidate = pose_data['candidate']

    # Define keypoint indices for body parts (these may vary based on dataset convention)
    neck_index = 1
    right_shoulder_index = 2
    left_shoulder_index = 5
    wrist_index = 8

    # Get coordinates for neck and shoulders
    neck = candidate[neck_index][:2]  # [x, y]
    right_shoulder = candidate[right_shoulder_index][:2]  # [x, y]
    left_shoulder = candidate[left_shoulder_index][:2]  # [x, y]
    wrist = candidate[wrist_index][:2] # [x, y]
    
    # Calculate bounding box for upper body
    top = int(neck[1])
    bottom = int(wrist[1])
    left = int(left_shoulder[0])
    right = int(right_shoulder[0])
    width = abs(left-right)  # Approximate height based on shoulder width
    height = abs(top-bottom)

    max_x = np.max(candidate[:, 0])
    max_y = np.max(candidate[:, 1])
    min_x = np.min(candidate[:, 0])
    min_y = np.min(candidate[:, 1])

    scale_x = width / (max_x-min_x)
    scale_y = height / (max_y-min_y)

    # Load the original image
    image_path = '/Users/ccy/Documents/CMU/Fall2024/Visual Learning and Recognition/Group_project/VisualLearning-Project/data/train/0/2.jpg'
    image = cv2.imread(image_path)

    print((int(left*scale_x), int(top*scale_y)), (int(right*scale_x), int(bottom*scale_y)))

    # Draw the rectangle on the image to segment the upper body
    upper_body_segmented = image.copy()
    cv2.rectangle(upper_body_segmented, (int(left*scale_x), int(top*scale_y)), (int(right*scale_x), int(bottom*scale_y)), (0, 0, 0), thickness=-1)

    # Save or display the result
    cv2.imwrite('upper_body_segmented.jpg', upper_body_segmented)
    print(pose_data)
    # Load the .mat file
    #data = loadmat('/Users/ccy/Documents/CMU/Fall2024/Visual Learning and Recognition/Group_project/VisualLearning-Project/data/train/0/2/pose.mat')

def main():
    parent_path = '/Users/ccy/Documents/CMU/Fall2024/Visual Learning and Recognition/Group_project/VisualLearning-Project/data/train'

    for root, dirs, files in os.walk(parent_path):
        for file in files:
            if file == 'coordinate.txt':
                coordinate_path = os.path.join(root, file)

                # 確定子目錄名稱，以便匹配對應的圖片文件名
                sub_dir_name = os.path.basename(root)  # e.g., "2" or "1_target"
                parent_dir = os.path.dirname(root)  # 取得上層目錄

                # 根據子目錄名稱來生成對應的圖片文件名
                image_file_name = f"{sub_dir_name}.jpg"
                image_path = os.path.join(parent_dir, image_file_name)

                # 確認圖片文件是否存在
                if os.path.exists(image_path):
                    reference_image_save_path = os.path.join(root, 'reference_mask.jpg')

                    # 輸出路徑並呼叫函數
                    #print("Image Path:", image_path)
                    #print("Coordinate Path:", coordinate_path)
                    #print("Reference Image Save Path:", reference_image_save_path)
                    read_coordinates_file(coordinate_path, image_path, reference_image_save_path)

    '''Try to load the pose.pkl file -- for reference mask'''
    #read_pose_file()

    '''Check the consistency of image shape''' # (256, 192, 3)
    check_image_shapes(parent_path)

    '''Turn mask into binary mask for Hint channel'''
    #for root, dirs, files in os.walk(parent_path):
    #    for file in files:
    #        if file == 'segment_vis.png':
    #            target_mask_path = os.path.join(root, file)
    #            target_save_path = os.path.join(root, 'segment_binary_mask.png')
    #            clothing_mask = turn_mask_into_binary_mask(target_mask_path, target_save_path)
#
    #            coordinate_save_path = os.path.join(root,'coordinate.txt')
    #            record_the_contour_and_the_coordinates(clothing_mask,coordinate_save_path)
    

if __name__ == '__main__':
    main()





