import cv2
import os
import numpy as np
import csv
from datetime import datetime

DATASET_DIR = "dataset"
TRAINER_DIR = "trainer"
TRAINER_FILE = "trainer.yml"
ATTENDANCE_FILE = "attendance.csv"

# Initialize Face Detector (Haar Cascade)
# OpenCV usually comes with this xml, but we might need to find its path or download it.
# We will try to use the one included in cv2.data
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def create_dataset_dir(name):
    """
    Creates a directory for a user in the dataset folder.
    Returns the user ID (integer) and the directory path.
    """
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        
    # Simple ID generation: count subdirectories + 1
    # We need a stable mapping of ID -> Name. 
    # For simplicity, we'll use a mapping file or just folder structure "ID.Name"
    
    # Check existing folders to find next ID
    existing_ids = []
    for folder in os.listdir(DATASET_DIR):
        try:
            eid = int(folder.split('.')[0])
            existing_ids.append(eid)
        except:
            pass
            
    new_id = 1
    if existing_ids:
        new_id = max(existing_ids) + 1
        
    user_folder = f"{new_id}.{name}"
    user_path = os.path.join(DATASET_DIR, user_folder)
    
    if not os.path.exists(user_path):
        os.makedirs(user_path)
        
    return new_id, user_path

def save_training_image(image, user_id, count, save_path):
    """
    Detects face in the image and saves the cropped face to the save_path.
    Returns True if a face was saved, False otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Crop and save
        cv2.imwrite(f"{save_path}/User.{user_id}.{count}.jpg", gray[y:y+h, x:x+w])
        return True
    return False

def train_recognizer():
    """
    Trains the LBPH recognizer on all images in the dataset directory.
    Saves the model to trainer/trainer.yml.
    Returns the number of users trained.
    """
    if not os.path.exists(DATASET_DIR):
        return 0

    print("Training faces. It will take a few seconds. Wait ...")
    
    faces = []
    ids = []
    
    # Traverse through all user folders
    for user_folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, user_folder)
        if not os.path.isdir(folder_path):
            continue
            
        try:
            user_id = int(user_folder.split('.')[0])
        except:
            continue
            
        for image_name in os.listdir(folder_path):
            if image_name.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(folder_path, image_name)
                img_numpy = np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), 'uint8')
                
                faces.append(img_numpy)
                ids.append(user_id)
                
    if not faces:
        return 0

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    
    if not os.path.exists(TRAINER_DIR):
        os.makedirs(TRAINER_DIR)
        
    recognizer.write(os.path.join(TRAINER_DIR, TRAINER_FILE))
    print(f"{len(np.unique(ids))} faces trained. Exiting Program")
    return len(np.unique(ids))

def load_recognizer():
    fname = os.path.join(TRAINER_DIR, TRAINER_FILE)
    if not os.path.exists(fname):
        return None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(fname)
    return recognizer

def get_user_map():
    """
    Returns a dictionary mapping ID -> Name based on dataset folder names.
    """
    user_map = {}
    if os.path.exists(DATASET_DIR):
        for folder in os.listdir(DATASET_DIR):
            try:
                parts = folder.split('.')
                uid = int(parts[0])
                name = ".".join(parts[1:])
                user_map[uid] = name
            except:
                pass
    return user_map

def get_registered_users():
    """
    Returns a list of names of registered users.
    """
    names = []
    if os.path.exists(DATASET_DIR):
        for folder in os.listdir(DATASET_DIR):
            try:
                parts = folder.split('.')
                name = ".".join(parts[1:])
                names.append(name)
            except:
                pass
    return names

def mark_attendance(name):
    """
    Marks attendance for the given name if not already marked for today.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time", "Exit Time"])

    with open(ATTENDANCE_FILE, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)

    for line in lines:
        if line and len(line) >= 2:
            existing_name = line[0]
            existing_date = line[1]
            if existing_name == name and existing_date == date_str:
                return False

    with open(ATTENDANCE_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, date_str, time_str, ""])
    
    return True
