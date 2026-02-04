import streamlit as st
import cv2
import numpy as np
import utils
import pandas as pd
import time
import os

st.set_page_config(page_title="Attendance System", layout="wide")

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ["Home", "Mark Attendance", "Register User"])

if choice == "Home":
    st.title("Face Recognition Attendance System (OpenCV)")
    st.image("https://media.istockphoto.com/id/1199046636/vector/facial-recognition-system-identification-of-a-person.jpg?s=612x612&w=0&k=20&c=L_vGZ4yJ8M3n0U7m0rJ2b5q_8u3v_9_Z8z_9_Z8z_9.jpg", use_column_width=True)
    st.write("Welcome! This system uses OpenCV LBPH Face Recognizer.")
    
    st.subheader("System Status")
    if os.path.exists(os.path.join(utils.TRAINER_DIR, utils.TRAINER_FILE)):
        st.success("Model is Trained and Ready.")
    else:
        st.warning("Model is NOT Trained. Please register users and then click Train.")

    # Show today's attendance
    st.subheader("Today's Attendance")
    if os.path.exists(utils.ATTENDANCE_FILE):
        df = pd.read_csv(utils.ATTENDANCE_FILE)
        st.dataframe(df)

elif choice == "Register User":
    st.title("Register New User")
    
    # Display Registered Users
    st.subheader("Registered Users")
    users = utils.get_registered_users()
    if users:
        st.write(f"Total: {len(users)}")
        st.write(", ".join(users))
    else:
        st.info("No users registered yet.")
    
    st.markdown("---")
    
    name_input = st.text_input("Enter New User Name")
    st.write("To register, we need to capture multiple face samples.")
    
    if st.button("Start Capture"):
        if not name_input:
            st.error("Please enter a name first.")
        else:
            user_id, save_path = utils.create_dataset_dir(name_input)
            st.info(f"Capturing faces for {name_input} (ID: {user_id}). Look at the camera.")
            
            camera = cv2.VideoCapture(0)
            count = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frame_placeholder = st.empty()
            
            while count < 30: # Capture 30 images
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to read camera.")
                    break
                    
                saved = utils.save_training_image(frame, user_id, count, save_path)
                
                # Draw prompt on frame
                cv2.putText(frame, f"Captured: {count}/30", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if saved:
                    count += 1
                    progress_bar.progress(count / 30)
                    # Small delay to let user move head
                    time.sleep(0.1)
            
            camera.release()
            st.success(f"Captured 30 images for {name_input}.")
            st.balloons()
            
    st.markdown("---")
    st.subheader("Train Model")
    if st.button("Train System"):
        with st.spinner("Training model... this might take a minute."):
            n_faces = utils.train_recognizer()
        if n_faces > 0:
            st.success(f"Model trained with {n_faces} unique users.")
        else:
            st.error("No data found to train.")

elif choice == "Mark Attendance":
    st.title("Mark Attendance")
    
    recognizer = utils.load_recognizer()
    if recognizer is None:
        st.error("Model not found. Please register users and TRAIN the model first.")
    else:
        user_map = utils.get_user_map()
        face_cascade = utils.face_cascade
        
        run = st.checkbox('Run Camera')
        FRAME_WINDOW = st.image([])
        
        camera = cv2.VideoCapture(0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        while run:
            ret, frame = camera.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            
            for (x, y, w, h) in faces:
                # Predict
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                
                # Check confidence (LBPH: Lower is better. < 100 is relaxed match)
                if confidence < 100:
                    name = user_map.get(id, f"User {id}")
                    
                    # Mark attendance
                    is_new_entry = utils.mark_attendance(name)
                    
                    if is_new_entry:
                        status_color = (0, 255, 0) # Green
                        status_text = f"Marked: {name}"
                    else:
                        status_color = (0, 255, 255) # Yellow
                        status_text = f"{name} (Already Present)"
                    
                else:
                    name = "Unknown"
                    status_text = f"Unknown ({round(confidence)})"
                    status_color = (0, 0, 255) # Red
                
                # Checkbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), status_color, 2)
                
                # Background strip for text
                cv2.rectangle(frame, (x, y+h-35), (x+w, y+h), status_color, cv2.FILLED)
                cv2.putText(frame, status_text, (x+6, y+h-6), font, 0.6, (0, 0, 0), 1)
            
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        camera.release()
