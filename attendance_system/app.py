import streamlit as st
import cv2
import numpy as np
import face_rec_utils as utils
import pandas as pd
import time
import os
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Attendance System", layout="wide")

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ["Home", "Mark Attendance", "Register User"])


# Define WebRTC Processors at module level to prevent re-definition on script re-run
class RegistrationProcessor(VideoProcessorBase):
    def __init__(self):
        self.count = 0
        self.user_id = None
        self.save_path = None
        self.capturing = False
        # Load cascade once
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def update_config(self, user_id, save_path, capturing):
        self.user_id = user_id
        self.save_path = save_path
        self.capturing = capturing

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Ensure we have a valid image
        if img is None:
            return av.VideoFrame.from_ndarray(np.zeros((1, 1, 3), dtype=np.uint8), format="bgr24")

        # Capture logic
        if self.capturing and self.user_id is not None and self.count < 30:
            saved = utils.save_training_image(img, self.user_id, self.count, self.save_path)
            if saved:
                self.count += 1
        
        # Draw status
        cv2.putText(img, f"Captured: {self.count}/30", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

class AttendanceProcessor(VideoProcessorBase):
    def __init__(self):
        self.recognizer = utils.load_recognizer()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.user_map = utils.get_user_map()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if img is None:
             return av.VideoFrame.from_ndarray(np.zeros((1, 1, 3), dtype=np.uint8), format="bgr24")
             
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            # Predict
            if self.recognizer:
                try:
                    id, confidence = self.recognizer.predict(gray[y:y+h, x:x+w])
                    
                    if confidence < 100:
                        name = self.user_map.get(id, f"User {id}")
                        utils.mark_attendance(name)
                        status_text = f"{name}"
                        status_color = (0, 255, 0)
                    else:
                        status_text = "Unknown"
                        status_color = (0, 0, 255)
                except Exception as e:
                    status_text = "Error"
                    status_color = (0, 0, 255)
                    print(f"Prediction error: {e}")
            else:
                 status_text = "Model not loaded"
                 status_color = (0, 0, 255)

            cv2.rectangle(img, (x, y), (x+w, y+h), status_color, 2)
            cv2.putText(img, status_text, (x+6, y+h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

if choice == "Home":
    st.title("Face Recognition Attendance System (OpenCV)")
    st.image("https://media.istockphoto.com/id/1199046636/vector/facial-recognition-system-identification-of-a-person.jpg?s=612x612&w=0&k=20&c=L_vGZ4yJ8M3n0U7m0rJ2b5q_8u3v_9_Z8z_9_Z8z_9.jpg", width=600)
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
    st.write("To register, we need to capture face samples.")
    
    # We use a session state to track if we are capturing
    if "capturing" not in st.session_state:
        st.session_state.capturing = False
    
    # Start the streamer
    st.info("Click 'Start' to open camera. After camera opens, click 'Start Capture' to begin saving frames.")
    ctx = webrtc_streamer(
        key="registration", 
        video_processor_factory=RegistrationProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    )
    
    if ctx.video_processor:
        # Check if we should start capturing
        if st.button("Start Capture"):
            if not name_input:
                st.error("Please enter a name first.")
            else:
                user_id, save_path = utils.create_dataset_dir(name_input)
                ctx.video_processor.update_config(user_id, save_path, True)
                st.write(f"Capturing for {name_input}...")
        
        # Display progress if possible (note: ctx.video_processor.count is in another thread, might lag)
        
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
        st.write("Camera is running... Face recognition active.")
        
        webrtc_streamer(
            key="attendance",
            video_processor_factory=AttendanceProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        )


