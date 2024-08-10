"""
using mediapipe holistiic
No start button
Stop button stops the feed gives the transcription text

audio module included 
"""


import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import tensorflow as tf

# Initialize MediaPipe Holistic and Drawing
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
model = tf.keras.models.load_model('asl_model_v2_1.h5')
gesture_to_text_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: 'space', 27: 'del', 28: 'nothing'
}

# Streamlit application
def main():
    st.title("ASL Gesture Recognition")
    st.subheader("Automated Sign Language Detection")

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")

    transcribed_text = st.empty()
    if 'text' not in st.session_state:
        st.session_state['text'] = ""

    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # Process hand landmarks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
            )
            left_hand_landmarks = [(int(landmark.x * w), int(landmark.y * h)) for landmark in results.left_hand_landmarks.landmark]
            x_coords, y_coords = zip(*left_hand_landmarks)
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_min = max(x_min - 10, 0)
            y_min = max(y_min - 10, 0)
            x_max = min(x_max + 10, frame.shape[1])
            y_max = min(y_max + 10, frame.shape[0])

            hand_img = frame[y_min:y_max, x_min:x_max]
            hand_img_resized = cv2.resize(hand_img, (224, 224))
            hand_img_normalized = hand_img_resized / 255.0
            hand_img_batch = np.expand_dims(hand_img_normalized, axis=0)

            predictions = model.predict(hand_img_batch)
            predicted_class = np.argmax(predictions[0])
            predicted_text = gesture_to_text_mapping[predicted_class]

            if predicted_text == 'space':
                st.session_state['text'] += " "
            elif predicted_text == 'del':
                st.session_state['text'] = st.session_state['text'][:-1]
            elif predicted_text != 'nothing':
                st.session_state['text'] += predicted_text

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"Predicted: {predicted_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
            )
            right_hand_landmarks = [(int(landmark.x * w), int(landmark.y * h)) for landmark in results.right_hand_landmarks.landmark]
            x_coords, y_coords = zip(*right_hand_landmarks)
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_min = max(x_min - 10, 0)
            y_min = max(y_min - 10, 0)
            x_max = min(x_max + 10, frame.shape[1])
            y_max = min(y_max + 10, frame.shape[0])

            hand_img = frame[y_min:y_max, x_min:x_max]
            hand_img_resized = cv2.resize(hand_img, (224, 224))
            hand_img_normalized = hand_img_resized / 255.0
            hand_img_batch = np.expand_dims(hand_img_normalized, axis=0)

            predictions = model.predict(hand_img_batch)
            predicted_class = np.argmax(predictions[0])
            predicted_text = gesture_to_text_mapping[predicted_class]

            if predicted_text == 'space':
                st.session_state['text'] += " "
            elif predicted_text == 'del':
                st.session_state['text'] = st.session_state['text'][:-1]
            elif predicted_text != 'nothing':
                st.session_state['text'] += predicted_text

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"Predicted: {predicted_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)



        # Display the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame)

        # Check if stop button was pressed
        if stop_button_pressed:
            running = False

    cap.release()
    cv2.destroyAllWindows()

    # Display the transcribed text
    transcribed_text.write(f"**Transcribed Text:** {st.session_state['text']}")
     
    # Optionally play an audio file
    st.audio('output1.mp3', format="audio/wav", autoplay=True)

if __name__ == "__main__":
    main()
