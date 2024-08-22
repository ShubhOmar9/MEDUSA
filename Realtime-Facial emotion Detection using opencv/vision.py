import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load the face detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Create the model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
output = Dense(len(emotions), activation='softmax')(x)
emotion_model = Model(inputs=base_model.input, outputs=output)

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    _, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = frame[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (224, 224))
        face_roi = preprocess_input(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)

        # Predict the emotion
        prediction = emotion_model.predict(face_roi)[0]
        emotion_index = prediction.argmax()
        emotion_label = emotions[emotion_index]
        confidence = prediction[emotion_index]

        # Format the label with emotion and confidence
        label = f"{emotion_label}: {confidence:.2f}"

        # Draw the face detection box and emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()