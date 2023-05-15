import cv2
import numpy as np
import onnxruntime as ort

# Load the face detection model
face_detector = cv2.CascadeClassifier('C:/Users/tjdql/Desktop/haarcascade_frontalface_alt.xml')

# Load the face recognition model
model = ort.InferenceSession('C:/Users/tjdql/Desktop/holy.onnx')

# Define label dictionary
label_dict = {
    0: 'Sungbin',
    1: 'Yong',
    2: 'Unknown'
}

# Set the similarity threshold
similarity_threshold = 0.50

# Set up the video capture
cap = cv2.VideoCapture(0)

# Loop through each frame of the video
while True:
    # Read the frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = frame[y:y+h, x:x+w]

        # Preprocess the face ROI
        face_roi = cv2.resize(face_roi, (64, 64))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_roi = face_roi.astype('float32') / 255.0

        # Run the face recognition model on the face ROI
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        output = model.run([output_name], {input_name: face_roi[np.newaxis, ...]})[0][0]

        # Get the predicted label
        label = np.argmax(output)
        confidence = output[label]

        # If the confidence score is less than the threshold, set the label as 'Unknown'
        if confidence < similarity_threshold:
            label = 2

        # Draw a rectangle around the face
        color = (255, 0, 0) if label == 0 else (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Draw the predicted label on the rectangle
        text = label_dict[label]
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the result
    cv2.imshow("Face recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#
cap.release()
cv2.destroyAllWindows()
