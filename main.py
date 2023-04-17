import cv2

# Load video file
cap = cv2.VideoCapture('video.mp4')

# Load vehicle detection model
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# Initialize vehicle count
count = 0

# Loop through each frame in the video
while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect vehicles in the frame
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)

    # Draw rectangles around detected vehicles and update count
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        count += 1

    # Display vehicle count on screen
    cv2.putText(frame, f'Count: {count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with vehicle detections and count
    cv2.imshow('Vehicle Detection', frame)

    # Exit if user presses 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
