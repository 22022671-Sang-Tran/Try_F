import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Initially set finger count to 0 for each cap
        finger_count = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand index to check label (left or right)
                handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                handLabel = results.multi_handedness[handIndex].classification[0].label

                # Set variable to keep landmarks positions (x and y)
                hand_landmarks_list = []

                # Fill list with x and y positions of each landmark
                for landmark in hand_landmarks.landmark:
                    hand_landmarks_list.append([landmark.x, landmark.y])
                    
                    
                # Calculate the angle between lines a and b
                a = hand_landmarks_list[4]  # Thumb Tip
                b = hand_landmarks_list[3]  # Thumb IP
                c = hand_landmarks_list[2]  # Thumb MCP

                angle = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))

                # Display the angle
                #cv2.putText(image, f"Angle: {angle:.2f} degrees", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


                # Test conditions for each finger: Count is increased if finger is
                # considered raised.
                # Thumb: TIP x position must be greater or lower than IP x position,
                # depending on hand label.
                if handLabel == "Left":
                    if hand_landmarks_list[4][0] > hand_landmarks_list[13][0] and angle<=180.0:
                        finger_count+=1
                    elif hand_landmarks_list[4][0] < hand_landmarks_list[13][0] and angle > 180.0:
                        finger_count+=1
                elif handLabel == "Right":
                    if hand_landmarks_list[4][0] < hand_landmarks_list[13][0] and angle>180.0:
                        finger_count += 1
                    elif hand_landmarks_list[4][0] > hand_landmarks_list[13][0] and angle<=180.0:
                        finger_count +=1

                # Other fingers: TIP y position must be lower than PIP y position,
                # as the image origin is in the upper left corner.
                if hand_landmarks_list[8][1] < hand_landmarks_list[7][1]:  # Index finger
                    finger_count += 1
                if hand_landmarks_list[12][1] < hand_landmarks_list[11][1]:  # Middle finger
                    finger_count += 1
                if hand_landmarks_list[16][1] < hand_landmarks_list[15][1]:  # Ring finger
                    finger_count += 1
                if hand_landmarks_list[20][1] < hand_landmarks_list[19][1]:  # Pinky
                    finger_count += 1

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            # Display finger count
            cv2.putText(image, f"Fingers: {finger_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display image
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
