import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


canvas = np.zeros((480, 640, 3), dtype=np.uint8)


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  
current_color_index = 0
points = []

def count_fingers(landmarks):
    """Count raised fingers based on landmark positions"""
    finger_tips = [4, 8, 12, 16, 20] 
    count = 0
    
    
    if landmarks[4].x < landmarks[3].x: 
        count += 1
    
    
    for tip in finger_tips[1:]:
        if landmarks[tip].y < landmarks[tip-2].y: 
            count += 1
            
    return count

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
           
            finger_count = count_fingers(landmarks.landmark)
            
           
            if finger_count == 3:
                current_color_index = (current_color_index + 1) % len(colors)
                points = []  
                cv2.waitKey(500) 
            
           
            elif finger_count == 5:
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                points = []
                cv2.waitKey(500)  
            
           
            else:
                x = int(landmarks.landmark[8].x * frame.shape[1]) 
                y = int(landmarks.landmark[8].y * frame.shape[0])
                points.append((x, y))
    
    
    if len(points) > 1:
        for i in range(1, len(points)):
            cv2.line(canvas, points[i-1], points[i], colors[current_color_index], 5)
    
    
    cv2.rectangle(frame, (10, 10, 50, 50), colors[current_color_index], -1)
    cv2.putText(frame, f"Color {current_color_index + 1}", (70, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Canvas", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()