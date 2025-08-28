import cv2
import dlib
import numpy as np
import pygame
from scipy.spatial import distance as dist

class DrowsinessDetector:
    def __init__(self):
        # Initialize face and landmark detector
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("C:/Users/saksh/Sakshi/VS Code/Anti/shape_predictor_68_face_landmarks (1).dat")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Initialize alarm
        pygame.mixer.init()
        self.alarm_sound = pygame.mixer.Sound("C:/Users/saksh/Sakshi/VS Code/Anti/alarm.wav")
        
        # Drowsiness parameters
        self.EYE_AR_THRESH = 0.25  # Eye aspect ratio threshold
        self.EYE_AR_CONSEC_FRAMES = 48  # Consecutive frames threshold
        self.HEAD_THRESH = 25  # Head tilt threshold in degrees
        
        # Tracking variables
        self.ear_counter = 0
        self.head_counter = 0
        self.ALARM_ON = False
    
    def eye_aspect_ratio(self, eye):
        """Calculate eye aspect ratio."""
        A = dist.euclidean(eye[1], eye[5])  # Vertical distances
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
        return (A + B) / (2.0 * C)
    
    def get_head_pose_angle(self, shape):
        """Calculate head tilt angle based on facial landmarks."""
        nose_bridge = shape[27:31]
        nose_tip = shape[30]
        
        # Calculate nose bridge mean position
        nose_bridge_mean_x = np.mean([p[0] for p in nose_bridge])
        nose_bridge_mean_y = np.mean([p[1] for p in nose_bridge])
        
        # Calculate tilt angle in degrees
        angle = np.arctan2(
            nose_tip[1] - nose_bridge_mean_y,
            nose_tip[0] - nose_bridge_mean_x
        ) * 180 / np.pi
        return abs(angle)
    
    def detect_drowsiness(self):
        """Main drowsiness detection loop."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 0)
            
            for face in faces:
                shape = self.predictor(gray, face)
                shape = np.array([(p.x, p.y) for p in shape.parts()])
                
                # Extract eye regions
                left_eye = shape[42:48]
                right_eye = shape[36:42]
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                
                # Get head tilt angle
                head_angle = self.get_head_pose_angle(shape)
                
                # EAR logic
                if ear < self.EYE_AR_THRESH:
                    self.ear_counter += 1
                else:
                    self.ear_counter = 0
                
                # Head tilt logic
                if head_angle > self.HEAD_THRESH:
                    self.head_counter += 1
                else:
                    self.head_counter = 0
                
                # Alarm logic
                if self.ear_counter >= self.EYE_AR_CONSEC_FRAMES or self.head_counter >= self.EYE_AR_CONSEC_FRAMES:
                    if not self.ALARM_ON:
                        self.ALARM_ON = True
                        self.alarm_sound.play(-1)
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    if self.ALARM_ON:
                        self.ALARM_ON = False
                        self.alarm_sound.stop()
                
                # Visualize metrics
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Head Angle: {head_angle:.2f}", (10, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Drowsiness Detector", frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
    
    def run(self):
        """Start the drowsiness detection."""
        try:
            self.detect_drowsiness()
        except Exception as e:
            print(f"Error: {e}")

# Main execution
if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.run()
