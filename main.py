import cv2
import mediapipe as mp
import numpy as np

# ==========================================
#        USER CONFIGURATION SECTION
# ==========================================

EXERCISE_TYPE = 'bicep'  # Options: 'bicep' OR 'lateral' 
VIDEO_PATH = 'demo_curl.mp4' # Use 0 to access webcam for real-time detection

# ==========================================

class SmartanTrainer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.counter = 0
        self.stage = None
        self.feedback = "Start"

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
        return angle

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not results.pose_landmarks:
            return image

        landmarks = results.pose_landmarks.landmark

        # Get Keypoint Coordinates
        shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]

        h, w, _ = image.shape
        feedback_color = (0, 255, 0)
        elbow_warning_text = "" 

        # ==========================================
        #           LOGIC: BICEP CURL   
        # ==========================================
        if EXERCISE_TYPE == 'bicep':
            elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
            upper_arm_angle = self.calculate_angle(elbow, shoulder, hip)
            back_angle = self.calculate_angle(shoulder, hip, knee)
            
            # Rep Counting (Adjusted Thresholds)
            if elbow_angle > 150: # Relaxed from 160
                self.stage = "down"
            if elbow_angle < 60 and self.stage == 'down': # Relaxed from 30
                self.stage = "up"
                self.counter += 1
            
            # Feedback Rules
            if back_angle < 170: 
                self.feedback = "Fix Back!"
                feedback_color = (0, 0, 255)
            else:
                self.feedback = "Good Form"
                feedback_color = (0, 255, 0)

            if upper_arm_angle > 25:
                elbow_warning_text = "FIX ELBOW!" 
            
            # Display Angle
            cv2.putText(image, str(int(elbow_angle)), 
                        tuple(np.multiply(elbow, [w, h]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # ==========================================
        #           LOGIC: LATERAL RAISE
        # ==========================================
        elif EXERCISE_TYPE == 'lateral':
            arm_angle = self.calculate_angle(hip, shoulder, elbow)
            wrist_y = wrist[1]
            shoulder_y = shoulder[1]
            
            # Feedback Logic
            upper_limit = shoulder_y - 0.10

            if wrist_y < upper_limit: 
                self.feedback = "LOWER ARMS!"
                feedback_color = (0, 0, 255)
            elif arm_angle > 65:
                self.feedback = "Good Hold"
                feedback_color = (0, 255, 0)
            else:
                self.feedback = "Go Up"
                feedback_color = (255, 255, 255)

            # Rep Counting
            if arm_angle < 30: self.stage = "down"
            if arm_angle > 75 and self.stage == 'down':
                self.stage = "up"
                self.counter += 1
            
            # Display Angle
            cv2.putText(image, str(int(arm_angle)), 
                        tuple(np.multiply(shoulder, [w, h]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # ==========================================
        #              UI DRAWING
        # ==========================================
        
        # Main Info Box
        cv2.rectangle(image, (0,0), (w, 80), (245, 117, 16), -1)
        
        # Reps
        cv2.putText(image, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Feedback
        cv2.putText(image, 'FEEDBACK', (150,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, self.feedback, (150,60), cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_color, 2, cv2.LINE_AA)

        # Mode Indicator
        cv2.putText(image, f"MODE: {EXERCISE_TYPE.upper()}", (w - 200, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Warning Box (Bicep Only)
        if elbow_warning_text != "":
            cv2.rectangle(image, (0, 100), (250, 150), (0, 0, 255), -1) 
            cv2.putText(image, elbow_warning_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Skeleton
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        return image

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    trainer = SmartanTrainer()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video Finished")
            break
        
        frame = cv2.resize(frame, (800, 600))
        output_image = trainer.process_frame(frame)
        cv2.imshow('Smartan AI Trainer', output_image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()