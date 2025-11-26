import cv2
import numpy as np
import time
import mediapipe as mp

class MediaPipeRehabTracker:
    """
    Stroke rehabilitation tracking using MediaPipe only
    Tracks all the metrics mentioned in your document
    """
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False
        )
        
        # Metrics storage
        self.metrics_history = []
        self.start_time = None
        self.smoothed_nose_to_ear = None   # running estimate of nose→ear vector
        self.ear_smoothing_alpha = 0.2     # how fast it adapts (0.1–0.3 is reasonable)


        
    def calculate_trunk_rotation(self, landmarks):
        """Calculate trunk rotation (yaw) for cross-midline reaching"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Calculate shoulder line angle
        shoulder_vector = np.array([
            right_shoulder.x - left_shoulder.x,
            right_shoulder.y - left_shoulder.y
        ])
        
        # Calculate hip line angle
        hip_vector = np.array([
            right_hip.x - left_hip.x,
            right_hip.y - left_hip.y
        ])
        
        # Trunk rotation is the angle between shoulder and hip lines
        shoulder_angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])
        hip_angle = np.arctan2(hip_vector[1], hip_vector[0])
        
        rotation = np.degrees(shoulder_angle - hip_angle)
        
        return rotation
    
    def calculate_postural_alignment(self, landmarks):
        """
        Posture metric based on hip–shoulder–ear alignment on the most reliable side,
        with an adaptive, smoothed ear position anchored to the nose.

        - Choose left or right based on combined shoulder+ear visibility.
        - Torso lean: angle between hip→shoulder and vertical.
        - Neck alignment: angle between hip→shoulder and shoulder→ear_eff.
        - ear_eff is built from a smoothed nose→ear vector:
            * when ear is good -> update smoothing
            * when ear is noisy -> reuse previous smoothed vector
        - Returns a combined score in degrees: lower = better posture.
        """
        import numpy as np

        # --- Landmarks we need ---
        l_sh  = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_sh  = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        nose  = landmarks[self.mp_pose.PoseLandmark.NOSE.value]

        def vis(lm):
            return getattr(lm, "visibility", 1.0)

        # --- 1) Pick side with better shoulder+ear visibility ---
        left_score  = vis(l_sh) + vis(l_ear)
        right_score = vis(r_sh) + vis(r_ear)

        if left_score >= right_score:
            sh, hip, ear = l_sh, l_hip, l_ear
        else:
            sh, hip, ear = r_sh, r_hip, r_ear

        # --- 2) Torso vector hip→shoulder ---
        v_torso = np.array([sh.x - hip.x, sh.y - hip.y], dtype=float)
        v_up    = np.array([0.0, -1.0], dtype=float)  # image "up"

        eps = 1e-6
        torso_norm = np.linalg.norm(v_torso)
        up_norm    = np.linalg.norm(v_up)
        if torso_norm < eps:
            # Can't say anything meaningful
            return 0.0

        # --- 3) Build an adaptive, smoothed nose→ear vector ---
        raw_n2e = np.array([ear.x - nose.x, ear.y - nose.y], dtype=float)
        raw_n2e_norm = np.linalg.norm(raw_n2e)

        # Heuristics: what counts as a "good" ear sample?
        ear_visible   = vis(ear) >= 0.5
        ear_reasonable = 0.01 < raw_n2e_norm < 0.5  # normalized-image distance range

        if ear_visible and ear_reasonable:
            # Update exponential moving average of nose→ear
            if self.smoothed_nose_to_ear is None:
                self.smoothed_nose_to_ear = raw_n2e.copy()
            else:
                alpha = getattr(self, "ear_smoothing_alpha", 0.2)
                self.smoothed_nose_to_ear = (
                    (1.0 - alpha) * self.smoothed_nose_to_ear + alpha * raw_n2e
                )

        # Decide which vector to use for ear position
        if self.smoothed_nose_to_ear is not None:
            # Reconstruct a stable ear from nose + smoothed offset
            ear_eff_x = nose.x + self.smoothed_nose_to_ear[0]
            ear_eff_y = nose.y + self.smoothed_nose_to_ear[1]
        else:
            # Early frames: no history yet, fall back to raw ear
            ear_eff_x, ear_eff_y = ear.x, ear.y

        # --- 4) Head/neck vector shoulder→effective ear ---
        v_head = np.array([ear_eff_x - sh.x, ear_eff_y - sh.y], dtype=float)
        head_norm = np.linalg.norm(v_head)

        # --- 5) Torso lean angle (hip→shoulder vs vertical) ---
        dot_torso_up = np.dot(v_torso, v_up) / (torso_norm * up_norm)
        dot_torso_up = np.clip(dot_torso_up, -1.0, 1.0)
        torso_angle_deg = np.degrees(np.arccos(dot_torso_up))
        # 0° = perfectly vertical torso

        # --- 6) Neck alignment: hip→shoulder vs shoulder→ear_eff ---
        neck_excess_deg = 0.0
        if head_norm > eps:
            dot_th = np.dot(v_torso, v_head) / (torso_norm * head_norm)
            dot_th = np.clip(dot_th, -1.0, 1.0)
            neck_angle_deg = np.degrees(np.arccos(dot_th))

            # Allow some "neutral" neck curvature (e.g. 10°) without penalty
            NEUTRAL_NECK = 10.0
            neck_excess_deg = max(0.0, neck_angle_deg - NEUTRAL_NECK)

        # --- 7) Combined posture score ---
        NECK_WEIGHT = 0.7  # how much neck contributes vs torso
        posture_score = torso_angle_deg + NECK_WEIGHT * neck_excess_deg

        return posture_score






    def calculate_shoulder_distance(self, landmarks):
        """Calculate shoulder distance for range of motion tracking"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        distance = np.sqrt(
            (right_shoulder.x - left_shoulder.x)**2 +
            (right_shoulder.y - left_shoulder.y)**2
        )
        
        return distance
    
    def calculate_shoulder_heights(self, landmarks):
        """Calculate individual shoulder heights for posture assessment"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        return {
            'left_height': left_shoulder.y,
            'right_height': right_shoulder.y,
            'height_difference': abs(left_shoulder.y - right_shoulder.y)
        }
    
    def detect_cross_midline_reaching(self, landmarks):
        """Detect if either hand crosses body midline"""
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # Calculate body midline (x-coordinate)
        midline_x = (left_shoulder.x + right_shoulder.x) / 2
        
        left_crosses = left_wrist.x < midline_x    # left hand crosses to subject’s right
        right_crosses = right_wrist.x > midline_x  # right hand crosses to subject’s left
        
        # Calculate how far the hand crossed
        left_cross_distance = (left_wrist.x - midline_x) if left_crosses else 0
        right_cross_distance = (midline_x - right_wrist.x) if right_crosses else 0
        
        return {
            'left_crosses': left_crosses,
            'right_crosses': right_crosses,
            'any_crossing': left_crosses or right_crosses,
            'left_cross_distance': left_cross_distance,
            'right_cross_distance': right_cross_distance
        }
    
    def assess_bilateral_arm_use(self, landmarks):
        """Assess if both arms are being used in coordination"""
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # Calculate wrist heights relative to shoulders
        left_wrist_relative = left_wrist.y - left_shoulder.y
        right_wrist_relative = right_wrist.y - right_shoulder.y
        
        # Check if wrists are at similar heights (indicating bilateral use)
        height_similarity = abs(left_wrist_relative - right_wrist_relative)
        
        # Calculate wrist-to-wrist distance
        wrist_distance = np.sqrt(
            (right_wrist.x - left_wrist.x)**2 +
            (right_wrist.y - left_wrist.y)**2
        )
        
        return {
            'height_similarity': height_similarity,
            'wrist_distance': wrist_distance,
            'coordinated': height_similarity < 0.1  # Threshold for coordination
        }
    
    def calculate_shoulder_range_of_motion(self, landmarks):
        """Calculate shoulder range of motion"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        
        # Calculate angles
        def calculate_angle(shoulder, elbow):
            vector = np.array([elbow.x - shoulder.x, elbow.y - shoulder.y])
            angle = np.arctan2(vector[1], vector[0])
            return np.degrees(angle)
        
        left_angle = calculate_angle(left_shoulder, left_elbow)
        right_angle = calculate_angle(right_shoulder, right_elbow)
        
        return {
            'left_shoulder_angle': left_angle,
            'right_shoulder_angle': right_angle
        }
    
    def process_frame(self, frame):
        """Process a single frame and return all metrics"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        metrics = {
            'timestamp': time.time() - self.start_time if self.start_time else 0
        }
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Calculate all metrics
            metrics['trunk_rotation'] = self.calculate_trunk_rotation(landmarks)
            metrics['postural_alignment'] = self.calculate_postural_alignment(landmarks)
            metrics['shoulder_distance'] = self.calculate_shoulder_distance(landmarks)
            metrics['shoulder_heights'] = self.calculate_shoulder_heights(landmarks)
            metrics['cross_midline'] = self.detect_cross_midline_reaching(landmarks)
            metrics['bilateral_arm_use'] = self.assess_bilateral_arm_use(landmarks)
            metrics['shoulder_rom'] = self.calculate_shoulder_range_of_motion(landmarks)
            
            # Draw pose landmarks on frame
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
            
            # Draw metrics on frame
            self._draw_metrics_on_frame(frame, metrics)
        else:
            # No pose detected
            cv2.putText(frame, "No pose detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame, metrics
    
    def _draw_metrics_on_frame(self, frame, metrics):
        """Draw metrics overlay on video frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        y_pos = 35
        line_height = 30
        
        # Trunk rotation
        if 'trunk_rotation' in metrics:
            text = f"Trunk Rotation: {metrics['trunk_rotation']:.1f} deg"
            color = (0, 255, 0) if abs(metrics['trunk_rotation']) < 10 else (0, 165, 255)
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += line_height
        
        # Postural alignment
        if 'postural_alignment' in metrics:
            text = f"Posture Tilt: {metrics['postural_alignment']:.1f} deg"
            color = (0, 255, 0) if metrics['postural_alignment'] < 10 else (0, 165, 255)
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += line_height
        
        # Cross-midline reaching
        if 'cross_midline' in metrics:
            crossing = metrics['cross_midline']
            if crossing['any_crossing']:
                text = "Cross-Midline: ACTIVE"
                cv2.putText(frame, text, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                text = "Cross-Midline: None"
                cv2.putText(frame, text, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            y_pos += line_height
        
        # Bilateral arm use
        if 'bilateral_arm_use' in metrics:
            bilateral = metrics['bilateral_arm_use']
            if bilateral['coordinated']:
                text = "Bilateral Use: YES"
                color = (0, 255, 0)
            else:
                text = "Bilateral Use: NO"
                color = (128, 128, 128)
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += line_height
        
        # Shoulder alignment
        if 'shoulder_heights' in metrics:
            heights = metrics['shoulder_heights']
            diff = heights['height_difference'] * 100  # Convert to percentage
            text = f"Shoulder Level: {diff:.1f}%"
            color = (0, 255, 0) if diff < 5 else (0, 165, 255)
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += line_height
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save", (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run_tracking_session(self, camera_index=0, duration_seconds=None):
        """
        Run a tracking session
        
        Args:
            camera_index: Camera device index (usually 0)
            duration_seconds: Session duration (None = infinite)
        
        Returns:
            list: All collected metrics
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return []
        
        self.start_time = time.time()
        self.metrics_history = []
        
        end_time = self.start_time + duration_seconds if duration_seconds else None
        
        print("Tracking started. Press 'q' to quit, 's' to save metrics.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Process frame
            processed_frame, metrics = self.process_frame(frame)
            self.metrics_history.append(metrics)
            
            # Display
            cv2.imshow('MediaPipe Rehabilitation Tracking', processed_frame)
            
            # Check for quit or save
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_metrics()
                print("Metrics saved!")
            
            # Check duration
            if end_time and time.time() >= end_time:
                print("Session duration reached")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return self.metrics_history
    
    def save_metrics(self, filename='rehab_metrics.csv'):
        """Save metrics to CSV file"""
        import pandas as pd
        
        if not self.metrics_history:
            print("No metrics to save")
            return
        
        # Flatten nested dictionaries for CSV
        flat_metrics = []
        for m in self.metrics_history:
            flat = {'timestamp': m.get('timestamp', 0)}
            
            # Add simple metrics
            for key in ['trunk_rotation', 'postural_alignment', 'shoulder_distance']:
                if key in m:
                    flat[key] = m[key]
            
            # Flatten nested metrics
            if 'cross_midline' in m:
                for k, v in m['cross_midline'].items():
                    flat[f'cross_midline_{k}'] = v
            
            if 'bilateral_arm_use' in m:
                for k, v in m['bilateral_arm_use'].items():
                    flat[f'bilateral_{k}'] = v
            
            if 'shoulder_heights' in m:
                for k, v in m['shoulder_heights'].items():
                    flat[f'shoulder_{k}'] = v
            
            if 'shoulder_rom' in m:
                for k, v in m['shoulder_rom'].items():
                    flat[f'rom_{k}'] = v
            
            flat_metrics.append(flat)
        
        df = pd.DataFrame(flat_metrics)
        df.to_csv(filename, index=False)
        print(f"Saved {len(flat_metrics)} data points to {filename}")