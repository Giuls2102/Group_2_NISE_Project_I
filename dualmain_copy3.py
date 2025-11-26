import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mediapipe_rehab_tracker import MediaPipeRehabTracker
import pandas as pd
import cv2
import numpy as np
import time
from collections import deque
from datetime import datetime
import mediapipe as mp

# NEW: networking & CSV for Unity-controlled recording
import socket
import threading
import csv

# -------------------------------------------------------------------
# UDP control from Unity for raw landmark recording
# -------------------------------------------------------------------
UDP_IP = "127.0.0.1"
UDP_PORT = 5007  # must match MediapipeBridge.pythonPort in Unity

mp_is_recording = False
mp_current_trial = None
mp_file_handle = None
mp_start_time = 0.0
mp_lock = threading.Lock()
mp_current_filepath = None
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mediapipe_logs")
mp_signal_start = False
mp_signal_stop  = False

def listen_for_unity():
    global mp_signal_start, mp_signal_stop, mp_start_time
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"[Python Mediapipe] Listening for Unity on {UDP_IP}:{UDP_PORT}")

    while True:
        data, _ = sock.recvfrom(1024)
        msg = data.decode("utf-8").strip()
        print("[Python Mediapipe] Received:", msg)
        m = msg.upper()
        with mp_lock:
            if m.startswith("START") or m == "S":
                mp_signal_start = True
                mp_start_time = time.time()
            elif m.startswith("STOP") or m == "E":
                mp_signal_stop = True



def record_landmarks_if_needed(camera_name, landmarks):
    """
    Write raw pose landmarks to CSV if Unity has sent START and not yet STOP.

    camera_name: "front" or "side" (or any string label)
    landmarks:   iterable of Mediapipe landmarks with .x, .y, .z
    """
    global mp_is_recording, mp_file_handle, mp_start_time

    with mp_lock:
        if not mp_is_recording or mp_file_handle is None or landmarks is None:
            return

        t = time.time() - mp_start_time
        writer = csv.writer(mp_file_handle)

        for i, lm in enumerate(landmarks):
            # Mediapipe landmarks typically have .x, .y, .z
            x, y, z = lm.x, lm.y, lm.z
            writer.writerow([f"{t:.4f}", camera_name, i, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])


# start UDP listener in background thread
threading.Thread(target=listen_for_unity, daemon=True).start()


# Import triangulation - with fallback
try:
    from triangulation.triangulation_integration import init_stereo_from_calib, process_frame_pair_and_triangulate
    TRIANGULATION_AVAILABLE = True
except ImportError:
    print("⚠️  Triangulation module not found - running in 2D mode only")
    TRIANGULATION_AVAILABLE = False


# Smoothing classes
class LandmarkSmoother:
    """Exponential Moving Average smoother for landmarks"""
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.prev_landmarks = None
    
    def smooth(self, landmarks_array):
        if landmarks_array is None or landmarks_array.size == 0:
            return landmarks_array
        
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks_array.copy()
            return landmarks_array
        
        if self.prev_landmarks.shape != landmarks_array.shape:
            self.prev_landmarks = landmarks_array.copy()
            return landmarks_array
        
        smoothed = self.alpha * landmarks_array + (1 - self.alpha) * self.prev_landmarks
        self.prev_landmarks = smoothed.copy()
        return smoothed
    
    def reset(self):
        self.prev_landmarks = None


class Points3DSmoother:
    """Rolling average smoother for 3D points"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
    
    def smooth(self, pts3d):
        if pts3d is None or pts3d.size == 0:
            return pts3d
        
        self.history.append(pts3d.copy())
        
        if len(self.history) == 0:
            return pts3d
        
        smoothed = np.mean(list(self.history), axis=0)
        return smoothed
    
    def reset(self):
        self.history.clear()


class DualCameraRehabTracker:
    """
    Enhanced dual-camera tracking system with:
    - Preview mode before recording
    - Optional triangulation (toggle on/off)
    - TXT output for Unity (with % and units)
    - CSV output for detailed metrics
    - Global timestamp synchronization
    """
    def __init__(self):
        self.tracker_front = MediaPipeRehabTracker()
        self.tracker_side = MediaPipeRehabTracker()
        self.start_time = None
        self.global_start_time = None
        self.combined_metrics = []
        
        # Recording state
        self.is_recording = False
        self.is_preview_mode = True
        
        # Triangulation state
        self.use_triangulation = False
        self.triangulation_initialized = False
        self.maps1 = None
        self.maps2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.K1 = None
        self.D1 = None
        self.K2 = None
        self.D2 = None
        
        # Camera landmarks
        self.front_landmarks = [11, 12, 13, 14, 15, 16, 23, 24]
        self.side_landmarks = [0, 7, 8, 11, 12, 23, 24]
        
        # Smoothers
        self.smoother_front_2d = LandmarkSmoother(alpha=0.4)
        self.smoother_side_2d = LandmarkSmoother(alpha=0.4)
        self.smoother_3d = Points3DSmoother(window_size=5)
        
        # Tracking loss counters
        self.frames_without_detection_front = 0
        self.frames_without_detection_side = 0
        
        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start_datetime = None
        self.trial_number = 0
        
        # Session statistics
        self.session_stats = {
            'front_camera': {
                'cross_midline_count': 0,
                'bilateral_coordination_scores': [],
                'shoulder_alignment_scores': [],
                'trunk_rotation_angles': [],
            },
            'side_camera': {
                'postural_lean_angles': [],
                'posture_good_count': 0,
                'posture_total_count': 0,
            }
        }
    
    def toggle_triangulation(self):
        """Toggle triangulation on/off"""
        if not TRIANGULATION_AVAILABLE:
            print("⚠️  Triangulation not available - missing module")
            return False
        
        if not self.is_preview_mode:
            print("⚠️  Cannot toggle during recording")
            return self.use_triangulation
        
        self.use_triangulation = not self.use_triangulation
        status = "ON ✅" if self.use_triangulation else "OFF"
        print(f"🔄 Triangulation: {status}")
        return self.use_triangulation
    
    def start_recording(self):
        """Start recording metrics"""
        if not self.is_recording:
            self.is_recording = True
            self.is_preview_mode = False
            self.start_time = time.time()
            self.global_start_time = time.time()
            self.session_start_datetime = datetime.now()
            self.combined_metrics = []
            self.session_stats = {
                'front_camera': {
                    'cross_midline_count': 0,
                    'bilateral_coordination_scores': [],
                    'shoulder_alignment_scores': [],
                    'trunk_rotation_angles': [],
                },
                'side_camera': {
                    'postural_lean_angles': [],
                    'posture_good_count': 0,
                    'posture_total_count': 0,
                }
            }
            print(f"🔴 RECORDING STARTED - {self.session_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Global Start Time: {self.global_start_time:.6f}")
    
    def stop_recording(self):
        """Stop recording metrics"""
        if self.is_recording:
            self.is_recording = False
            self.is_preview_mode = True
            print("⏹️  RECORDING STOPPED")
    
    def process_front_camera(self, frame, results=None):
        """Process front camera - SHOULDERS AND ARMS"""
        if results is None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.tracker_front.pose.process(frame_rgb)
        
        metrics = {
            'timestamp': time.time() - self.start_time if self.start_time else 0,
            'global_time': time.time()
        }
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Smooth landmarks
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            smoothed_landmarks_array = self.smoother_front_2d.smooth(landmarks_array)
            
            for i, lm in enumerate(landmarks):
                lm.x = smoothed_landmarks_array[i, 0]
                lm.y = smoothed_landmarks_array[i, 1]
                lm.z = smoothed_landmarks_array[i, 2]
            
            self.frames_without_detection_front = 0

            # NEW: record raw landmarks to CSV if Unity requested it
            record_landmarks_if_needed("front", landmarks)
            
            # Calculate metrics
            metrics['shoulder_distance'] = self.tracker_front.calculate_shoulder_distance(landmarks)
            metrics['shoulder_heights'] = self.tracker_front.calculate_shoulder_heights(landmarks)
            metrics['cross_midline'] = self.tracker_front.detect_cross_midline_reaching(landmarks)
            metrics['bilateral_arm_use'] = self.tracker_front.assess_bilateral_arm_use(landmarks)
            metrics['shoulder_rom'] = self.tracker_front.calculate_shoulder_range_of_motion(landmarks)
            metrics['trunk_rotation'] = self.tracker_front.calculate_trunk_rotation(landmarks)
            
            # Update session stats only when recording
            if self.is_recording:
                if metrics['cross_midline'].get('any_crossing', False):
                    self.session_stats['front_camera']['cross_midline_count'] += 1
                
                if metrics['bilateral_arm_use'].get('coordinated', False):
                    self.session_stats['front_camera']['bilateral_coordination_scores'].append(1)
                else:
                    self.session_stats['front_camera']['bilateral_coordination_scores'].append(0)
                
                self.session_stats['front_camera']['shoulder_alignment_scores'].append(
                    metrics['shoulder_heights'].get('height_difference', 0) * 100
                )
                self.session_stats['front_camera']['trunk_rotation_angles'].append(
                    abs(metrics['trunk_rotation'])
                )
            
            # Draw
            self._draw_selective_landmarks(frame, results.pose_landmarks, self.front_landmarks, (0, 255, 0))
            self.draw_front_camera_overlay(frame, metrics)
        else:
            self.frames_without_detection_front += 1
            if self.frames_without_detection_front > 30:
                self.smoother_front_2d.reset()
            
            cv2.putText(frame, "No pose detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame, metrics, results
    
    def process_side_camera(self, frame, results=None):
        """Process side camera - POSTURE with enhanced visualization"""
        if results is None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.tracker_side.pose.process(frame_rgb)
        
        metrics = {
            'timestamp': time.time() - self.start_time if self.start_time else 0,
            'global_time': time.time()
        }
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Smooth landmarks
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            smoothed_landmarks_array = self.smoother_side_2d.smooth(landmarks_array)
            
            for i, lm in enumerate(landmarks):
                lm.x = smoothed_landmarks_array[i, 0]
                lm.y = smoothed_landmarks_array[i, 1]
                lm.z = smoothed_landmarks_array[i, 2]
            
            self.frames_without_detection_side = 0

            # NEW: record raw landmarks to CSV if Unity requested it
            record_landmarks_if_needed("side", landmarks)
            
            # Calculate posture metrics
            metrics['postural_alignment'] = self.tracker_side.calculate_postural_alignment(landmarks)
            metrics['trunk_rotation'] = self.tracker_side.calculate_trunk_rotation(landmarks)
            metrics['shoulder_rom'] = self.tracker_side.calculate_shoulder_range_of_motion(landmarks)
            
            # Update session stats only when recording
            if self.is_recording:
                self.session_stats['side_camera']['postural_lean_angles'].append(
                    metrics['postural_alignment']
                )
                self.session_stats['side_camera']['posture_total_count'] += 1
                if metrics['postural_alignment'] < 15:
                    self.session_stats['side_camera']['posture_good_count'] += 1
            
            # Draw ENHANCED posture visualization
            self._draw_enhanced_posture_visualization(frame, landmarks)
            self.draw_side_camera_overlay(frame, metrics)
        else:
            self.frames_without_detection_side += 1
            if self.frames_without_detection_side > 30:
                self.smoother_side_2d.reset()
            
            cv2.putText(frame, "No pose detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame, metrics, results
    
    def _draw_selective_landmarks(self, frame, pose_landmarks, landmark_indices, color):
        """Draw only specified landmarks"""
        h, w, _ = frame.shape
        
        for idx in landmark_indices:
            if idx < len(pose_landmarks.landmark):
                landmark = pose_landmarks.landmark[idx]
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 2)
        
        # Draw connections
        if 11 in landmark_indices:
            connections = [
                (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
                (11, 23), (12, 24), (23, 24)
            ]
            
            for connection in connections:
                pt1_idx, pt2_idx = connection
                if pt1_idx < len(pose_landmarks.landmark) and pt2_idx < len(pose_landmarks.landmark):
                    pt1 = pose_landmarks.landmark[pt1_idx]
                    pt2 = pose_landmarks.landmark[pt2_idx]
                    
                    pt1_coords = (int(pt1.x * w), int(pt1.y * h))
                    pt2_coords = (int(pt2.x * w), int(pt2.y * h))
                    
                    cv2.line(frame, pt1_coords, pt2_coords, (255, 255, 255), 2)
    
    def _draw_enhanced_posture_visualization(self, frame, landmarks):
        """Draw ENHANCED spine/posture visualization with all reference lines"""
        h, w, _ = frame.shape
        
        # Get landmarks
        nose = landmarks[0]
        left_ear = landmarks[7]
        right_ear = landmarks[8]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Choose visible ear
        ear = left_ear if left_ear.visibility > right_ear.visibility else right_ear
        if max(left_ear.visibility, right_ear.visibility) < 0.3:
            ear = nose
        
        # Midpoints
        mid_shoulder_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
        mid_shoulder_y = int((left_shoulder.y + right_shoulder.y) / 2 * h)
        mid_hip_x = int((left_hip.x + right_hip.x) / 2 * w)
        mid_hip_y = int((left_hip.y + right_hip.y) / 2 * h)
        
        ear_coords = (int(ear.x * w), int(ear.y * h))
        nose_coords = (int(nose.x * w), int(nose.y * h))
        
        # 1. Draw TORSO line (hip -> shoulder) in GREEN
        cv2.line(frame, (mid_hip_x, mid_hip_y), (mid_shoulder_x, mid_shoulder_y), (0, 255, 0), 3)
        
        # 2. Draw HEAD/NECK line (shoulder -> ear) in CYAN
        cv2.line(frame, (mid_shoulder_x, mid_shoulder_y), ear_coords, (0, 255, 255), 3)
        
        # 3. Draw NOSE landmark and line to shoulder in YELLOW
        cv2.line(frame, nose_coords, (mid_shoulder_x, mid_shoulder_y), (255, 255, 0), 2)
        
        # 4. Draw VERTICAL REFERENCE LINE from hip (ideal posture) in GREEN
        reference_height = 250
        cv2.line(frame, (mid_hip_x, mid_hip_y), (mid_hip_x, mid_hip_y - reference_height), 
                 (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "IDEAL", (mid_hip_x + 10, mid_hip_y - reference_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 5. Draw key points with labels
        cv2.circle(frame, nose_coords, 6, (255, 255, 0), -1)
        cv2.putText(frame, "NOSE", (nose_coords[0] + 10, nose_coords[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        cv2.circle(frame, ear_coords, 6, (0, 255, 255), -1)
        cv2.putText(frame, "EAR", (ear_coords[0] + 10, ear_coords[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        cv2.circle(frame, (mid_shoulder_x, mid_shoulder_y), 8, (255, 255, 0), -1)
        cv2.putText(frame, "MID-SHOULDER", (mid_shoulder_x + 10, mid_shoulder_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        cv2.circle(frame, (mid_hip_x, mid_hip_y), 8, (0, 255, 0), -1)
        cv2.putText(frame, "MID-HIP", (mid_hip_x + 10, mid_hip_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # 6. Draw angle indicator for torso lean
        # Calculate angle from vertical
        torso_vec = np.array([mid_shoulder_x - mid_hip_x, mid_shoulder_y - mid_hip_y])
        vertical_vec = np.array([0, -1])
        
        torso_norm = np.linalg.norm(torso_vec)
        if torso_norm > 0:
            torso_angle = np.degrees(np.arccos(
                np.clip(np.dot(torso_vec / torso_norm, vertical_vec), -1.0, 1.0)
            ))
            
            # Draw arc to show angle
            arc_radius = 80
            if torso_angle > 5:  # Only show if there's significant lean
                cv2.ellipse(frame, (mid_hip_x, mid_hip_y), (arc_radius, arc_radius),
                           -90, 0, torso_angle if mid_shoulder_x > mid_hip_x else -torso_angle,
                           (255, 0, 255), 2)
                
                # Draw angle text
                angle_text_x = mid_hip_x + 50 if mid_shoulder_x > mid_hip_x else mid_hip_x - 100
                cv2.putText(frame, f"{torso_angle:.1f}°", (angle_text_x, mid_hip_y - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    def draw_front_camera_overlay(self, frame, metrics):
        """Draw front camera metrics"""
        h, w = frame.shape[:2]
        y_pos = 35
        line_height = 30
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        cv2.putText(frame, "FRONT VIEW - Upper Body", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += line_height + 5
        
        if 'cross_midline' in metrics:
            crossing = metrics['cross_midline']
            if crossing.get('any_crossing', False):
                text = "Cross-Midline: ACTIVE"
                cv2.putText(frame, text, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                text = "Cross-Midline: None"
                cv2.putText(frame, text, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            y_pos += line_height
        
        if 'bilateral_arm_use' in metrics:
            bilateral = metrics['bilateral_arm_use']
            if bilateral.get('coordinated', False):
                text = "Bilateral: YES"
                color = (0, 255, 0)
            else:
                text = "Bilateral: NO"
                color = (128, 128, 128)
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += line_height
        
        if 'shoulder_heights' in metrics:
            heights = metrics['shoulder_heights']
            diff = heights.get('height_difference', 0) * 100
            text = f"Shoulder Level: {diff:.1f}%"
            color = (0, 255, 0) if diff < 5 else (0, 165, 255)
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += line_height
        
        if 'trunk_rotation' in metrics:
            text = f"Trunk Rotation: {metrics['trunk_rotation']:.1f}°"
            color = (0, 255, 0) if abs(metrics['trunk_rotation']) < 10 else (0, 165, 255)
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def draw_side_camera_overlay(self, frame, metrics):
        """Draw side camera metrics"""
        h, w = frame.shape[:2]
        y_pos = 35
        line_height = 30
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 190), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        cv2.putText(frame, "SIDE VIEW - Posture", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        y_pos += line_height + 5
        
        if 'postural_alignment' in metrics:
            lean = metrics['postural_alignment']
            text = f"Posture score: {lean:.1f} deg"
            color = (0, 255, 0) if lean < 10 else (0, 165, 255)
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += line_height
            
            if lean < 20:
                assessment = "Posture: EXCELLENT"
                color = (0, 255, 0)
            elif lean < 40:
                assessment = "Posture: GOOD"
                color = (0, 255, 255)
            else:
                assessment = "Posture: NEEDS WORK"
                color = (0, 165, 255)
            cv2.putText(frame, assessment, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += line_height
        
        if 'shoulder_rom' in metrics:
            rom = metrics['shoulder_rom']
            left_angle = rom.get('left_shoulder_angle', 0)
            right_angle = rom.get('right_shoulder_angle', 0)
            
            text = f"Shoulder (L/R): {left_angle:.0f}° / {right_angle:.0f}°"
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def extract_front_camera_metrics(self, metrics):
        front_specific = {}
        if 'cross_midline' in metrics:
            front_specific['cross_midline'] = metrics['cross_midline']
        if 'bilateral_arm_use' in metrics:
            front_specific['bilateral_coordination'] = metrics['bilateral_arm_use']
        if 'shoulder_heights' in metrics:
            front_specific['shoulder_alignment'] = metrics['shoulder_heights']
        if 'trunk_rotation' in metrics:
            front_specific['trunk_rotation_frontal'] = metrics['trunk_rotation']
        return front_specific
    
    def extract_side_camera_metrics(self, metrics):
        side_specific = {}
        if 'postural_alignment' in metrics:
            side_specific['postural_lean'] = metrics['postural_alignment']
        if 'shoulder_rom' in metrics:
            side_specific['shoulder_elevation_sagittal'] = metrics['shoulder_rom']
        if 'trunk_rotation' in metrics:
            side_specific['trunk_lateral_flexion'] = metrics['trunk_rotation']
        return side_specific
    
    def combine_metrics(self, front_metrics, side_metrics, timestamp):
        combined = {
            'timestamp': timestamp,
            'global_time': time.time(),
            'front_camera': front_metrics,
            'side_camera': side_metrics
        }
        
        combined['fusion'] = {}
        
        if 'shoulder_alignment' in front_metrics and 'postural_lean' in side_metrics:
            shoulder_diff = front_metrics['shoulder_alignment'].get('height_difference', 0)
            lean_angle = side_metrics['postural_lean']
            
            combined['fusion']['overall_posture_score'] = (shoulder_diff * 100) + lean_angle
            combined['fusion']['posture_assessment'] = 'Good' if combined['fusion']['overall_posture_score'] < 15 else 'Needs Improvement'
        
        return combined
    
    def save_unity_summary_txt(self, output_dir='.'):
        """Save session averages in TXT format for Unity with % and units"""
        stats = self.session_stats
        total_frames = len(self.combined_metrics)
        
        if total_frames == 0:
            print("⚠️  No metrics to save")
            return None
        
        # Calculate averages
        cross_midline_pct = (stats['front_camera']['cross_midline_count'] / total_frames * 100)
        bilateral_avg = np.mean(stats['front_camera']['bilateral_coordination_scores']) * 100 if stats['front_camera']['bilateral_coordination_scores'] else 0
        shoulder_align_avg = np.mean(stats['front_camera']['shoulder_alignment_scores']) if stats['front_camera']['shoulder_alignment_scores'] else 0
        trunk_rotation_avg = np.mean(stats['front_camera']['trunk_rotation_angles']) if stats['front_camera']['trunk_rotation_angles'] else 0
        
        postural_lean_avg = np.mean(stats['side_camera']['postural_lean_angles']) if stats['side_camera']['postural_lean_angles'] else 0
        posture_good_pct = (stats['side_camera']['posture_good_count'] / stats['side_camera']['posture_total_count'] * 100) if stats['side_camera']['posture_total_count'] > 0 else 0
        
        overall_score = 100 - (shoulder_align_avg + abs(trunk_rotation_avg)/2 + postural_lean_avg/2)
        overall_score = max(0, min(100, overall_score))
        
        # Create filename
        filename = os.path.join(output_dir, f'unity_summary_{self.session_id}.txt')
        
        # Write to TXT file
        with open(filename, 'w') as f:
            f.write("=== REHABILITATION SESSION SUMMARY ===\n\n")
            f.write(f"SessionID: {self.session_id}\n")
            f.write(f"DateTime: {self.session_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"GlobalStartTime: {self.global_start_time:.6f}\n")
            f.write(f"Duration: {self.combined_metrics[-1]['timestamp']:.1f} seconds\n")
            f.write(f"TotalFrames: {total_frames}\n")
            f.write(f"TriangulationUsed: {'Yes' if self.use_triangulation else 'No'}\n\n")
            
            f.write("--- FRONT CAMERA METRICS ---\n")
            f.write(f"CrossMidlinePercentage: {cross_midline_pct:.1f}%\n")
            f.write(f"BilateralCoordination: {bilateral_avg:.1f}%\n")
            f.write(f"ShoulderAlignment: {shoulder_align_avg:.2f}%\n")
            f.write(f"TrunkRotation: {trunk_rotation_avg:.1f} degrees\n\n")
            
            f.write("--- SIDE CAMERA METRICS ---\n")
            f.write(f"PosturalLean: {postural_lean_avg:.1f} degrees\n")
            f.write(f"GoodPosturePercentage: {posture_good_pct:.1f}%\n\n")
            
            f.write("--- OVERALL ASSESSMENT ---\n")
            f.write(f"OverallScore: {overall_score:.1f}%\n")
            
            if overall_score >= 80:
                f.write("Assessment: Excellent\n")
            elif overall_score >= 60:
                f.write("Assessment: Good\n")
            elif overall_score >= 40:
                f.write("Assessment: Fair\n")
            else:
                f.write("Assessment: NeedsImprovement\n")
        
        print(f"✅ Unity summary saved to: {filename}")
        return filename
    
    def save_detailed_csv(self, output_dir='.'):
        """Save detailed frame-by-frame metrics to CSV"""
        if not self.combined_metrics:
            print("⚠️  No metrics to save")
            return None
        
        flat_metrics = []
        
        for metric in self.combined_metrics:
            flat = {
                'timestamp': metric.get('timestamp', 0),
                'global_timestamp': metric.get('global_time', 0)
            }
            
            # Front camera metrics
            if 'cross_midline' in metric.get('front_camera', {}):
                cm = metric['front_camera']['cross_midline']
                flat['front_cross_midline_active'] = cm.get('any_crossing', False)
                flat['front_cross_midline_left'] = cm.get('left_crosses', False)
                flat['front_cross_midline_right'] = cm.get('right_crosses', False)
            
            if 'bilateral_coordination' in metric.get('front_camera', {}):
                bc = metric['front_camera']['bilateral_coordination']
                flat['front_bilateral_coordinated'] = bc.get('coordinated', False)
                flat['front_bilateral_height_similarity'] = bc.get('height_similarity', 0)
            
            if 'shoulder_alignment' in metric.get('front_camera', {}):
                sa = metric['front_camera']['shoulder_alignment']
                flat['front_shoulder_height_diff'] = sa.get('height_difference', 0)
            
            if 'trunk_rotation_frontal' in metric.get('front_camera', {}):
                flat['front_trunk_rotation'] = metric['front_camera']['trunk_rotation_frontal']
            
            # Side camera metrics
            if 'postural_lean' in metric.get('side_camera', {}):
                flat['side_postural_lean'] = metric['side_camera']['postural_lean']
            
            if 'shoulder_elevation_sagittal' in metric.get('side_camera', {}):
                se = metric['side_camera']['shoulder_elevation_sagittal']
                flat['side_left_shoulder_angle'] = se.get('left_shoulder_angle', 0)
                flat['side_right_shoulder_angle'] = se.get('right_shoulder_angle', 0)
            
            # Fusion metrics
            if 'fusion' in metric:
                if 'overall_posture_score' in metric['fusion']:
                    flat['fusion_posture_score'] = metric['fusion']['overall_posture_score']
                    flat['fusion_posture_assessment'] = metric['fusion']['posture_assessment']
            
            flat_metrics.append(flat)
        
        df = pd.DataFrame(flat_metrics)
        self.trial_number += 1
        filename = os.path.join(output_dir, f'detailed_metrics_{self.session_id}_{self.trial_number}.csv')
        df.to_csv(filename, index=False)
        print(f"📊 Detailed metrics saved to: {filename}")
        return filename


def main():
    print("="*70)
    print("=== DUAL-CAMERA Rehabilitation Tracker ===")
    print("="*70)
    print("\n🎯 Features:")
    print("  ✓ Preview mode before recording")
    print("  ✓ Triangulation toggle (if calibrated)")
    print("  ✓ TXT output for Unity (with % and units)")
    print("  ✓ CSV output for detailed analysis")
    print("  ✓ Global timestamp synchronization")
    print("  ✓ Enhanced side-view visualization")
    print("\n⌨️  Controls:")
    print("  'Q' - Quit")
    print("  'S' - Start/Stop Recording")
    print("  'T' - Toggle Triangulation (preview only)")
    print()
    
    # Check for triangulation
    use_triangulation = False
    if TRIANGULATION_AVAILABLE and os.path.exists("stereo_calibration.json"):
        response = input("Stereo calibration found. Enable triangulation? (y/n) [n]: ").strip().lower()
        use_triangulation = response == 'y'
    
    tracker = DualCameraRehabTracker()
    if use_triangulation:
        tracker.use_triangulation = True
    
    # ===== Camera selection =====
    # Use the indices you found with check_cams.py, e.g. 0 and 1


    BACKEND = cv2.CAP_AVFOUNDATION  # macOS

    def open_cam(idx, w=2560, h=2560, fps=30, warmup=8):
        cap = cv2.VideoCapture(idx, BACKEND)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {idx} with AVFoundation")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        cap.set(cv2.CAP_PROP_CONVERT_RGB,  1)

        # tiny warmup so Continuity Camera stabilizes
        for _ in range(warmup):
            cap.read()
            time.sleep(0.01)
        return cap

# FaceTime = 0, iPhone (Continuity Camera) = 1
    front_idx = 0
    side_idx = 1
    cap1 = open_cam(front_idx)  # iPhone
    cap2  = open_cam(side_idx)  # FaceTime

    ok1, f1 = cap1.read()
    ok2, f2 = cap2.read()
    print("front:", ok1, (None if not ok1 else f1.shape))
    print("side :", ok2, (None if not ok2 else f2.shape))
    
    # Initialize timestamps
    tracker.start_time = time.time()
    tracker.tracker_front.start_time = tracker.start_time
    tracker.tracker_side.start_time = tracker.start_time
    
    # Try to initialize stereo if enabled
    if tracker.use_triangulation:
        try:
            ret_init1, f_init1 = cap1.read()
            ret_init2, f_init2 = cap2.read()
            if ret_init1 and ret_init2:
                h, w = f_init1.shape[:2]
                if (f_init2.shape[0], f_init2.shape[1]) != (h, w):
                    f_init2 = cv2.resize(f_init2, (w, h))
                
                tracker.maps1, tracker.maps2, tracker.P1, tracker.P2, tracker.Q, (tracker.K1, tracker.D1, tracker.K2, tracker.D2) = init_stereo_from_calib(
                    "stereo_calibration.json", (w, h)
                )
                tracker.triangulation_initialized = True
                print("✅ Stereo rectification initialized")
        except Exception as e:
            print(f"⚠️  Stereo init failed: {e}")
            tracker.use_triangulation = False
    
    print("\n🎥 PREVIEW MODE - Position yourself and test settings")
    print("   Press 'S' to start recording when ready\n")
    
    frame_count = 0
    global mp_signal_start, mp_signal_stop, mp_start_time
    try:
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or frame1 is None or not ret2 or frame2 is None:
                break

            with mp_lock:
                    if mp_signal_start:
                        mp_signal_start = False
                        if not tracker.is_recording:
                            tracker.start_recording()
                            # optional: sync timestamps to Unity "START" moment
                            tracker.start_time = mp_start_time
                            tracker.global_start_time = mp_start_time
                            tracker.tracker_front.start_time = mp_start_time
                            tracker.tracker_side.start_time  = mp_start_time
                            print("▶️ Recording started by Unity (S).")

                    if mp_signal_stop:
                        mp_signal_stop = False
                        if tracker.is_recording:
                            tracker.stop_recording()
                        os.makedirs(LOG_DIR, exist_ok=True)
                        tracker.save_detailed_csv(output_dir=LOG_DIR)
                        print("💾 detailed_metrics saved (Unity STOP).")
            
            # Ensure same size
            if frame1.shape[:2] != frame2.shape[:2]:
                frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
            
            # Process based on triangulation mode
            if tracker.use_triangulation and tracker.triangulation_initialized:
                try:
                    tri_out = process_frame_pair_and_triangulate(
                        frame_front=frame1, frame_side=frame2,
                        front_tracker=tracker.tracker_front,
                        side_tracker=tracker.tracker_side,
                        maps1=tracker.maps1, maps2=tracker.maps2,
                        P1=tracker.P1, P2=tracker.P2,
                        landmark_indices=None, undistort=False)
                    
                    results_front = tri_out.get("results_front")
                    results_side = tri_out.get("results_side")
                    
                    processed_frame1, metrics1, _ = tracker.process_front_camera(frame1.copy(), results_front)
                    processed_frame2, metrics2, _ = tracker.process_side_camera(frame2.copy(), results_side)
                    
                    pts3d = tri_out.get("pts3d", None)
                    if pts3d is not None and pts3d.size > 0:
                        smoothed_pts3d = tracker.smoother_3d.smooth(pts3d)
                
                except Exception as e:
                    print(f"Triangulation error: {e}")
                    processed_frame1, metrics1, _ = tracker.process_front_camera(frame1.copy())
                    processed_frame2, metrics2, _ = tracker.process_side_camera(frame2.copy())
            else:
                processed_frame1, metrics1, _ = tracker.process_front_camera(frame1.copy())
                processed_frame2, metrics2, _ = tracker.process_side_camera(frame2.copy())
            
            # Store metrics only if recording
            if tracker.is_recording:
                front_metrics = tracker.extract_front_camera_metrics(metrics1)
                side_metrics = tracker.extract_side_camera_metrics(metrics2)
                timestamp = time.time() - tracker.start_time
                combined = tracker.combine_metrics(front_metrics, side_metrics, timestamp)
                tracker.combined_metrics.append(combined)
                frame_count += 1
                
            
            # Create display
            height = max(processed_frame1.shape[0], processed_frame2.shape[0])
            
            scale1 = height / processed_frame1.shape[0]
            width1 = int(processed_frame1.shape[1] * scale1)
            resized1 = cv2.resize(processed_frame1, (width1, height))
            
            scale2 = height / processed_frame2.shape[0]
            width2 = int(processed_frame2.shape[1] * scale2)
            resized2 = cv2.resize(processed_frame2, (width2, height))
            
            # Labels
            cv2.putText(resized1, f"FRONT CAM {front_idx}",
                       (20, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            cv2.putText(resized2, f"SIDE CAM {side_idx}",
                       (20, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 3)
            
            # Combine
            separator = np.ones((height, 5, 3), dtype=np.uint8) * 255
            split_screen = np.hstack([resized1, separator, resized2])
            split_screen = cv2.resize(split_screen, (1920, 1080))

            
            # Status bar
            mode_text = "🔴 RECORDING" if tracker.is_recording else "⚪ PREVIEW"
            mode_color = (0, 0, 255) if tracker.is_recording else (255, 255, 255)
            
            triangulation_status = "3D-ON" if tracker.use_triangulation else "2D"
            
            if tracker.is_recording:
                elapsed_time = time.time() - tracker.start_time
                status = f"{mode_text} | Time: {elapsed_time:.1f}s | Frames: {frame_count} | {triangulation_status}"
            else:
                status = f"{mode_text} | R=record T=toggle-3D S=save Q=quit | {triangulation_status}"
            
            cv2.putText(split_screen, status,
                       (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
            
            cv2.imshow('Dual-Camera Rehabilitation Tracking', split_screen)
            
            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n⏹️  Stopping...")
                break
            elif key == ord('s') or key == ord('S'):
                if not tracker.is_recording:
                    tracker.start_recording()
                else:
                    tracker.stop_recording()
            elif key == ord('t') or key == ord('T'):
                tracker.toggle_triangulation()
            elif key == ord('r') or key == ord('R'):
                if not tracker.is_recording and len(tracker.combined_metrics) > 0:
                    print("\n💾 Saving metrics...")
                    tracker.save_unity_summary_txt()
                    tracker.save_detailed_csv()
                    print("✅ Metrics saved successfully!")
                elif tracker.is_recording:
                    print("⚠️  Stop recording first (press 'R')")
                else:
                    print("⚠️  No data to save")
    
    except KeyboardInterrupt:
        print("\n  Recording interrupted")
    
    finally:
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
    
    # Final save if data exists
    print("\n" + "="*70)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if tracker.combined_metrics:
        print(f"✅ Session complete! Collected {len(tracker.combined_metrics)} data points")
        if not tracker.is_recording:
            print("\n💾 Auto-saving session data...")
            tracker.save_unity_summary_txt()
            tracker.save_detailed_csv()
            print("✅ All data saved!")
            print(f"\n📁 Files created:")
            print(f"   - unity_summary_{tracker.session_id}_{timestamp}.txt")
            print(f"   - detailed_metrics_{tracker.session_id}_{timestamp}.csv")
    else:
        print("ℹ️  No data recorded in this session")
    print("="*70)


if __name__ == "__main__":
    main()