import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import yt_dlp
import matplotlib.pyplot as plt
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CricketPhase(Enum):
    STANCE = "Stance"
    STRIDE = "Stride"  
    DOWNSWING = "Downswing"
    IMPACT = "Impact"
    FOLLOW_THROUGH = "Follow-through"
    RECOVERY = "Recovery"

@dataclass
class BiomechanicalMetrics:
    frame_number: int
    timestamp: float
    front_elbow_angle: Optional[float] = None
    spine_lean: Optional[float] = None
    head_knee_alignment: Optional[float] = None
    front_foot_angle: Optional[float] = None
    wrist_velocity: Optional[float] = None
    phase: Optional[CricketPhase] = None

@dataclass
class ShotEvaluation:
    footwork: float
    head_position: float
    swing_control: float
    balance: float
    follow_through: float
    overall_score: float
    feedback: Dict[str, str]
    skill_grade: str

class CricketAnalyzer:
    def __init__(self, config_path: Optional[str] = None):
        # Init MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Balanced speed vs accuracy
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize tracking variables
        self.metrics_history: List[BiomechanicalMetrics] = []
        self.fps_counter = []
        self.current_phase = CricketPhase.STANCE
        self.previous_keypoints = None
        
        # Create output directory
        os.makedirs('output', exist_ok=True)
        
    def load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration with defaults"""
        default_config = {
            "angle_thresholds": {
                "good_elbow_min": 100,
                "good_elbow_max": 140,
                "spine_lean_max": 25,
                "head_knee_threshold": 50
            },
            "ideal_ranges": {
                "elbow_angle": (110, 130),
                "spine_lean": (10, 20),
                "front_foot_angle": (30, 60)
            },
            "phase_detection": {
                "velocity_threshold": 5.0,
                "acceleration_threshold": 10.0
            },
            "output": {
                "target_fps": 30,
                "overlay_alpha": 0.7
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config

    def download_video(self, url: str, output_path: str = "input_video.mp4") -> str:
        """Download video from YouTube"""
        try:
            ydl_opts = {
                'format': 'mp4[height<=720]/best[height<=720]/best',
                'outtmpl': output_path,
                'quiet': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Downloading video from {url}")
                ydl.download([url])
                
            return output_path
        except Exception as e:
            logger.error(f"Failed to download video: {e}")
            raise

    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points"""
        try:
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            return angle
        except:
            return None

    def extract_keypoints(self, results) -> Optional[Dict[str, np.ndarray]]:
        """Extract keypoints from MediaPipe results"""
        if not results.pose_landmarks:
            return None
            
        landmarks = results.pose_landmarks.landmark
        h, w = self.frame_height, self.frame_width
        
        keypoints = {}
        landmark_map = {
            'nose': self.mp_pose.PoseLandmark.NOSE,
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE
        }
        
        for name, landmark_idx in landmark_map.items():
            landmark = landmarks[landmark_idx]
            if landmark.visibility > 0.5:  # Only use visible landmarks
                keypoints[name] = np.array([landmark.x * w, landmark.y * h])
            else:
                keypoints[name] = None
                
        return keypoints

    def calculate_biomechanical_metrics(self, keypoints: Dict[str, np.ndarray], 
                                      frame_num: int, timestamp: float) -> BiomechanicalMetrics:
        """Calculate biomechanical metrics from keypoints"""
        metrics = BiomechanicalMetrics(frame_num, timestamp)
        
        try:
            # Determine batting stance (left/right handed)
            front_side = 'left'  # Left side is front for right-handed batsman
            back_side = 'right'
            
            # Front elbow angle
            if all(k is not None for k in [
                keypoints.get(f'{front_side}_shoulder'),
                keypoints.get(f'{front_side}_elbow'),
                keypoints.get(f'{front_side}_wrist')
            ]):
                metrics.front_elbow_angle = self.calculate_angle(
                    keypoints[f'{front_side}_shoulder'],
                    keypoints[f'{front_side}_elbow'],
                    keypoints[f'{front_side}_wrist']
                )
            
            # Spine lean (shoulder line vs vertical)
            if keypoints.get('left_shoulder') is not None and keypoints.get('right_shoulder') is not None:
                shoulder_line = keypoints['right_shoulder'] - keypoints['left_shoulder']
                vertical = np.array([0, 1])
                metrics.spine_lean = np.degrees(np.arctan2(shoulder_line[0], shoulder_line[1]))
            
            # Head-knee alignment
            if keypoints.get('nose') is not None and keypoints.get(f'{front_side}_knee') is not None:
                head_pos = keypoints['nose']
                knee_pos = keypoints[f'{front_side}_knee']
                metrics.head_knee_alignment = abs(head_pos[0] - knee_pos[0])
            
            # Front foot angle (approximate using ankle-knee vector)
            if keypoints.get(f'{front_side}_ankle') is not None and keypoints.get(f'{front_side}_knee') is not None:
                foot_vector = keypoints[f'{front_side}_knee'] - keypoints[f'{front_side}_ankle']
                horizontal = np.array([1, 0])
                metrics.front_foot_angle = np.degrees(np.arccos(
                    np.dot(foot_vector, horizontal) / np.linalg.norm(foot_vector)
                ))
            
            # Wrist velocity (if previous frame exists)
            if (self.previous_keypoints and 
                keypoints.get(f'{front_side}_wrist') is not None and 
                self.previous_keypoints.get(f'{front_side}_wrist') is not None):
                
                current_wrist = keypoints[f'{front_side}_wrist']
                prev_wrist = self.previous_keypoints[f'{front_side}_wrist']
                velocity = np.linalg.norm(current_wrist - prev_wrist)
                metrics.wrist_velocity = velocity
            
            # Phase detection based on wrist velocity and position
            metrics.phase = self.detect_phase(keypoints, metrics)
            
        except Exception as e:
            logger.warning(f"Error calculating metrics for frame {frame_num}: {e}")
            
        return metrics

    def detect_phase(self, keypoints: Dict[str, np.ndarray], 
                    metrics: BiomechanicalMetrics) -> CricketPhase:
        """Detect cricket shot phase using heuristics"""
        # Simple phase detection based on wrist velocity and position
        if metrics.wrist_velocity is None:
            return CricketPhase.STANCE
            
        velocity_threshold = self.config["phase_detection"]["velocity_threshold"]
        
        if metrics.wrist_velocity < velocity_threshold:
            return CricketPhase.STANCE
        elif metrics.wrist_velocity > velocity_threshold * 2:
            # Check if wrist is moving downward (downswing) or upward (follow-through)
            front_wrist = keypoints.get('left_wrist')  # Assuming right-handed
            if front_wrist is not None and self.previous_keypoints:
                prev_wrist = self.previous_keypoints.get('left_wrist')
                if prev_wrist is not None:
                    if front_wrist[1] > prev_wrist[1]:  # Moving down
                        return CricketPhase.DOWNSWING
                    else:  # Moving up
                        return CricketPhase.FOLLOW_THROUGH
        
        return CricketPhase.STRIDE

    def draw_overlays(self, frame: np.ndarray, results, metrics: BiomechanicalMetrics) -> np.ndarray:
        """Draw pose skeleton and metrics overlay on frame"""
        overlay_frame = frame.copy()
        
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                overlay_frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        
        # Draw metrics overlay
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Phase indicator
        cv2.putText(overlay_frame, f"Phase: {metrics.phase.value if metrics.phase else 'Unknown'}", 
                   (10, y_offset), font, font_scale, (255, 255, 0), thickness)
        y_offset += 30
        
        # Elbow angle with feedback
        if metrics.front_elbow_angle is not None:
            angle_text = f"Elbow: {metrics.front_elbow_angle:.1f}°"
            color = (0, 255, 0) if (self.config["angle_thresholds"]["good_elbow_min"] <= 
                                  metrics.front_elbow_angle <= 
                                  self.config["angle_thresholds"]["good_elbow_max"]) else (0, 0, 255)
            
            cv2.putText(overlay_frame, angle_text, (10, y_offset), font, font_scale, color, thickness)
            
            # Feedback cue
            feedback = "✅ Good elbow elevation" if color == (0, 255, 0) else "❌ Adjust elbow angle"
            cv2.putText(overlay_frame, feedback, (200, y_offset), font, 0.5, color, 1)
            y_offset += 30
        
        # Spine lean
        if metrics.spine_lean is not None:
            lean_text = f"Spine Lean: {abs(metrics.spine_lean):.1f}°"
            color = (0, 255, 0) if abs(metrics.spine_lean) <= self.config["angle_thresholds"]["spine_lean_max"] else (0, 0, 255)
            cv2.putText(overlay_frame, lean_text, (10, y_offset), font, font_scale, color, thickness)
            y_offset += 30
        
        # Head-knee alignment
        if metrics.head_knee_alignment is not None:
            alignment_text = f"Head-Knee: {metrics.head_knee_alignment:.1f}px"
            color = (0, 255, 0) if metrics.head_knee_alignment <= self.config["angle_thresholds"]["head_knee_threshold"] else (0, 0, 255)
            cv2.putText(overlay_frame, alignment_text, (10, y_offset), font, font_scale, color, thickness)
            
            feedback = "✅ Good head position" if color == (0, 255, 0) else "❌ Head not over front knee"
            cv2.putText(overlay_frame, feedback, (200, y_offset), font, 0.5, color, 1)
            y_offset += 30
        
        # Wrist velocity
        if metrics.wrist_velocity is not None:
            velocity_text = f"Wrist Speed: {metrics.wrist_velocity:.1f}"
            cv2.putText(overlay_frame, velocity_text, (10, y_offset), font, font_scale, (255, 0, 255), thickness)
        
        return overlay_frame

    def evaluate_shot(self) -> ShotEvaluation:
        """Evaluate the complete shot and generate scores"""
        if not self.metrics_history:
            logger.warning("No metrics available for evaluation")
            return ShotEvaluation(0, 0, 0, 0, 0, 0, {}, "Beginner")
        
        # Extract valid metrics
        elbow_angles = [m.front_elbow_angle for m in self.metrics_history if m.front_elbow_angle is not None]
        spine_leans = [abs(m.spine_lean) for m in self.metrics_history if m.spine_lean is not None]
        head_alignments = [m.head_knee_alignment for m in self.metrics_history if m.head_knee_alignment is not None]
        wrist_velocities = [m.wrist_velocity for m in self.metrics_history if m.wrist_velocity is not None]
        
        # Score each category (1-10)
        footwork_score = self.score_footwork()
        head_position_score = self.score_head_position(head_alignments)
        swing_control_score = self.score_swing_control(elbow_angles, wrist_velocities)
        balance_score = self.score_balance(spine_leans)
        follow_through_score = self.score_follow_through()
        
        overall_score = np.mean([footwork_score, head_position_score, swing_control_score, 
                               balance_score, follow_through_score])
        
        # Generate feedback
        feedback = self.generate_feedback(footwork_score, head_position_score, 
                                        swing_control_score, balance_score, follow_through_score)
        
        # Determine skill grade
        skill_grade = self.determine_skill_grade(overall_score)
        
        return ShotEvaluation(
            footwork=footwork_score,
            head_position=head_position_score,
            swing_control=swing_control_score,
            balance=balance_score,
            follow_through=follow_through_score,
            overall_score=overall_score,
            feedback=feedback,
            skill_grade=skill_grade
        )

    def score_footwork(self) -> float:
        """Score footwork based on foot positioning and movement"""
        # Placeholder - would analyze foot angles and positioning
        foot_angles = [m.front_foot_angle for m in self.metrics_history if m.front_foot_angle is not None]
        if not foot_angles:
            return 5.0
        
        ideal_min, ideal_max = self.config["ideal_ranges"]["front_foot_angle"]
        good_angles = sum(1 for angle in foot_angles if ideal_min <= angle <= ideal_max)
        score = (good_angles / len(foot_angles)) * 10
        return min(10.0, max(1.0, score))

    def score_head_position(self, head_alignments: List[float]) -> float:
        """Score head position consistency"""
        if not head_alignments:
            return 5.0
        
        threshold = self.config["angle_thresholds"]["head_knee_threshold"]
        good_positions = sum(1 for alignment in head_alignments if alignment <= threshold)
        score = (good_positions / len(head_alignments)) * 10
        return min(10.0, max(1.0, score))

    def score_swing_control(self, elbow_angles: List[float], wrist_velocities: List[float]) -> float:
        """Score swing control based on elbow angles and velocity smoothness"""
        if not elbow_angles:
            return 5.0
        
        # Score elbow angle consistency
        ideal_min, ideal_max = self.config["ideal_ranges"]["elbow_angle"]
        good_angles = sum(1 for angle in elbow_angles if ideal_min <= angle <= ideal_max)
        angle_score = (good_angles / len(elbow_angles)) * 5
        
        # Score velocity smoothness
        velocity_score = 5.0
        if wrist_velocities and len(wrist_velocities) > 1:
            velocity_variance = np.var(wrist_velocities)
            # Lower variance = smoother swing = higher score
            velocity_score = max(1.0, 5.0 - (velocity_variance / 10))
        
        total_score = angle_score + velocity_score
        return min(10.0, max(1.0, total_score))

    def score_balance(self, spine_leans: List[float]) -> float:
        """Score balance based on spine lean consistency"""
        if not spine_leans:
            return 5.0
        
        ideal_max = self.config["ideal_ranges"]["spine_lean"][1]
        good_balance = sum(1 for lean in spine_leans if lean <= ideal_max)
        score = (good_balance / len(spine_leans)) * 10
        return min(10.0, max(1.0, score))

    def score_follow_through(self) -> float:
        """Score follow-through based on phase progression"""
        phases = [m.phase for m in self.metrics_history if m.phase is not None]
        if not phases:
            return 5.0
        
        # Check if follow-through phase is present
        has_follow_through = CricketPhase.FOLLOW_THROUGH in phases
        has_complete_sequence = len(set(phases)) >= 3  # At least 3 different phases
        
        score = 5.0
        if has_follow_through:
            score += 3.0
        if has_complete_sequence:
            score += 2.0
        
        return min(10.0, max(1.0, score))

    def generate_feedback(self, footwork: float, head_position: float, swing_control: float, 
                         balance: float, follow_through: float) -> Dict[str, str]:
        """Generate actionable feedback for each category"""
        feedback = {}
        
        if footwork < 6:
            feedback["footwork"] = "Work on front foot positioning. Aim to point towards the ball."
        else:
            feedback["footwork"] = "Good foot positioning. Maintain this stance consistency."
        
        if head_position < 6:
            feedback["head_position"] = "Keep your head more directly over the front knee for better balance."
        else:
            feedback["head_position"] = "Excellent head position. Good balance maintained."
        
        if swing_control < 6:
            feedback["swing_control"] = "Focus on elbow positioning and smoother swing tempo."
        else:
            feedback["swing_control"] = "Good swing control. Elbow positioning is consistent."
        
        if balance < 6:
            feedback["balance"] = "Reduce lateral spine lean. Stay more upright through the shot."
        else:
            feedback["balance"] = "Good balance maintained throughout the shot."
        
        if follow_through < 6:
            feedback["follow_through"] = "Ensure complete follow-through. Finish high with the bat."
        else:
            feedback["follow_through"] = "Good follow-through. Complete shot execution."
        
        return feedback

    def determine_skill_grade(self, overall_score: float) -> str:
        """Determine skill grade based on overall score"""
        if overall_score >= 8.0:
            return "Advanced"
        elif overall_score >= 6.0:
            return "Intermediate"
        else:
            return "Beginner"

    def create_temporal_analysis(self) -> str:
        """Create temporal analysis chart"""
        if len(self.metrics_history) < 2:
            return None
        
        timestamps = [m.timestamp for m in self.metrics_history]
        elbow_angles = [m.front_elbow_angle if m.front_elbow_angle else 0 for m in self.metrics_history]
        spine_leans = [abs(m.spine_lean) if m.spine_lean else 0 for m in self.metrics_history]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(timestamps, elbow_angles, 'b-', label='Elbow Angle')
        plt.axhline(y=self.config["ideal_ranges"]["elbow_angle"][0], color='g', linestyle='--', alpha=0.7, label='Ideal Range')
        plt.axhline(y=self.config["ideal_ranges"]["elbow_angle"][1], color='g', linestyle='--', alpha=0.7)
        plt.ylabel('Elbow Angle (degrees)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(timestamps, spine_leans, 'r-', label='Spine Lean')
        plt.axhline(y=self.config["ideal_ranges"]["spine_lean"][1], color='g', linestyle='--', alpha=0.7, label='Max Ideal')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Spine Lean (degrees)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = 'output/temporal_analysis.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return chart_path

    def analyze_video(self, video_path: str) -> Dict:
        """Main analysis function"""
        logger.info(f"Starting analysis of video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {self.frame_width}x{self.frame_height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = 'output/annotated_video.mp4'
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.frame_width, self.frame_height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            timestamp = frame_count / fps
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            # Extract keypoints and calculate metrics
            keypoints = self.extract_keypoints(results)
            metrics = self.calculate_biomechanical_metrics(keypoints, frame_count, timestamp)
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Draw overlays
            annotated_frame = self.draw_overlays(frame, results, metrics)
            
            # Write frame
            out.write(annotated_frame)
            
            # Update previous keypoints
            self.previous_keypoints = keypoints
            
            # FPS tracking
            frame_time = time.time() - frame_start
            if frame_time > 0:
                self.fps_counter.append(1.0 / frame_time)
            
            frame_count += 1
            
            # Progress logging
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                avg_fps = np.mean(self.fps_counter[-30:]) if self.fps_counter else 0
                logger.info(f"Progress: {progress:.1f}% - Processing FPS: {avg_fps:.1f}")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Calculate final metrics
        total_time = time.time() - start_time
        avg_fps = len(self.metrics_history) / total_time
        
        logger.info(f"Analysis complete. Average FPS: {avg_fps:.1f}")
        
        # Generate evaluation
        evaluation = self.evaluate_shot()
        
        # Create temporal analysis
        chart_path = self.create_temporal_analysis()
        
        # Save evaluation
        eval_data = {
            "timestamp": datetime.now().isoformat(),
            "video_info": {
                "path": video_path,
                "resolution": f"{self.frame_width}x{self.frame_height}",
                "fps": fps,
                "total_frames": total_frames,
                "processing_fps": avg_fps
            },
            "evaluation": asdict(evaluation),
            "metrics_summary": {
                "total_frames_analyzed": len(self.metrics_history),
                "valid_pose_detections": sum(1 for m in self.metrics_history if any([
                    m.front_elbow_angle, m.spine_lean, m.head_knee_alignment
                ])),
                "phases_detected": list(set(m.phase.value for m in self.metrics_history if m.phase))
            }
        }
        
        with open('output/evaluation.json', 'w') as f:
            json.dump(eval_data, f, indent=2)
        
        # Also save simplified text evaluation
        with open('output/evaluation.txt', 'w') as f:
            f.write("CRICKET COVER DRIVE ANALYSIS REPORT\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Overall Score: {evaluation.overall_score:.1f}/10\n")
            f.write(f"Skill Grade: {evaluation.skill_grade}\n\n")
            f.write("CATEGORY SCORES:\n")
            f.write(f"Footwork: {evaluation.footwork:.1f}/10\n")
            f.write(f"Head Position: {evaluation.head_position:.1f}/10\n")
            f.write(f"Swing Control: {evaluation.swing_control:.1f}/10\n")
            f.write(f"Balance: {evaluation.balance:.1f}/10\n")
            f.write(f"Follow-through: {evaluation.follow_through:.1f}/10\n\n")
            f.write("FEEDBACK:\n")
            for category, feedback in evaluation.feedback.items():
                f.write(f"{category.title()}: {feedback}\n")
        
        logger.info("Analysis saved to output/ directory")
        
        return eval_data

def main():
    """Main execution function"""
    # Configuration
    VIDEO_URL = "https://youtube.com/shorts/vSX3IRxGnNY"
    
    try:
        # Initialize analyzer
        analyzer = CricketAnalyzer()
        
        # Download video
        video_path = analyzer.download_video(VIDEO_URL, "input_video.mp4")
        
        # Analyze video
        results = analyzer.analyze_video(video_path)
        
        # Print summary
        evaluation = results["evaluation"]
        print("\n" + "="*50)
        print("CRICKET ANALYSIS COMPLETE")
        print("="*50)
        print(f"Overall Score: {evaluation['overall_score']:.1f}/10")
        print(f"Skill Grade: {evaluation['skill_grade']}")
        print(f"\nProcessing FPS: {results['video_info']['processing_fps']:.1f}")
        print("\nFiles generated:")
        print("- output/annotated_video.mp4")
        print("- output/evaluation.json")
        print("- output/evaluation.txt")
        if os.path.exists('output/temporal_analysis.png'):
            print("- output/temporal_analysis.png")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()