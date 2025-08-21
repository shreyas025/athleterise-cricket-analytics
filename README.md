# athleterise-cricket-analytics

## Real-Time Cover Drive Analysis System

A comprehensive Python-based system that processes cricket videos in real-time, performs biomechanical analysis using pose estimation, and provides detailed technical feedback with scoring.

## 🎯 Features

### Core Functionality
- **Real-time video processing** with OpenCV pipeline
- **AI pose estimation** using MediaPipe for 13+ key body landmarks
- **Biomechanical metrics calculation** including elbow angles, spine lean, head-knee alignment
- **Automatic shot phase detection** (Stance → Stride → Impact → Follow-through)
- **Live video overlays** with real-time feedback cues
- **Comprehensive scoring system** across 5 key categories
- **Multi-format exports** (MP4, JSON, TXT, PNG charts)

### Advanced Features (Bonus)
- ✅ **Automatic Phase Segmentation** - Detects cricket shot phases using joint velocities
- ✅ **Real-Time Performance** - Optimized for 10+ FPS processing on CPU
- ✅ **Skill Grade Prediction** - Beginner/Intermediate/Advanced classification
- ✅ **Streamlit Web App** - User-friendly interface for video upload and analysis
- ✅ **Robustness & Error Handling** - Graceful degradation and comprehensive logging
- ✅ **Modular Design** - Clean API for integration with other systems

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/shreyas025/athleterise-cricket-analytics.git
cd athleterise-cricket-analytics
```

2. **Create virtual environment**
```bash
python -m venv cricket_analytics_env
source cricket_analytics_env/bin/activate  # Linux/Mac
# or
cricket_analytics_env\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Command Line Analysis**
```bash
python cover_drive_analysis.py
```
This will analyze the default YouTube video and save results to the `output/` directory.

2. **Web Interface**
```bash
streamlit run streamlit_app.py
```

3. **Custom Video Analysis**
```python
from cover_drive_analysis import CricketAnalyzer

analyzer = CricketAnalyzer()
results = analyzer.analyze_video("path/to/your/video.mp4")
```

## 📊 Analysis Output

### Generated Files
```
output/
├── annotated_video.mp4      # Video with pose overlays and live metrics
├── evaluation.json          # Detailed analysis results and metrics
├── evaluation.txt          # Human-readable summary report
└── temporal_analysis.png   # Time-series charts of key angles
```

### Scoring Categories
- **Footwork** (1-10): Foot positioning and movement quality
- **Head Position** (1-10): Balance and visual alignment
- **Swing Control** (1-10): Bat path consistency and elbow positioning  
- **Balance** (1-10): Body stability throughout the shot
- **Follow-through** (1-10): Shot completion and finish quality

### Sample Evaluation Output
```json
{
  "overall_score": 7.2,
  "skill_grade": "Intermediate",
  "footwork": 6.8,
  "head_position": 8.1,
  "swing_control": 7.0,
  "balance": 7.5,
  "follow_through": 6.6,
  "feedback": {
    "footwork": "Good foot positioning. Work on maintaining balance through the shot.",
    "head_position": "Excellent head stability. Keep eyes on the ball contact point."
  }
}
```

## ⚙️ Configuration

Created a `config.json` file to customize analysis parameters:

```json
{
  "angle_thresholds": {
    "good_elbow_min": 100,
    "good_elbow_max": 140,
    "spine_lean_max": 25,
    "head_knee_threshold": 50
  },
  "ideal_ranges": {
    "elbow_angle": [110, 130],
    "spine_lean": [10, 20],
    "front_foot_angle": [30, 60]
  }
}
```

## 🧠 Technical Architecture

### Pose Estimation Pipeline
1. **Video Input** → OpenCV capture and frame processing
2. **MediaPipe Pose** → Real-time joint detection and tracking
3. **Keypoint Extraction** → 13 key body landmarks per frame
4. **Biomechanics Engine** → Angle calculations and metric computation
5. **Phase Detection** → Shot phase classification using velocity analysis
6. **Overlay Rendering** → Real-time visual feedback generation
7. **Evaluation Engine** → Multi-category scoring and feedback generation

### Performance Optimizations
- **Lightweight pose models** for real-time processing
- **Efficient memory management** with frame buffering
- **Vectorized calculations** using NumPy operations
- **Smart caching** for repeated computations
- **Graceful degradation** on missing detections

## 📈 Biomechanical Metrics

### Primary Measurements
- **Front Elbow Angle**: Shoulder-Elbow-Wrist angle (ideal: 110-130°)
- **Spine Lean**: Lateral tilt from vertical axis (target: <20°)
- **Head-Knee Alignment**: Horizontal distance between head and front knee
- **Front Foot Direction**: Angle relative to batting crease
- **Wrist Velocity**: Speed of bat movement through shot

### Phase Detection Logic
```
Stance     → Low wrist velocity, stable positions
Stride     → Moderate movement, foot positioning
Downswing  → High velocity, downward wrist motion
Impact     → Velocity peak, minimal position change
Follow-through → High velocity, upward motion
Recovery   → Decreasing velocity, return to stable position
```

## 🎮 Web Application Features

### User Interface
- **Drag-and-drop video upload** with format validation
- **YouTube URL processing** with automatic download
- **Real-time analysis progress** with FPS monitoring
- **Interactive results dashboard** with expandable sections
- **One-click file downloads** for all generated content
- **Customizable analysis parameters** via sidebar controls

### Analysis Visualization
- **Live metric overlays** on video frames
- **Category score breakdown** with visual indicators  
- **Temporal analysis charts** showing angle progression
- **Phase detection timeline** with color-coded segments
- **Actionable feedback panels** with specific recommendations


## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 4GB (8GB recommended)
- **OS**: Windows 10, macOS 10.14+, Ubuntu 18.04+

### Supported Video Formats
- MP4, AVI, MOV, MKV
- Resolution: 480p to 4K
- Frame rates: 15-60 FPS
- Codecs: H.264, H.265, VP9

### Dependencies
```
opencv-python    # Computer vision and video processing
mediapipe         # AI pose estimation
numpy             # Numerical computations
matplotlib         # Chart generation
yt-dlp        # YouTube video downloading
streamlit       # Web application framework
```
---
