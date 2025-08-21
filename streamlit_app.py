import streamlit as st
import tempfile
import os
import json
from cover_drive_analysis import CricketAnalyzer
import cv2
import base64
from pathlib import Path

st.set_page_config(
    page_title="AthleteRise - Cricket Analytics", 
    page_icon="🏏",
    layout="wide"
)

def get_base64_of_file(path):
    """Convert file to base64 for download"""
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def create_download_link(file_path, link_text):
    """Create download link for file"""
    b64 = get_base64_of_file(file_path)
    file_name = os.path.basename(file_path)
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">{link_text}</a>'
    return href

def main():
    st.title("🏏 AthleteRise - AI Cricket Analytics")
    st.markdown("### Real-Time Cover Drive Analysis")
    
    # Sidebar
    st.sidebar.header("Configuration")
    use_custom_config = st.sidebar.checkbox("Use Custom Configuration")
    
    if use_custom_config:
        st.sidebar.subheader("Angle Thresholds")
        elbow_min = st.sidebar.slider("Elbow Angle Min", 80, 120, 100)
        elbow_max = st.sidebar.slider("Elbow Angle Max", 120, 160, 140)
        spine_lean_max = st.sidebar.slider("Max Spine Lean", 10, 40, 25)
        head_knee_threshold = st.sidebar.slider("Head-Knee Threshold", 20, 80, 50)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 Results", "🎯 About"])
    
    with tab1:
        st.header("Upload Video for Analysis")
        
        # Option 1: Upload file
        uploaded_file = st.file_uploader(
            "Choose a video file", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a cricket cover drive video for analysis"
        )
        
        # Option 2: YouTube URL
        st.subheader("Or analyze from YouTube")
        youtube_url = st.text_input(
            "YouTube URL", 
            placeholder="https://youtube.com/shorts/vSX3IRxGnNY",
            help="Paste a YouTube URL to analyze"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            analyze_button = st.button("🔍 Start Analysis", type="primary")
        
        with col2:
            if st.button("View Sample Report"):
                st.session_state.show_sample = True
        
        # Analysis process
        if analyze_button and (uploaded_file is not None or youtube_url.strip()):
            with st.spinner("Initializing analysis..."):
                try:
                    # Initialize analyzer with custom config if provided
                    config = None
                    if use_custom_config:
                        config = {
                            "angle_thresholds": {
                                "good_elbow_min": elbow_min,
                                "good_elbow_max": elbow_max,
                                "spine_lean_max": spine_lean_max,
                                "head_knee_threshold": head_knee_threshold
                            }
                        }
                        # Save temporary config
                        with open("temp_config.json", "w") as f:
                            json.dump(config, f)
                        analyzer = CricketAnalyzer("temp_config.json")
                    else:
                        analyzer = CricketAnalyzer()
                    
                    # Determine video source
                    if uploaded_file is not None:
                        # Save uploaded file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            video_path = tmp_file.name
                    else:
                        # Download from YouTube
                        st.info("Downloading video from YouTube...")
                        video_path = analyzer.download_video(youtube_url, "temp_video.mp4")
                    
                    # Show video info
                    cap = cv2.VideoCapture(video_path)
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps
                    cap.release()
                    
                    st.success(f"Video loaded: {duration:.1f}s duration, {fps} FPS")
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Run analysis
                    status_text.text("Analyzing video... This may take a few minutes.")
                    results = analyzer.analyze_video(video_path)
                    
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    
                    # Store results in session state
                    st.session_state.analysis_results = results
                    st.session_state.video_analyzed = True
                    
                    # Clean up temporary files
                    if uploaded_file is not None:
                        os.unlink(video_path)
                    
                    st.success(" Analysis completed successfully!")
                    st.info("Check the 'Results' tab to view your analysis.")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.exception(e)
    
    with tab2:
        st.header("📊 Analysis Results")
        
        if hasattr(st.session_state, 'analysis_results') and st.session_state.video_analyzed:
            results = st.session_state.analysis_results
            evaluation = results["evaluation"]
            
            # Overall Score Display
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.metric(
                    "Overall Score", 
                    f"{evaluation['overall_score']:.1f}/10",
                    delta=f"Grade: {evaluation['skill_grade']}"
                )
            
            with col2:
                processing_fps = results["video_info"]["processing_fps"]
                st.metric("Processing FPS", f"{processing_fps:.1f}")
            
            with col3:
                total_frames = results["metrics_summary"]["total_frames_analyzed"]
                st.metric("Frames Analyzed", total_frames)
            
            # Category Scores
            st.subheader("Category Breakdown")
            
            categories = ['footwork', 'head_position', 'swing_control', 'balance', 'follow_through']
            category_names = ['Footwork', 'Head Position', 'Swing Control', 'Balance', 'Follow-through']
            
            cols = st.columns(len(categories))
            for i, (category, name) in enumerate(zip(categories, category_names)):
                with cols[i]:
                    score = evaluation[category]
                    color = "normal"
                    if score >= 8:
                        color = "normal"
                    elif score >= 6:
                        color = "off" 
                    else:
                        color = "inverse"
                    
                    st.metric(name, f"{score:.1f}/10")
            
            # Feedback Section
            st.subheader("💡 Actionable Feedback")
            
            for category, feedback in evaluation['feedback'].items():
                with st.expander(f"{category.replace('_', ' ').title()} Feedback"):
                    st.write(feedback)
            
            # Phase Detection Results
            if "phases_detected" in results["metrics_summary"]:
                st.subheader("🎯 Shot Phases Detected")
                phases = results["metrics_summary"]["phases_detected"]
                if phases:
                    phase_cols = st.columns(len(phases))
                    for i, phase in enumerate(phases):
                        with phase_cols[i]:
                            st.info(phase)
                else:
                    st.warning("No distinct phases detected in the video")
            
            # Video Downloads
            st.subheader("📥 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if os.path.exists("output/annotated_video.mp4"):
                    st.markdown("**📹 Annotated Video**")
                    st.markdown(create_download_link("output/annotated_video.mp4", "Download Annotated Video"), unsafe_allow_html=True)
            
            with col2:
                if os.path.exists("output/evaluation.json"):
                    st.markdown("**📋 Detailed Report (JSON)**")
                    st.markdown(create_download_link("output/evaluation.json", "Download JSON Report"), unsafe_allow_html=True)
            
            with col3:
                if os.path.exists("output/evaluation.txt"):
                    st.markdown("**📄 Summary Report (TXT)**")
                    st.markdown(create_download_link("output/evaluation.txt", "Download Text Report"), unsafe_allow_html=True)
            
            # Temporal Analysis Chart
            if os.path.exists("output/temporal_analysis.png"):
                st.subheader("📈 Temporal Analysis")
                st.image("output/temporal_analysis.png", caption="Angle progression over time")
                st.markdown(create_download_link("output/temporal_analysis.png", "Download Chart"), unsafe_allow_html=True)
            
            # Raw Data Viewer
            with st.expander("🔍 View Raw Analysis Data"):
                st.json(results)
                
        else:
            st.info("No analysis results available. Please analyze a video first.")
            
            # Show sample report if requested
            if hasattr(st.session_state, 'show_sample') and st.session_state.show_sample:
                st.subheader("📋 Sample Analysis Report")
                
                sample_data = {
                    "overall_score": 7.2,
                    "skill_grade": "Intermediate",
                    "footwork": 6.8,
                    "head_position": 8.1,
                    "swing_control": 7.0,
                    "balance": 7.5,
                    "follow_through": 6.6,
                    "feedback": {
                        "footwork": "Good foot positioning. Work on maintaining balance through the shot.",
                        "head_position": "Excellent head stability. Keep eyes on the ball contact point.",
                        "swing_control": "Good elbow positioning. Focus on smoother follow-through.",
                        "balance": "Good balance maintained. Minor adjustment needed in spine angle.",
                        "follow_through": "Complete the follow-through with higher bat finish."
                    }
                }
                
                st.metric("Sample Overall Score", f"{sample_data['overall_score']}/10", 
                         delta=f"Grade: {sample_data['skill_grade']}")
                
                for category, feedback in sample_data['feedback'].items():
                    with st.expander(f"Sample {category.replace('_', ' ').title()} Feedback"):
                        st.write(f"Score: {sample_data[category]:.1f}/10")
                        st.write(feedback)
    
    with tab3:
        st.header(" About AthleteRise Cricket Analytics")
        
        st.markdown("""
        ### What I Analyzed
        
        Performed comprehensive biomechanical analysis of cricket cover drives:
        
        **Pose Estimation**
        - Real-time joint tracking using MediaPipe
        - 13+ key body landmarks analyzed per frame
        - Handles occlusions and missing data
        
        **Biomechanical Metrics**
        - **Front Elbow Angle**: Optimal range 110-130°
        - **Spine Lean**: Balance assessment, target <20°
        - **Head-Knee Alignment**: Balance and timing indicator
        - **Front Foot Direction**: Footwork quality measure
        
        **Shot Phases**
        - Stance → Stride → Downswing → Impact → Follow-through → Recovery
        - Automatic phase detection using joint velocities
        - Timing and flow analysis
        
        ** Scoring Categories**
        1. **Footwork** (1-10): Foot positioning and movement
        2. **Head Position** (1-10): Balance and eye alignment
        3. **Swing Control** (1-10): Bat path and tempo
        4. **Balance** (1-10): Body stability throughout shot
        5. **Follow-through** (1-10): Shot completion and finish
        
        ### Key Features
        - Real-time video processing (10+ FPS on CPU)
        - Live overlay annotations
        - Comprehensive scoring system
        - Actionable feedback generation
        - Temporal analysis charts
        - Multiple export formats
        - YouTube video support
        - Customizable thresholds
        
        ### Technical Specifications
        - **Input**: MP4, AVI, MOV, MKV videos or YouTube URLs
        - **Processing**: MediaPipe Pose estimation
        - **Output**: Annotated video + JSON/TXT reports + Charts
        - **Performance**: Optimized for real-time analysis
        - **Accuracy**: Biomechanics standards
        
        ---
        
        **Built with:** Python, OpenCV, MediaPipe, Streamlit
        """)

if __name__ == "__main__":
    main()