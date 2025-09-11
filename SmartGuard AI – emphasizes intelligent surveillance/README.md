# AI-Powered-Surveillance-System


An intelligent video surveillance system that automatically detects suspicious activities using deep learning and computer vision techniques. Designed to improve public safety in environments such as parking lots, banks, and campuses, this system leverages anomaly detection to monitor CCTV footage in real-time without requiring human intervention.

📌 Features

Loitering Detection – Identifies individuals staying too long in restricted areas.

Unusual Movement Patterns – Detects erratic or abnormal movements using optical flow analysis.

Object Abandonment – Recognizes unattended objects through background subtraction.

Crowd Anomalies – Identifies unusual crowd formations and behaviors.

Real-Time Alerts – Processes at 15–20 FPS with instant notifications.

Explainable AI – Provides visual cues on why an event was flagged as anomalous.

Synthetic Data Generation – Uses GANs to create rare-event data for improved training.

Interactive Dashboard – Streamlit-powered UI for live monitoring and visualization.

🏗️ System Architecture

The system follows a multi-stage pipeline:

Data Preparation – Preprocessing of the Avenue Dataset with spatial and temporal augmentations.

Object Detection & Tracking – YOLOv8 with Hungarian + Kalman filtering.

Anomaly Detection – Isolation Forest + LSTM for sequence modeling.

Synthetic Data Generation – GAN-based augmentation for rare anomalies.

System Integration – Streamlit dashboard + SQLite database for alerts and logs.

⚙️ Tech Stack

Languages & Frameworks

Python 3.8+

OpenCV 4.7.0

PyTorch 2.0.0

Ultralytics YOLOv8

Streamlit 1.22.0

Scikit-learn 1.2.2

Models & Algorithms

YOLOv8n (real-time object detection)

Isolation Forest (unsupervised anomaly detection)

LSTM Networks (behavioral sequence modeling)

Custom GAN (synthetic anomaly generation)

Optical Flow (Farneback method for motion estimation)

Other Libraries

SQLite3 (database for alerts)

NumPy, Pandas, Matplotlib, Plotly


🚀 Installation

Clone the repository:

git clone https://github.com/your-username/AI-Powered-Surveillance-System.git
cd AI-Powered-Surveillance-System


Install dependencies:

pip install -r requirements.txt


Download and prepare the Avenue Dataset


Run the main system:

python surveillance_main_system.py


Launch the dashboard:

streamlit run surveillance_dashboard.py

📊 Performance

Detection Accuracy (mAP): 89.2% (Avenue Dataset)

Processing Speed: 15–20 FPS on NVIDIA RTX 3060

False Positive Rate: <8%

🎥 Demo

📌 Demo Video
https://drive.google.com/file/d/15YT9GmwSb2yBl7YUQcdt3mz4Zg1ZBldw/view?usp=drive_link



📖 References

Avenue Dataset https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html

YOLOv8 Documentation

OpenCV Docs

PyTorch Docs

Streamlit Docs

👩‍💻 Author

Gauri Pandey
AI & Machine Learning Enthusiast 
