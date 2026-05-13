AI-Powered Multimodal Emotion Recognition System

A real-time multimodal emotion recognition system that combines facial expressions, speech, and text analysis to improve accuracy in complex and ambiguous scenarios.

Overview
Traditional emotion detection systems rely on a single modality such as face or text, which often leads to inaccurate predictions. This project addresses that limitation by integrating computer vision, natural language processing, and speech processing into a unified system for more robust emotion detection.

Features

Real-time facial emotion detection using CNN and Mediapipe
Speech-to-text conversion using Google Speech API
Text emotion classification using BERT (GoEmotions dataset)
Multimodal fusion engine for improved prediction accuracy
Interactive dashboard for emotion visualization using Streamlit
Automatic logging of results for analysis and evaluation

System Workflow
Input (Webcam + Microphone)
→ Face Emotion Detection (CNN)
→ Speech-to-Text → Text Emotion (BERT)
→ Fusion Engine
→ Result Logging (CSV)
→ Dashboard Visualization

Tech Stack
Python
PyTorch
OpenCV
Mediapipe
HuggingFace Transformers (BERT)
Streamlit
Google Speech API

Datasets Used
FER2013 – Facial emotion recognition (7 classes)
GoEmotions – Text-based emotion classification (28 labels)

Methodology

Preprocessed facial data using grayscale conversion, normalization, and resizing (48x48)
Tokenized text using BERT tokenizer with padding and truncation
Converted speech input to text before NLP processing
Implemented weighted fusion of facial and textual outputs
Used threading for real-time synchronization

Results
Face Emotion (CNN): ~72% accuracy
Text Emotion (BERT): ~84% accuracy
Multimodal Fusion: ~80% accuracy

Fusion improves prediction stability and reduces misclassification compared to single-modality systems.
