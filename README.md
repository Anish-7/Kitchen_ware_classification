Kitchenware Image Classification Project
Overview
This project focuses on building a high-performance image classification model capable of distinguishing between four types of kitchenware: cups, knives, scissors, and spoons. We have used various machine learning and deep learning techniques, including CNNs, MobileNetV2, and HOG-SVM classifiers, and incorporated reinforcement learning to enhance the model's performance using real-time feedback.

webapp :
https://kwc-anish-13384109738.us-central1.run.app

Project Structure
Chapter 1: Introduction
Overview of the project’s motivation, problem statement, and objectives.

Chapter 2: Literature Review
A detailed review of key image classification models such as CNNs, MobileNetV2, and traditional methods like HOG + SVM.

Chapter 3: Preprocessing and Model Selection

3.1 Data Collection: Process of collecting and augmenting kitchenware images.
3.2 Model Architectures: Explanation of CNN, HOG-SVM, and MobileNetV2 models.
3.3 Discussion of Results: Analysis of model performance.
3.4 Integration of Deep Reinforcement Learning: Adding RL feedback mechanisms for dynamic learning.
Chapter 4: Deployment

4.1 Flask Web App Design: Web interface for classification.
4.2 Docker Containerization: Containerized deployment for consistent environments.
4.3 Google Cloud Run: Cloud deployment for scalability.
Chapter 5: Conclusion
Summary of findings, contributions, and future work.

Installation

Navigate to the project directory:

cd kitchenware-classification

Install dependencies:

pip install -r requirements.txt

Run the Flask web application:

python app.py

Technologies Used
Deep Learning Framework: TensorFlow/Keras
Web Framework: Flask
Deployment: Docker, Google Cloud Run
Image Processing: OpenCV, HOG features


Acknowledgments
I would like to Thank you Prof. Dr. S. Twieg for helping me out in this project.
