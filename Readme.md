# Heart Attack Risk Prediction Using Retinal Eye Image


## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Problem Statement](#problem-statement)
3.  [Objectives](#objectives)
4.  [Key Features](#key-features)
5.  [System Architecture](#system-architecture)
6.  [Technologies Used](#technologies-used)
7.  [Dataset](#dataset)
8.  [Setup and Installation](#setup-and-installation)
9.  [Usage](#usage)
10. [Testing](#testing)
11. [Results](#results)
12. [Applications](#applications)
13. [Limitations](#limitations)
14. [Future Scope](#future-scope)
15. [Author](#author)
16. [Supervisor](#supervisor)
17. [Acknowledgements](#acknowledgements)
18. [License](#license)

---

## Project Overview

This project presents a system for predicting the risk of heart attack using retinal eye images, leveraging Machine Learning (ML) and Artificial Intelligence (AI) techniques. The core idea is that the retina, being a highly vascularized tissue, reflects subtle microvascular changes caused by conditions like hypertension and heart disease, often before clinical symptoms appear.

By analyzing morphological features of retinal blood vessels through advanced image processing and deep learning models (specifically Fuzzy C-Means clustering and Recurrent Neural Networks), this system aims to provide a non-invasive, accurate, and early detection method for cardiovascular risks. This approach facilitates timely interventions, personalized treatment planning, and strengthens preventive healthcare strategies, potentially reducing the global burden of cardiovascular diseases.

## Problem Statement

> To develop a model that predicts risk accurately of heart attack using non-invasive method with retinal images.

## Objectives

*   **Build an Accurate Health Risk Prediction Model:** Develop a reliable model using ML/AI (RNN, AdaBoost, Classification) to predict heart attack likelihood from retinal images.
*   **Utilize Non-Invasive Data Collection:** Employ retinal imaging as a safe and accessible method for cardiovascular risk assessment.
*   **Enable Early Detection:** Identify at-risk individuals proactively, allowing for timely preventive measures and improved patient outcomes.
*   **Create a User-Friendly Interface:** Design an intuitive interface for healthcare professionals to easily upload images, view predictions, and access insights.

## Key Features

*   **Retinal Image Analysis:** Processes retinal fundus images to extract key vascular features.
*   **Feature Clustering:** Utilizes Fuzzy C-Means (FCM) to group retinal features and visualize patterns (e.g., vessel thickness, tortuosity).
*   **Risk Prediction:** Employs Recurrent Neural Networks (RNN) and potentially hybrid models to predict heart attack risk levels (e.g., Low, Medium, High).
*   **Non-Invasive:** Provides a safe alternative to traditional invasive diagnostic procedures.
*   **User Authentication:** Secure login for administrators/users.
*   **Result Visualization:** Displays risk predictions along with relevant parameters (Age, SBP, DBP, BMI, Hemoglobin) and cluster images.

## System Architecture

The system follows a multi-stage pipeline:

1.  **User-End Flowchart:** Handles user authentication (Admin/User) and retinal image upload.
2.  **Clustering (Preprocessing):**
    *   Receives the uploaded retinal image.
    *   Applies Fuzzy C-Means (FCM) clustering for feature extraction and segmentation (identifying significant regions, patterns like blood vessels, optic disc).
    *   Generates a "Clustered Image" visualizing these segments.
3.  **RNN Model Prediction:**
    *   The clustered image/extracted features are fed into a trained Recurrent Neural Network (RNN).
    *   The RNN analyzes sequential/spatial patterns to predict the risk of a heart attack.
    *   Outputs the risk level and associated parameters.
4.  **Output Display:** Presents the prediction results, risk factors, and visualizations to the user/admin via the interface.

*(Reference: Figure 5.1 in the report)*

## Technologies Used

*   **Programming Language:** Python
*   **Machine Learning / Deep Learning:**
    *   TensorFlow / Keras
    *   Scikit-learn
    *   Recurrent Neural Networks (RNN)
    *   Fuzzy C-Means (FCM) Clustering
    *   (Potentially: AdaBoost, CNNs, RetinaNet/YOLO for feature detection)
*   **Image Processing:** OpenCV
*   **Data Handling:** Pandas, NumPy
*   **Visualization:** Matplotlib
*   **Development Environment:** Jupyter Notebook (for experimentation)
*   **Web Framework (for Deployment):** Flask / Django (mentioned as potential)
*   **Operating System:** Windows 10 (Development Env)
*   **Hardware:** Intel Core i5+, 8GB+ RAM, SSD recommended

## Dataset

The system requires a dataset of retinal fundus images. For training, these images should ideally be annotated with corresponding cardiovascular risk levels or outcomes. The report mentions training on a comprehensive dataset, but the specific public or private dataset used is not detailed here. Preprocessing steps are applied to clean, normalize, and augment the images.

*Note: Ensure you have the necessary permissions and ethical approvals if using sensitive medical image data.*

## Setup and Installation

1.  **Prerequisites:**
    *   Python (Version 3.7+ recommended)
    *   Pip (Python package installer)
    *   Git
    *   Virtualenv (Recommended)

2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

3.  **Create and Activate Virtual Environment:**
    ```bash
    # For Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure `requirements.txt` lists all necessary libraries like tensorflow, opencv-python, pandas, numpy, scikit-learn, matplotlib, etc.)*

5.  **Dataset Preparation:**
    *   Place your retinal eye images in the designated input folder (e.g., `data/input_images/`). Ensure they meet format/quality requirements mentioned in project documentation.
    *   Make sure the trained model file (e.g., `model.h5` or TensorFlow SavedModel) is placed in the correct directory (e.g., `models/`).

## Usage

1.  **Preprocessing (If needed separately):**
    ```bash
    python scripts/preprocess.py --input_dir data/input_images/ --output_dir data/processed_images/
    ```
    *(Adjust arguments based on your script)*

2.  **Run Prediction:**
    ```bash
    python scripts/predict.py --image_path path/to/single/image.jpg
    # OR
    python scripts/predict.py --input_dir data/processed_images/ --output_dir results/
    ```
    *(Adjust arguments based on your script)*

3.  **Run Evaluation (If ground truth is available):**
    ```bash
    python scripts/evaluate.py --predictions_dir results/ --ground_truth_dir data/test_labels/
    ```
    *(Adjust arguments based on your script)*

4.  **Run Web Application (If implemented with Flask/Django):**
    ```bash
    python app.py
    ```
    Access the application via your browser, typically at `http://127.0.0.1:5000` or `http://localhost:5000`.

## Testing

Unit and integration tests can be run to ensure components are working correctly.

1.  **Run Tests (using pytest):**
    ```bash
    pytest tests/
    ```
    *(Assuming tests are located in the `tests/` directory and configured for pytest)*

## Results

The system predicts the heart attack risk level based on the input retinal image and displays relevant parameters. Key outputs include:

*   **Risk Level:** Categorical (Low/Medium/High) or Percentage (e.g., 60% High Risk).
*   **Associated Parameters:** Displays factors like Age group, SBP, DBP, BMI, Hemoglobin range corresponding to the cluster/prediction.
*   **Visualizations:**
    *   Cluster Image: Shows segmented regions of the retina based on FCM.
    *   Parameter Graphs: Visual representation of features contributing to risk.

**Screenshots:**

*(Reference: Figures 8.1 - 8.7 in the report provide visual examples of the UI and outputs.)*

![Homepage](https://github.com/user-attachments/assets/b7f879fc-f601-47c9-ae0e-bc1b6291cc4b) <!-- Replace with actual path -->
![Login Page](https://github.com/user-attachments/assets/bd422b4f-e9e3-4b6f-ab6d-94f399e22040) <!-- Replace with actual path -->
![Clustering Input](https://github.com/user-attachments/assets/a559dd4c-a34b-4821-b654-406916e55556) <!-- Replace with actual path -->
![Cluster Image Output](https://github.com/user-attachments/assets/cab5c7f0-75ab-44be-9453-e518a9a87e84)<!-- Replace with actual path -->
![High Risk Output](https://github.com/user-attachments/assets/4d18c5a5-3d34-46e4-9412-7f97b1ef0341)<!-- Replace with actual path -->
![No Risk Output](https://github.com/user-attachments/assets/d8493754-8c78-40b1-ae9b-f2fd9743c2b3)<!-- Replace with actual path -->
![Feature Graph](https://github.com/user-attachments/assets/971eb3d1-2997-4dc5-a7ed-f8377f8a0565)<!-- Replace with actual path -->

*Reported Accuracy: The report mentions visualizations aiming for high accuracy (e.g., 98% target mentioned contextually on page 56), but formal validation metrics should be checked via the evaluation script.*

## Applications

*   **Early Diagnosis & Intervention:** Identify cardiovascular risk before symptoms manifest.
*   **Personalized Treatment Plans:** Tailor healthcare based on individual retinal biomarkers.
*   **Non-Invasive Monitoring:** Accessible and safe screening, especially for elderly or remote populations.
*   **Telemedicine & Remote Healthcare:** Facilitate remote consultations and diagnostics.
*   **Population Health Management:** Enable large-scale screening campaigns.
*   **Screening in High-Risk Occupations:** Monitor cardiovascular health in demanding professions (aviation, military).
*   **Geriatric Care:** Improve monitoring in nursing homes and assisted living.

## Limitations

*   **Scalability:** Processing large image volumes can be computationally intensive.
*   **Data Quality & Variability:** Prediction accuracy heavily depends on input image quality and consistency.
*   **Generalization:** Model performance may vary across diverse populations if the training data lacks representation.
*   **Privacy & Ethical Concerns:** Handling sensitive medical data requires robust security and compliance (HIPAA/GDPR).
*   **Integration Challenges:** Integrating with existing EHR systems can be complex.

## Future Scope

*   **Enhance Early Detection:** Further refine models for even earlier risk identification.
*   **Wearable Technology Integration:** Connect with smart glasses or portable retinal scanners for real-time monitoring.
*   **Improve Accuracy:** Incorporate more advanced AI models and larger, diverse datasets.
*   **Telemedicine Expansion:** Develop robust telemedicine platforms incorporating this technology.
*   **Personalized Healthcare:** Generate detailed, personalized health plans.
*   **Cross-Domain Applications:** Adapt the methods for predicting other systemic diseases (e.g., diabetes, neurological disorders) via retinal analysis.
*   **Federated Learning:** Implement privacy-preserving continuous model improvement.
*   **Regulatory & Clinical Validation:** Conduct rigorous trials for clinical adoption.
*   **Ecosystem Integration:** Develop standardized APIs for seamless EHR integration.

## Author

*   **PRASHANTHA K T** (*************)
    *   B.Tech, Artificial Intelligence and Machine Learning (AIML)
    *   University Visvesvaraya College of Engineering (UVCE), Bangalore University

## License

<!-- Specify the license for your project. Example: MIT License -->
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details (if applicable).
<!-- Or, if it's purely academic and not licensed: -->
This project was developed for academic purposes at University Visvesvaraya College of Engineering. Please contact the author for permissions regarding usage or distribution.

---
