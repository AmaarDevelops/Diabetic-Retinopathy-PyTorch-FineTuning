# 👁️ Diabetic Retinopathy (DR) Classifier Web App

## 🎯 Project Overview

This project implements a deep learning pipeline using PyTorch and the EfficientNet-B0 architecture to classify fundus images for Diabetic Retinopathy (DR). 
The trained model is deployed via a lightweight **Flask** web application, allowing users to upload an image and instantly receive an automated diagnosis (No DR, Mild, Moderate, Severe, Proliferative DR).

The core pipeline includes stratified data splitting, pre-processing, fine-tuning, and deployment.

---

## 🚀 Getting Started

Follow these steps to set up the project locally and run the Flask application.

### Prerequisites

You need Python 3.8+ installed on your system.

### 1. Clone the Repository

``bash

git clone https://github.com/AmaarDevelops/Diabetic-Retinopathy-PyTorch-FineTuning

cd Diabetic-Retinopathy-PyTorch-FineTuning

2. Set Up the Environment It's highly recommended to use a virtual environment.
3. 
python3 -m venv venv
source venv/bin/activate

# Create and activate environment (Windows)
python -m venv venv

.\venv\Scripts\activate

3. Install DependenciesInstall all necessary Python libraries (PyTorch, Flask, PIL, etc.).
   
Bash pip install torch torchvision pandas scikit-learn flask pillow

🧠 Model and Data Setup 
1. Obtain and Structure DataThis project relies on the Diabetic Retinopathy dataset. You must download and place the data so that your data structure looks like this :-
   
 :/Project_Root
  |-- /data
      |-- /images
          |-- /colored_images
              |-- /Mild
              |-- /No_DR
              |-- ... (other class folders)
      |-- train.csv
  |-- /data/processed  <-- Created by data_spliter.py
  |-- /models
      |-- best_fine_tuned_model.pth <-- Your trained model file
  |-- /src
      |-- data_spliter.py
      |-- app.py
      
4. Run Data Preprocessing Execute the data_spliter.py script to perform stratified splitting and organize the images into numeric class folders (0 through 4) required by the PyTorch data loaders.
5. 
6. Bash python src/data_spliter.py
7. 
8. Place Trained ModelEnsure your trained EfficientNet-B0 weights are placed in the expected location:./models/best_fine_tuned_model.pth
   
10. 💻 Running the Web Application Navigate to the project root and run the main Flask application.Bashpython app.py
    
The application will start, and you will see output like: * Running on [http://127.0.0.1:5000](http://127.0.0.1:5000) (Press CTRL+C to quit)

Open your web browser and navigate to http://127.0.0.1:5000. 

You can now upload a retinal image for instant diagnosis.
⚙️ Core Project File/FolderDescription 

app.py Flask Backend: Initializes the server, loads the PyTorch model, handles file uploads, and performs prediction.

src/data_spliter.pyData Preparation: Reads the train.csv, performs stratified split (Training/Validation/Testing), and copies images into a numeric folder structure.

/models Directory containing the trained PyTorch model state dictionary (best_fine_tuned_model.pth)

./templates Contains index.html, the main frontend interface for image upload and result display. 

/static Contains static assets: style.css (styling) and script.js (AJAX submission/UI updates).predict_image() The core inference function in app.py that preprocesses the image and runs the forward pass.

🔬 Model Details Architecture: Fine-tuned EfficientNet-B0 (Transfer Learning). Input Size: 256x256 pixels.Output Classes (Diagnosis): 5 classes, mapped as follows:0: No_DR1: Mild2: Moderate3: Severe4: Proliferate_DR (Proliferative Diabetic Retinopathy)

🤝 Contributing If you wish to contribute, please clone the repository, create a new branch for your feature or fix, and submit a pull request.

📄 LicenseThis project is open-source and available under the MIT License. (You can change this to your preferred license).
