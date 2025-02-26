
# Fertilizer Recommendation System

## Overview
The Fertilizer Recommendation System is a web-based application that suggests the best fertilizer based on soil type, crop type, and nutrient levels. The system uses a machine learning model trained with TensorFlow to provide accurate recommendations.

## Features
- User-friendly web interface for fertilizer prediction.
- Accepts soil type, crop type, and nutrient levels (Nitrogen, Phosphorous, Potassium) as input.
- Utilizes a trained deep learning model to recommend the best fertilizer.
- Stores new user inputs and predictions in a CSV file for continuous learning.
- Built with Flask for backend functionality and HTML, CSS, and JavaScript for the frontend.

## Project Structure
```
├── app.py                   # Flask backend API
├── index.html               # Frontend UI
├── styles.css               # Styling for the frontend
├── fertilizer.h5            # Trained deep learning model
├── label_encoder.pkl        # Label encoder for categorical data
├── scaler.pkl               # Scaler for normalizing input data
├── f2.csv                   # Dataset for storing input-output pairs
├── FertilizerRecommendation.ipynb  # Jupyter Notebook with model training
├── static/                  # Folder containing images and additional resources
└── templates/               # Folder containing HTML templates
```

## Installation & Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Flask
- TensorFlow
- Pandas
- NumPy
- Joblib

### Installation Steps
1. Clone the repository or download the project files.
2. Install the required Python libraries:
   ```sh
   pip install flask tensorflow pandas numpy joblib
   ```
3. Place all required files (`fertilizer.h5`, `label_encoder.pkl`, `scaler.pkl`, and `f2.csv`) in the project directory.
4. Run the application:
   ```sh
   python app.py
   ```
5. Open your browser and navigate to `http://127.0.0.1:5000/` to use the application.

## Usage
1. Select the soil type from the dropdown menu.
2. Select the crop type from the dropdown menu.
3. Enter values for Nitrogen, Phosphorous, and Potassium.
4. Click the "Predict Fertilizer" button to receive a recommendation.

## Model & Data
- The model is a TensorFlow deep learning model trained to predict fertilizers based on input parameters.
- The dataset (`f2.csv`) is used to store user inputs and past predictions for continuous learning.
- Label encoding and feature scaling are used to preprocess input data before making predictions.

## Future Improvements
- Improve model accuracy with additional training data.
- Add support for more soil and crop types.
- Enhance UI with more interactive features.
- Deploy the application online for broader access.

## Author
Developed as part of an AI project on Fertilizer Recommendation.

## License
This project is for educational purposes. Feel free to use and modify as needed.

![Screenshot 2025-02-26 211624](https://github.com/user-attachments/assets/67a541e0-8da3-4ebb-830c-395cfd3ebfd4)
![Screenshot 2025-02-26 211720](https://github.com/user-attachments/assets/d6a7e194-9bca-4145-891d-777b81e34bb6)




