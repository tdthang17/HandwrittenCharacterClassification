# Handwritten Character Classification

A web-based application that recognizes handwritten characters (letters and numbers) using deep learning. The application allows users to either draw characters or upload images for recognition.

## Features

- **Interactive Drawing Canvas**: Draw characters with your mouse or touch input
- **Image Upload**: Upload images containing handwritten characters
- **Real-time Prediction**: Get instant character recognition results
- **Confidence Score**: See the model's confidence level for each prediction
- **Responsive Design**: Works on both desktop and mobile devices


## Model Architecture

The application uses a deep learning model based on VGG19 architecture with the following layers:

1. VGG19 base model (pre-trained on ImageNet, frozen weights)
2. Flatten layer
3. Dense layer (224 units, ReLU activation)
4. Dropout (0.1)
5. Dense layer (416 units, sigmoid activation)
6. Dropout (0.1)
7. Output layer (62 units, softmax activation)

The model is trained to recognize 62 different characters (0-9, A-Z, a-z).

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tdthang17/HandwrittenCharacterClassification.git
   cd HandwrittenCharacterClassification
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask development server:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:8000/
   ```

3. Use the application:
   - Draw a character on the canvas or upload an image
   - Click "Predict Now" or wait for auto-prediction
   - View the predicted character and confidence score

## Training the Model

The model was trained on the [Handwritten English Characters and Digits](https://www.kaggle.com/datasets/sujaymann/handwritten-english-characters-and-digits) dataset. To retrain the model:

1. Download the dataset and place it in the appropriate directory
2. Run the training notebook: `model/Train_Chu_So_Viet_Tay.ipynb`
3. The trained model will be saved to `model/model.keras`

## Project Structure

```
HandwrittenCharacterClassification/
├── app.py                # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── static/               # Static files (CSS, JS, images)
│   ├── style.css         # Styling for the web interface
│   └── Images/           # Image assets
├── templates/            # HTML templates
│   └── index.html        # Main web interface
└── model/                # Model training files
    ├── Train_Chu_So_Viet_Tay.ipynb  # Training notebook
    └── model.keras       # Trained model weights
```

## Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla JS)
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV, PIL
- **Data Processing**: NumPy, Pandas

## Acknowledgments

- The model was trained using the [Handwritten English Characters and Digits](https://www.kaggle.com/datasets/sujaymann/handwritten-english-characters-and-digits) dataset
- Uses the VGG19 model pre-trained on ImageNet
- Built with Flask and TensorFlow

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
