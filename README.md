# 🫁 Pneumonia Detection using Deep Learning

This project uses Convolutional Neural Networks (CNNs) to detect pneumonia from chest X-ray images. The model is trained on the [Kaggle Chest X-ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), and built using Python, TensorFlow/Keras, OpenCV, and Matplotlib.

---

## 🔍 Project Overview

Pneumonia is a serious lung infection that can be detected using chest X-rays. This deep learning project automates the diagnosis by classifying X-ray images as either **"Normal"** or **"Pneumonia"**.

---

## 📁 Dataset

- Source: Kaggle Chest X-ray Dataset
- Structure:
chest_xray/
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
├── test/
└── val/

---

## 🛠️ Tools & Technologies

- Python 🐍
- TensorFlow / Keras
- OpenCV
- Matplotlib & Seaborn
- Jupyter Notebook / VS Code

---

## 📌 Features

- CNN-based binary image classifier
- Image preprocessing with OpenCV
- Data augmentation using ImageDataGenerator
- Real-time prediction on custom images
- Model saving and reusability

---

## ⚙️ Installation & Setup

1. **Clone the repo**
 ```bash
 git clone https://github.com/your-username/pneumonia-detection.git
 cd pneumonia-detection
2.Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
3.Install Required Libraries
pip install -r requirements.txt
4.Run the Script
python main.py

🧠 Model Architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

📊 Evaluation
The model is trained for 5 epochs using training and validation datasets.

Test accuracy and loss are calculated on a separate test set.

Output example:
Test Accuracy: 0.92

🔍 Sample Prediction
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        print("Prediction: Pneumonia Detected")
    else:
        print("Prediction: Normal X-ray")
💾 Model Saving
model.save("pneumonia_model.h5")
print("Model saved successfully!")

🙌 Acknowledgements
Kaggle for the dataset

TensorFlow & Keras team

OpenCV and Matplotlib contributors

📌 Future Work
Web app deployment with Streamlit or Flask

Integration with cloud services or mobile apps

Extended training with more epochs or larger datasets

👤 Author
Vinay Kumar
B.Tech CSE (AI & ML), Srinivasa Ramanujan Institute of Technology

