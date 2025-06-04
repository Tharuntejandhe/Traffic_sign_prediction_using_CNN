# 🚦 Traffic Sign Prediction Using CNN

This project implements a Convolutional Neural Network (CNN) to classify traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The model is trained to accurately identify various traffic signs, aiding in the development of autonomous driving systems and advanced driver-assistance systems (ADAS).

---

## 📁 Dataset

The model utilizes the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) dataset, which comprises over 50,000 images categorized into 43 classes of traffic signs. This dataset is widely used for benchmarking traffic sign recognition algorithms.

---

## 🧠 Model Architecture

The CNN architecture is designed to extract features from input images and classify them into one of the 43 traffic sign categories. The model includes convolutional layers for feature extraction, pooling layers for dimensionality reduction, and fully connected layers for classification.

---

## 🛠️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Tharuntejandhe/Traffic_sign_prediction_using_CNN.git
   cd Traffic_sign_prediction_using_CNN
   ```

2. **Install dependencies:**

   Ensure you have Python installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

1. **Prepare the dataset:**

   Download the GTSRB dataset and place it in the appropriate directory as expected by the code.

2. **Train the model:**

   Run the training script to train the CNN model:

   ```bash
   python train.py
   ```

3. **Evaluate the model:**

   After training, evaluate the model's performance on the test set:

   ```bash
   python evaluate.py
   ```

---

## 📊 Results

The trained CNN model achieves high accuracy in classifying traffic signs, demonstrating its effectiveness for real-world applications in traffic sign recognition.

---

## 📂 Repository Structure

```
Traffic_sign_prediction_using_CNN/
├── detect.py
├── model_traffic_data1.h5
├── traffic-sign-detection.ipynb
├── Train.csv
├── Test.csv
├── Meta.csv
├── README.md
└── requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository and submit a pull request.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For more information on traffic sign recognition using CNNs, you can refer to this helpful tutorial:

👉 [Traffic Signs Recognition Using CNN and Keras – Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/12/traffic-signs-recognition-using-cnn-and-keras-in-python/)
