## 🌱 Real-Time Plant Disease Detection

This project uses Computer Vision and a Deep Learning model (TensorFlow/Keras) to detect plant diseases in real-time using a webcam. The model can classify multiple plant species and their common diseases with high accuracy.

📌 Features

- Real-time plant disease detection using OpenCV.

- Deep learning model trained with TensorFlow/Keras.

- Supports multiple crops like Apple, Corn, Tomato, Potato, Grape, and more.

- Uses a deque-based smoothing mechanism to stabilize predictions over time.

- Simple and interactive — just run and detect diseases instantly.


Dataset Link:
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset


📂 Project Structure
├── main.py                # Real-time detection script
├── trained_model.keras    # Trained deep learning model
├── Test_Plant_Disease.ipynb  # Notebook for testing/evaluating the model
├── README.md              # Project documentation


🚀 How to Run

1. Clone the repository or copy project files into your workspace.

2. Make sure you have the trained model file: trained_model.keras.

3. Run the script:
   python main.py

4. The webcam will open, and predictions will be displayed on the screen.

5. Press q to quit the application


🌾 Supported Classes

The model can detect the following diseases and healthy states:

Apple: Apple Scab, Black Rot, Cedar Apple Rust, Healthy

Blueberry: Healthy

Cherry: Powdery Mildew, Healthy

Corn (Maize): Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy

Grape: Black Rot, Esca (Black Measles), Leaf Blight, Healthy

Orange: Haunglongbing (Citrus Greening)

Peach: Bacterial Spot, Healthy

Pepper (Bell): Bacterial Spot, Healthy

Potato: Early Blight, Late Blight, Healthy

Raspberry: Healthy

Soybean: Healthy

Squash: Powdery Mildew

Strawberry: Leaf Scorch, Healthy

Tomato: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy


📊 Model Training (Optional)

If you want to train or test the model yourself:

Open Test_Plant_Disease.ipynb.

Use a dataset like PlantVillage.

Train and save your model as trained_model.keras.


⚡ Future Improvements

- Add a web interface for mobile/remote usage.

- Integrate with IoT devices (Raspberry Pi, drones) for field monitoring.

- Extend dataset for more plant species and diseases.

