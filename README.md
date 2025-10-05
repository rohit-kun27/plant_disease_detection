<h1>🌱 Plant Disease Detection using VGG19</h1>

<h2>Overview</h2>

This project uses a pre-trained deep learning model (VGG19) to accurately detect and classify 15 different types of plant diseases from the PlantVillage Dataset.
By leveraging transfer learning and fine-tuning, the model achieves high accuracy in identifying infected leaves, helping farmers and researchers with early disease diagnosis.

<h2>Features</h2>
<li>Detects 15 types of plant diseases from leaf images.</li>
<li>Built using pre-trained model VGG19 with fine-tuning for high accuracy.</li>
<li>Data augmentation applied for better generalization.</li>
<li>Achieved ~92% validation accuracy..</li>
<li>Includes a simple web app (Streamlit) for real-time prediction.</li>

<h2>Datasets</h2>
Dataset: <a href= "https://www.kaggle.com/datasets/emmarex/plantdisease">Plant Village Dataset</a>
<li>Total Images: ~20,000</li>
<li>15 Classes (Healthy + Diseased Leaves)</li>
<li>Split used:</li>
    - 80% Training
    - 20% Validation


<h2>Tech Stack</h2>
<li>Language: Python</li>
<li>Deep Learning Framework: TensorFlow / Keras</li>
<li>Model: VGG19 (Pre-trained on ImageNet)</li>
<li>Libraries Used:</li>
  * numpy, pandas, matplotlib, tensorflow, keras
  * sklearn (for metrics)
  * streamlit / gradio (for app interface


## How to Run 
- Clone the project:
```bash
git clone https://github.com/rohit-kun27/plant-disease-detection.git
cd plant-disease-detection
```

- Download Model </br>
  [Google Drive Link](https://drive.google.com/file/d/1tH8keyyQsoE302ROIetsKjMATrjikzJU/view?usp=drive_link)
  
- Install Dependencies:
```bash
pip install -r requirements.txt
```
- Run Streamlit app:
 ```bash
python -m streamlit run app.py
```
- Upload an image:
Upload a leaf image — the model will display:
  - Predicted disease name
  - Confidence score
 
## Fine Tunning Details 
- Base Model: VGG19 (ImageNet weights)
- Frozen Layers: All convolutional layers initially
- Unfrozen Layers: Last 4 convolutional blocks during fine-tuning
- Optimizer: Adam (lr=1e-5 during fine-tuning)
- Loss Function: Categorical Crossentropy


