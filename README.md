

# Object Localization: Feature Extraction vs. End-to-End Deep Learning

## Project Overview

This project, developed for the **ECE-4563 Machine Learning** course at **NYU Tandon**, explores and compares two fundamentally different paradigms for object localization:

1. **Classical Pipeline:** Dimensionality reduction via **Principal Component Analysis (PCA)** followed by a **Multi-Layer Perceptron (MLP)** regressor.
2. **End-to-End Deep Learning:** A **Convolutional Neural Network (CNN)** that processes raw pixel data directly to predict spatial coordinates.

The objective is to analyze how much spatial information is preserved through global feature compression (PCA) versus local hierarchical feature learning (CNN).

## Authors

* **Yanbo (Bob) Wang**
* **Peijie (Max) Ma**

---

## Task Definition

The task is a **regression problem** where the models must predict the bounding box of an MNIST digit randomly placed and scaled on a 64 $\times 64$ grayscale canvas.

### Input

* 64 $\times$ 64 Grayscale images.

### Output

The models predict a 4-element vector representing the normalized bounding box:



where $(x_c, y_c)$ is the center of the box, and $(w, h)$ are the width and height.

---

## Methodology & Pipeline

### 1. Synthetic Dataset Generation

Since MNIST digits are 28 $\times$ 28, we generate a 64 $\times$ 64 canvas, randomly scale the digit, and place it at a random valid location. The ground truth bounding box is calculated during this synthesis process.

### 2. Feature Extraction & Optimization (PCA+MLP)

* **K-Fold Cross-Validation:** We perform a 5-fold CV over a grid of PCA components $k$ \in $\{16, 32, 64, 128, 256, 512\}$.
* **Optimization Criterion:** The optimal k is selected based on the highest **Mean Intersection over Union (IoU)**:


* **Architecture:** The MLP consists of fully connected layers with ReLU activation and Dropout (p=0.1) to prevent overfitting on the latent features.

### 3. End-to-End Learning (CNN)

* **Architecture:** A series of `Conv2d` layers followed by `ReLU` and `AdaptiveAvgPool2d` to extract hierarchical spatial features.
* **Regressor:** A final set of linear layers with a `Sigmoid` activation ensures the output coordinates remain within the [0, 1] range.

---

## Evaluation Metrics

We use two primary metrics to evaluate performance:

1. **Mean Squared Error (MSE):** Measures the pixel-wise distance between predicted and ground-truth coordinates.
2. **Intersection over Union (IoU):** Measures the overlap between the predicted and actual bounding boxes. This is the gold standard for localization tasks.



---

## Requirements

To run this project, you need the following Python libraries:

* `numpy`
* `matplotlib`
* `scikit-learn`
* `torch`
* `torchvision`

## Environment Setup

We recommend using a virtual environment.

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

pip install -r requirements.txt


## Usage

1. **Open the Notebook:** Load `final_proj_v2.ipynb` in Jupyter or VS Code.
2. **Dataset Generation:** Run the first few cells to download MNIST and generate the 64 \times 64 synthetic samples.
3. **PCA Grid Search:** Execute the K-Fold CV section to find the optimal k for the MLP.
4. **Training:** Run the final training cells for both the optimized PCA+MLP and the CNN.
5. **Visualizations:** Use the provided plotting functions to see the scatter plots and qualitative results (bounding boxes overlaid on test images).

---

## Summary of Results

The project provides a comprehensive comparison through:

* **Quantitative Analysis:** MSE and IoU comparison tables.
* **Convergence Analysis:** CV Mean MSE vs. k plots.
* **Qualitative Analysis:** Visual display of 12 random test samples showing the "Ground Truth" (Green) vs. "Prediction" (Red).
