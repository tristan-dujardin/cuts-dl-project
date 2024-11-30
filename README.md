# Retina Model Training and Testing Framework

This project provides a framework for training and testing a deep learning model to work with retina image datasets. The repository includes utilities for data handling, model training, testing, and image transformation.

---

## Features
- **Data Splitting**: Automatically splits the dataset into training, validation, and test sets.
- **Customizable Model Training**: Allows parameter adjustments such as learning rate and the number of epochs.
- **Image Transformations**: Interprets and transforms images using k-means clustering.
- **Save and Load Model**: Save trained model weights or load pre-trained weights.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tristan-dujardin/cuts-dl-project.git
   cd cuts-dl-project
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the project directory structure is as follows:
   ```
   .
   ├── data/
   ├── src/
   │   ├── main.py
   │   ├── dataset/
   │   │   ├── dataset_utils.py
   │   │   └── retina.py
   │   ├── model/
   │   │   └── model.py
   │   └── image/
   │       └── kmeans_interpreter.py
   ├── models/
   │   └── weights.pth (optional: pre-trained model weights)
   ├── images/
   ├── README.md
   └── requirements.txt
   ```

---
## Loading the data
Run the `src/dataset/dataset.py` script from the root of the project, this will download the dataset in the correct folder.

---
## Usage

Run the `src/main.py` script from the root of the project:

### Train the Model
```bash
python src/main.py --mode train --lr 0.03 --epochs 10
```

- **`--mode`**: Set to `train` to train the model.
- **`--lr`**: Learning rate for the optimizer (default: `0.03`).
- **`--epochs`**: Number of training epochs (default: `100`).

After training, you will be prompted to save the model weights:
```plaintext
Will you Save the model? [y/N]
```

### Test the Model
```bash
python src/main.py --mode test
```

This mode will load the model weights from `models/weights.pth` and evaluate its performance on the test set.

### Image Transformation
The script will process images in the `images/` directory using k-means clustering.

---

## Dataset
The project uses a custom retina dataset. The data is loaded via the `Retina` class in `dataset/retina.py`. You may need to adapt the data loading functionality to match your dataset.
The dataset is available [here](https://github.com/KrishnaswamyLab/CUTS/tree/main/data/retina.zip).

---

## Outputs
- **Model Checkpoints**: Saved in `models/weights.pth`.
- **Run Logs**: Prints training/validation loss during training and test loss during evaluation.
- **Transformed Images**: Saved in the `images/` directory.

---

## Requirements
- Python 3.8+
- PyTorch
- Additional libraries specified in `requirements.txt`.
