# WordWeaver

**WordWeaver** is a character-level text generation model designed for learning and generating text patterns. Currently, the model is trained on a dataset of popular names, making it ideal for generating new name suggestions or exploring text generation capabilities.

This repository includes an interactive Jupyter Notebook (`wordweaver.ipynb`) that implements the model, prepares the dataset, trains the neural network, and generates text outputs.

---

## Features

- **Character-level text generation**: Learn patterns from sequences of characters to generate new names.
- **Custom neural network design**: Includes flexible and modular layers such as `BatchNorm1d`, `FlattenConsecutive`, and `Sequential`.
- **Interactive and customizable**: Modify the dataset, model architecture, or hyperparameters within the notebook.
- **Trained on a names dataset**: Works with a curated dataset of names for generation.

---

## Dataset

The model operates on a dataset of popular names, such as:

```plaintext
emma
olivia
ava
isabella
sophia
charlotte
mia
amelia
harper
evelyn
abigail
emily
```

### Dataset Format:
- A plain text file (`.txt`) where each line contains a single name.
- The dataset includes a special end token (`.`) that signifies the end of a name.

You can replace this dataset with your own, provided it follows a similar format.

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/wordweaver.git
cd wordweaver
```

### 2. Install Dependencies
(Optional but recommended) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook
Start the notebook server:
```bash
jupyter notebook
```

Open `wordweaver.ipynb` from the notebook interface.

---

## Using the Notebook

### 1. Load the Dataset
The dataset is already provided in the notebook. You can replace it with your own dataset of names or other sequences by modifying the `words` list.

### 2. Train the Model
Run the training cells in the notebook. The key parameters include:
- **Block Size**: Determines the number of previous characters used as context for predicting the next character.
- **Embedding Size**: Dimensionality of the character embeddings.
- **Hidden Neurons**: Number of neurons in the fully connected layers.

### 3. Generate Names
Once trained, the model generates new names character by character. Simply run the generation cells to see the results.

Example generated output:
```plaintext
emma
sophial
amelianna
harpe.
olivialyn
```

---

## Model Architecture

The model architecture includes:
1. **Embedding Layer**: Encodes characters into dense vectors.
2. **Hidden Layers**:
   - Custom `FlattenConsecutive` layers to reduce input dimensions.
   - Fully connected layers with batch normalization and `tanh` activations.
3. **Output Layer**: Predicts probabilities for the next character.

---

## Training Details

- **Loss Function**: Cross-entropy loss for character-level prediction.
- **Optimizer**: Stochastic Gradient Descent (SGD) with dynamic learning rate adjustment.
- **Hyperparameters**:
  - Block size: `8`
  - Embedding size: `24`
  - Hidden layer size: `128`
  - Initial learning rate: `0.1`, decays to `0.01` after 150,000 steps.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgments

- PyTorch for providing a robust deep learning framework.
- The dataset of popular names used in this example for training.
