# WordWeaver

**WordWeaver** is a deep learning model designed to generate text character by character. Using a sequence-to-sequence neural network, this model learns patterns in text data and generates new words, similar to the way humans compose sentences.

This repository contains the implementation of the model, training scripts, and instructions to run the project locally.

---

## Features

- **Character-level text generation**: The model generates text based on the input sequence of characters.
- **Customizable model architecture**: The neural network architecture can be customized for various text generation tasks.
- **Training on your own dataset**: Easily train the model on any text dataset.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/wordweaver-model.git
   cd wordweaver-model
   ```

2. Set up a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset

This model is designed to work with character-level text datasets. You can use any text data, such as books, articles, or even your own corpus of text.

To prepare your dataset:
- Ensure that the text data is cleaned and formatted as a list of words or sentences.
- The text data should be stored in a plain `.txt` file.

---

## Training the Model

### 1. Prepare Your Dataset:
- Place your `.txt` dataset in the `data/` directory.

### 2. Training:
To start training, run the following command:

```bash
python train.py --dataset data/your-dataset.txt --epochs 10 --batch_size 32
```

Adjust the `epochs` and `batch_size` parameters as needed.

### 3. Evaluate the Model:
After training, you can evaluate the model on the validation set:

```bash
python evaluate.py --model saved_model.pth
```

---

## Generating Text

Once the model is trained, you can generate text based on a given prompt.

To generate text:

```bash
python generate.py --model saved_model.pth --prompt "Once upon a time"
```

The model will generate text that continues the prompt based on the learned patterns.

---

## Model Architecture

The model consists of the following components:
1. **Embedding Layer**: Converts characters into dense vectors.
2. **Fully Connected Layers**: A series of hidden layers with batch normalization and activation functions.
3. **Output Layer**: Produces logits for each character in the vocabulary.

---

## Hyperparameters

- **embedding size**: 24
- **hidden layer size**: 128
- **block size (sequence length)**: 8
- **learning rate**: 0.1 (decays after 150,000 steps)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- PyTorch for providing the deep learning framework.
- All datasets used in this project.
