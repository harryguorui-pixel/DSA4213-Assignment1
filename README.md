# DSA4213 Assignment 1 — Language Models, Embeddings, and Representations

National University of Singapore  
DSA4213: Natural Language Processing for Data Science  
Semester 1, AY2026/2027

## 1. Project Overview

This project investigates language modeling, input embedding strategies, and the transferability of learned language representations.

The assignment consists of three parts:

- **Part I — Language Model Comparison:** Train and compare Trigram, Vanilla RNN, LSTM, and causal Transformer language models on WikiText-2.
- **Part II — Embedding Variants:** Compare Random Trainable, Self-trained Skip-gram Frozen, and Public GloVe Frozen embeddings using the same Vanilla RNN architecture.
- **Part III — Downstream Representation Analysis:** Evaluate frozen RNN representations on AG News topic classification using identical linear classifiers.

All language models were trained from scratch, except for the explicitly permitted use of public pretrained GloVe word embeddings in Part II.

## 2. Environment and Dependencies

The experiments were conducted using:

- OS: Windows
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Python: Virtual environment (`.venv`)
- Deep learning framework: PyTorch 2.10.0+cu128
- CUDA: 12.8-compatible PyTorch build
- Gensim: 4.4.0
- scikit-learn: 1.9.1
- Random seed: 42

Other dependencies are listed in `requirements.txt`.

### Environment Setup

Create and activate a virtual environment using PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install a compatible CUDA-enabled PyTorch build following the official PyTorch installation instructions:

https://pytorch.org/get-started/locally/

Install the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip install gensim==4.4.0
```

The requirements file records the original CUDA-specific PyTorch version. Depending on the platform and package index, the CUDA wheel may need to be installed separately.

Verify GPU availability:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 3. Datasets and Preprocessing

### Part I and Part II: WikiText-2

Dataset: `Salesforce/wikitext`  
Configuration: `wikitext-2-raw-v1`

Dataset source: https://huggingface.co/datasets/Salesforce/wikitext

The original training, validation, and test splits were retained.

Preprocessing:

- Word-level tokenization using a regular expression
- Lowercase conversion
- Vocabulary size: 10,000
- Special tokens: `<pad>`, `<unk>`, `<eos>`
- Vocabulary constructed using training data only
- Sequence length: 32
- Batch size: 64
- Random seed: 42

The vocabulary and encoded datasets are saved under `data/processed/`.

Run data preparation and basic checks:

```powershell
python main.py
```

**Important:** Running `main.py` regenerates the processed WikiText-2 data. Existing processed files should be preserved when reproducing downstream experiments from saved checkpoints.

### Part III: AG News

Dataset: `fancyzhx/ag_news`

Dataset source: https://huggingface.co/datasets/fancyzhx/ag_news

The task is four-class news topic classification:

- World
- Sports
- Business
- Sci/Tech

Data splits:

| Split | Samples |
|---|---:|
| Training | 8,000 |
| Validation | 2,000 |
| Test | 7,600 |

Training and validation subsets were created through seeded stratified sampling from the official training split. The complete official test split was retained.

The original WikiText-2 tokenizer and vocabulary were reused without rebuilding the vocabulary. Inputs were padded or truncated to 32 tokens.

The observed unknown-token rate was approximately 16.5%, and more than 86% of examples exceeded the maximum sequence length.

Prepare the classification data:

```powershell
python -m src.part3.dataset
```

Processed data is stored under `data/part3/processed/`.

## 4. Part I — Language Model Training

Four next-token language models were trained using the same WikiText-2 corpus.

Models:

1. Trigram with Add-k smoothing
2. Vanilla RNN
3. LSTM
4. Transformer with causal self-attention

Neural training configuration:

| Parameter | Value |
|---|---|
| Vocabulary size | 10,000 |
| Embedding dimension | 100 |
| Sequence length | 32 |
| Batch size | 64 |
| Epochs | 5 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Random seed | 42 |

The best neural checkpoints were selected using validation loss.

### Training Commands

```powershell
python run_ngram.py
python run_rnn.py
python run_lstm.py
python run_transformer.py
```

### Evaluation

```powershell
python evaluate_all.py
```

The unified evaluation uses a window length of 32 and evaluates target positions starting at index 2.

All four models were evaluated on the same 236,790 target tokens.

### Part I Results

| Model | Test CE | Test PPL | Training Time |
|---|---:|---:|---:|
| Trigram | 6.5192 | 678.02 | 3.34 s |
| RNN | 4.6292 | 102.43 | 61.60 s |
| LSTM | 4.6784 | 107.60 | 68.88 s |
| Transformer | 4.6474 | 104.31 | 49.34 s |

Main comparison results are stored in `results/metrics/part1_final_comparison.csv`.

## 5. Part II — Embedding Variants

Part II compares three input embedding strategies using the same Vanilla RNN architecture.

| Setting | Embedding | Trainability |
|---|---|---|
| A | Random initialization | Trainable during LM training |
| B | Self-trained Skip-gram | Frozen |
| C | Public GloVe 100d | Frozen |

All embeddings have dimension 100.

The RNN architecture uses two recurrent layers with hidden dimension 256 and dropout 0.2.

### Setting A

Reuses the trained Part I Vanilla RNN.

### Setting B

Skip-gram embeddings were trained using Gensim Word2Vec on the WikiText-2 training split only.

Configuration:

- Vector size: 100
- Window size: 5
- Negative samples: 5
- Epochs: 5
- Seed: 42

The resulting embedding table was frozen during RNN training.

### Setting C

Uses public pretrained `glove-wiki-gigaword-100` word embeddings.

- Matched words: 9,919 / 9,997
- Vocabulary coverage: 99.22%
- OOV words: 78
- OOV initialization: seeded random normal
- Entire embedding table frozen during RNN training

### Part II Commands

Prepare and validate embedding configurations:

```powershell
python -m tests.test_part2_setup
python -m src.part2.train_skipgram
python -m src.part2.prepare_glove
```

Train the two additional RNN language models:

```powershell
python run_part2.py --setting B
python run_part2.py --setting C
```

Evaluate and generate results:

```powershell
python evaluate_part2.py
python generate_part2.py
python plot_part2.py
```

### Part II Results

| Setting | Test CE | Test PPL | RNN Training Time |
|---|---:|---:|---:|
| Random Trainable | 4.629212 | 102.4333 | 61.60 s |
| Skip-gram Frozen | 4.557257 | 95.3216 | 53.78 s |
| GloVe Frozen | 4.574358 | 96.9657 | 55.73 s |

Self-trained Skip-gram embedding training required an additional 18.14 seconds.

The Skip-gram setting achieved the lowest test perplexity. However, the public GloVe embeddings were trained on additional external text, which limits direct attribution of performance differences to the embedding algorithm alone.

## 6. Part III — Downstream Representation Analysis

Part III evaluates whether the three RNN representations from Part II transfer effectively to supervised news classification.

### Representation Extraction

The three pretrained language models were frozen.

Each news article was encoded using the existing WikiText-2 tokenizer and vocabulary. Its representation was extracted as the final valid hidden state of the last RNN layer.

Padding tokens were excluded from the recurrent computation using `pack_padded_sequence`.

Each representation has dimension 256.

Extract and cache the representations:

```powershell
python -m src.part3.representations
```

Cached features are saved under `data/part3/features/`.

### Linear Classifier Training

A separate linear classifier was trained for each representation.

All classifiers use the same architecture and training procedure:

| Parameter | Value |
|---|---|
| Classifier | Linear(256, 4) |
| Trainable parameters | 1,028 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Maximum epochs | 10 |
| Early stopping patience | 3 |
| Model selection | Lowest validation loss |
| Random seed | 42 |

Train the classifiers:

```powershell
python -m src.part3.classifier
```

### Evaluation

```powershell
python evaluate_part3.py
```

The evaluation reports test accuracy, macro-F1, class-level precision/recall/F1, confusion matrices, and representative classification errors.

### Part III Results

| Representation | LM Test PPL | Test Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Random | 102.4333 | 44.42% | 0.4353 |
| Skip-gram | 95.3216 | 62.59% | 0.6221 |
| GloVe | 96.9657 | 64.57% | 0.6416 |

GloVe achieved the highest downstream classification performance, although Skip-gram achieved the lowest language-model perplexity.

This demonstrates that the language-model perplexity ranking did not fully agree with the downstream classification ranking in this experiment.

Limitations include domain shift between WikiText-2 and AG News, unknown tokens, substantial input truncation, and the additional external training corpus used by GloVe.

## 7. Output Files and Checkpoints

### Language Model Checkpoints

```text
checkpoints/
    ngram.pkl
    rnn_best.pt
    lstm_best.pt
    transformer_best.pt
    part2/
        rnn_skipgram_best.pt
        rnn_glove_best.pt
```

### Part III Classifier Checkpoints

```text
checkpoints/part3/
    classifier_random_best.pt
    classifier_skipgram_best.pt
    classifier_glove_best.pt
```

### Experiment Results

```text
results/
    metrics/
        part1_final_comparison.csv
        part2/
        part3/
            part3_final_comparison.csv
    figures/
        part2/
        part3/
    generated_text/
        part2/
```

Large generated datasets, cached representations, and potentially large model checkpoints may be excluded from version control. If they are unavailable in a fresh clone, reproduce them using the commands above.

The Part III workflow requires the three pretrained RNN checkpoints from Parts I and II.

## 8. Reproducibility

All experiments use random seed 42.

Configuration files:

- `configs/config.yaml`: Part I
- `configs/part2.yaml`: Part II
- `configs/part3.yaml`: Part III

For full reproduction, run data preparation before model training, and complete Parts I and II before Part III.

For reproducing only Part III from existing language-model checkpoints, run:

```powershell
python -m src.part3.dataset
python -m src.part3.representations
python -m src.part3.classifier
python evaluate_part3.py
```

All commands should be executed from the repository root.

GPU hardware, library versions, and numerical implementation differences may affect exact results.

## 9. References and Tool Use

External resources used in this project include:

- WikiText-2 dataset: https://huggingface.co/datasets/Salesforce/wikitext
- AG News dataset: https://huggingface.co/datasets/fancyzhx/ag_news
- GloVe embeddings: https://nlp.stanford.edu/projects/glove/
- Gensim Word2Vec: https://radimrehurek.com/gensim/models/word2vec.html
- PyTorch: https://pytorch.org/

ChatGPT was used as a coding and experiment-planning assistant, including assistance with implementation structure, debugging, validation procedures, and report drafting. Experimental measurements and reported results were obtained from locally executed code.