# Dravidian Multi-Lingual AI-generated Reviews Detection

A deep learning framework for detecting AI-generated product reviews in Dravidian languages using cultural, syntactic, and semantic analysis. The model employs a multi-modal architecture combining Indic-BERT with specialized encoders for comprehensive text analysis.

## Features

- Cultural context encoding using Indic-BERT
- Syntactic analysis with multi-head attention
- Semantic analysis for coherence detection
- Support for multiple Dravidian languages
- High-performance classification with F1-score optimization

## Requirements

```
torch>=1.9.0
transformers>=4.11.0
pandas>=1.3.0
scikit-learn>=0.24.0
tqdm>=4.62.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dravidian-mlf-detector.git
cd dravidian-mlf-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

1. Prepare your training data in CSV format with columns 'DATA' (text) and 'LABEL' ('AI' or 'Human').

2. Update the data path in `train.py`:
```python
train_df = pd.read_csv('path_to_training_data.csv')
```

3. Run training:
```bash
python train.py
```

### Inference

1. Load a trained model and run inference:
```python
from inference import inference
from transformers import AutoTokenizer
import torch
from models import DravidianMLFDetector
from config import dravidian_config

# Initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DravidianMLFDetector(dravidian_config).to(device)
model.load_state_dict(torch.load('best_malayalam_model.pth'))
tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')

# Predict
texts = ["Your text here"]
predictions = inference(model, texts, tokenizer, device)
```

## Model Configuration

You can modify the model configuration in `config.py`:
```python
dravidian_config = {
    'cultural_model_path': 'ai4bharat/indic-bert',
    'semantic_model_path': 'ai4bharat/indic-bert',
    'embedding_dim': 384,
    'num_attention_heads': 8
}
```

## Project Structure

```
dravidian-mlf-detector/
├── models.py         # Model architecture
├── dataset.py       # Dataset handling
├── train.py         # Training script
├── inference.py     # Inference script
├── config.py        # Configuration
└── README.md
```

## Training Data Format

The training data should be a CSV file with the following format:

```csv
DATA,LABEL
"Sample text 1",Human
"Sample text 2",AI
```

## Model Performance

The model is evaluated using macro F1-score, considering both precision and recall for AI and human-generated text detection. The validation phase during training provides detailed classification reports. This framework was submitted as a part of **Shared Task on Detecting AI generated product reviews in Dravidian lannguages: DravidianLangTech@NAACL 2025, where it achieved macro f1-scores of 0.93 on Tamil and 0.915 on Malayalam.**


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

