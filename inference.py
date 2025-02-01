import torch
from transformers import AutoTokenizer
from typing import List
from models import DravidianMLFDetector
from config import dravidian_config

def inference(
    model: torch.nn.Module,
    texts: List[str],
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int = 16
) -> List[str]:
    model.eval()
    predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=encodings['input_ids'],
                attention_mask=encodings['attention_mask']
            )
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(['AI' if p == 1 else 'Human' for p in preds.cpu().numpy()])
    
    return predictions

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = DravidianMLFDetector(dravidian_config).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load('best_malayalam_model.pth'))
    tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
    
    # Example inference
    test_texts = ["Sample text 1", "Sample text 2"]
    predictions = inference(model, test_texts, tokenizer, device)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()