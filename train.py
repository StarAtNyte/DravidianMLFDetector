import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm

from models import DravidianMLFDetector
from dataset import MultilingualTextDataset
from config import dravidian_config

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    language: str,
    epochs: int = 10,
    learning_rate: float = 1e-4
) -> float:
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_f1 = 0

    for epoch in tqdm(range(epochs), desc=f"Training {language} Model"):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        f1 = f1_score(val_labels, val_preds, average='macro')
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Validation F1: {f1:.4f}")
        print("Classification Report:")
        print(classification_report(val_labels, val_preds, target_names=['Human', 'AI']))
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f'best_{language}_model.pth')
    
    return best_f1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = DravidianMLFDetector(dravidian_config).to(device)
    tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
    
    # Load and split data
    train_df = pd.read_csv('path_to_training_data.csv')
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = MultilingualTextDataset(train_df, tokenizer)
    val_dataset = MultilingualTextDataset(val_df, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Train model
    best_f1 = train_model(model, train_loader, val_loader, device, "malayalam")
    print(f"Best F1 Score: {best_f1}")

if __name__ == "__main__":
    main()