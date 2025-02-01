import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any

class CulturalEncoder(nn.Module):
    def __init__(self, lang_config: Dict[str, Any]):
        super().__init__()
        self.cultural_encoder = AutoModel.from_pretrained(
            lang_config['cultural_model_path']
        )
        self.cultural_tokenizer = AutoTokenizer.from_pretrained(
            lang_config['cultural_model_path']
        )
        
        self.context_projection = nn.Sequential(
            nn.Linear(self.cultural_encoder.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, lang_config['embedding_dim'])
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cultural_outputs = self.cultural_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cultural_features = cultural_outputs.last_hidden_state[:, 0, :]
        return self.context_projection(cultural_features)

class SyntacticEncoder(nn.Module):
    def __init__(self, lang_config: Dict[str, Any]):
        super().__init__()
        self.syntactic_attention = nn.MultiheadAttention(
            embed_dim=lang_config['embedding_dim'],
            num_heads=lang_config['num_attention_heads'],
            dropout=0.3
        )
        
        self.complexity_layer = nn.Sequential(
            nn.Linear(lang_config['embedding_dim'], 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, lang_config['embedding_dim'])
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        attended_features, _ = self.syntactic_attention(
            embeddings, embeddings, embeddings
        )
        return self.complexity_layer(attended_features.mean(dim=1))

class SemanticAnalyzer(nn.Module):
    def __init__(self, lang_config: Dict[str, Any]):
        super().__init__()
        self.semantic_encoder = AutoModel.from_pretrained(
            lang_config['semantic_model_path']
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            lang_config['semantic_model_path']
        )
        
        self.coherence_scorer = nn.Sequential(
            nn.Linear(self.semantic_encoder.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, lang_config['embedding_dim'])
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        semantic_outputs = self.semantic_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        semantic_repr = semantic_outputs.last_hidden_state
        pooled_repr = semantic_repr.mean(dim=1)
        return self.coherence_scorer(pooled_repr)

class DravidianMLFDetector(nn.Module):
    def __init__(self, lang_config: Dict[str, Any]):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
        
        self.cultural_encoder = CulturalEncoder(lang_config)
        self.syntactic_encoder = SyntacticEncoder(lang_config)
        self.semantic_analyzer = SemanticAnalyzer(lang_config)
        
        combined_dim = lang_config['embedding_dim'] * 3
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cultural_features = self.cultural_encoder(input_ids, attention_mask)
        base_features = self.cultural_encoder.cultural_encoder(input_ids, attention_mask).last_hidden_state
        syntactic_features = self.syntactic_encoder(base_features)
        semantic_features = self.semantic_analyzer(input_ids, attention_mask)
        
        combined_features = torch.cat([
            cultural_features,
            syntactic_features,
            semantic_features
        ], dim=1)
        
        return F.log_softmax(self.classifier(combined_features), dim=1)