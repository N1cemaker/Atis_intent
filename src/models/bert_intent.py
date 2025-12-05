import torch
import torch.nn as nn
from transformers import BertModel

class BERTIntentClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_model="prajjwal1/bert-tiny", dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.fc(self.dropout(cls_emb))
        return logits
