from transformers import BertModel
import torch.nn as nn


class HierarchicalSentimentClassifier(nn.Module):
    def __init__(self, hidden_size=768):
        super(HierarchicalSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-uncased")
        self.dropout = nn.Dropout(0.2)

        self.neutral_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        neutral_logit = self.neutral_head(pooled).squeeze(-1)
        sentiment_logits = self.sentiment_head(pooled)
        return neutral_logit, sentiment_logits
