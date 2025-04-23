import torch.nn as nn


class TFIDFTransformerClassifier(nn.Module):
    def __init__(
        self, tfidf_dim=8000, d_model=256, num_classes=3, num_layers=3, nhead=8
    ):
        super().__init__()
        self.input_proj = nn.Linear(tfidf_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)

        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)

        x = x.squeeze(0)
        return self.classifier(x)
