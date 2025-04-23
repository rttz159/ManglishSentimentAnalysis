import torch
from transformers import XLMRobertaTokenizer
from preprocessing import Preprocessing
import streamlit as st
from rttz.model_3 import SemanticClassifier

labels = ["Negative", "Neutral", "Positive"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocessing_instance = Preprocessing()


@st.cache_resource
def load_model():
    state_dict = torch.load(
        "data/semantic_classifier_3_2_dict.pth", map_location=device
    )
    model = SemanticClassifier()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@st.cache_resource
def get_tokenizer():
    return XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")


def preprocess_text(text, max_length=128, device="cuda"):
    processed_text = preprocessing_instance.preprocessing_pipeline_roberta(text)
    processed_text = str(processed_text)
    tokenizer = get_tokenizer()
    encoding = tokenizer(
        processed_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
    }


def predict(text):
    inputs = preprocess_text(text, device=device)
    model = load_model()

    with torch.no_grad():
        output = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])

    predicted_label = torch.argmax(output, dim=1).cpu().item()
    return labels[int(predicted_label)]
