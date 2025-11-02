import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import re
import traceback

# ============================
# Text Cleaning
# ============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", '', text)
    text = re.sub(r"\d+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# ============================
# Model Class
# ============================
class IndoBERT_CNN_LSTM(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.conv1 = nn.Conv1d(768, 128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        logits = self.fc(h_n.squeeze(0))
        return logits

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="Deteksi Ujaran Kebencian TikTok", layout="wide")
st.title("💬🔥 Deteksi Ujaran Kebencian pada Komentar TikTok")

input_text = st.text_area("📝 Masukkan Komentar TikTok:", height=200)

if st.button("🔍 Deteksi"):
    try:
        device = torch.device("cpu")

        tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
        bert_model = BertModel.from_pretrained('indobenchmark/indobert-base-p1').to(device)

        model = IndoBERT_CNN_LSTM(bert_model)
        model.load_state_dict(torch.load("model_hatespeech.pt", map_location=device))
        model.eval()

        cleaned = clean_text(input_text)
        st.write("🧹 Teks setelah dibersihkan:")
        st.code(cleaned)

        tokens = tokenizer(cleaned, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        confidence_no_hate = probs[0][0].item()
        confidence_hate = probs[0][1].item()

        if pred == 0:
            st.success(f"✅ Komentar **Tidak Mengandung Kebencian** (Confidence {confidence_no_hate:.2f})")
        else:
            st.error(f"❌ Komentar **Mengandung Ujaran Kebencian** (Confidence {confidence_hate:.2f})")

    except Exception:
        st.error("❌ Terjadi error saat deteksi.")
        st.code(traceback.format_exc())
