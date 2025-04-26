# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI()

# Load model and tokenizer
model_path = "./bert_department_model" # .pt uzantısını kaldırdım ve klasör yolunu belirttim
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Önce temel modeli yükle
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)  # Temel modeli 6 sınıf için yapılandır

# Eğer kaydedilmiş ağırlıklar varsa yükle
try:
    model.load_state_dict(torch.load("bert_department_model.pt", map_location=device))
    print("Model ağırlıkları başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Label mapping (manuel olarak girmen gerek)
label_mapping = {
    0: "Trend",
    1: "Jackets",
    2: "Intimate",
    3: "Bottoms",
    4: "Dresses",
    5: "Tops"
}

class ReviewRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(review: ReviewRequest):
    inputs = tokenizer(review.text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return {"department": label_mapping.get(prediction, "Unknown")}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)