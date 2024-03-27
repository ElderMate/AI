from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import uvicorn


# 모델과 토크나이저 로드
model_name = "./model/kobert"
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

id2tag = {0: 'O', 1: 'METHOD', 2: 'LOCATION', 3: 'TIME', 4: 'COST'}


app = FastAPI()

class Item(BaseModel):
    message: str

@app.post("/keywords/")
async def create_item(item: Item):
    model.eval()
    inputs = tokenizer(item.message, return_tensors="pt", padding=True, truncation=True, max_length=512,
                       return_offsets_mapping=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    offset_mapping = inputs["offset_mapping"].detach().cpu().numpy()[0]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)

    labels = [id2tag[id] for id in predictions[0].cpu().numpy()]
    extracted_info = {"METHOD": "", "LOCATION": "", "TIME": "", "COST": ""}

    for i, (offset, label) in enumerate(zip(offset_mapping, labels)):
        if label != "O":
            start, end = offset
            extracted_text = item.message[start:end]
            extracted_info[label] += extracted_text + " "

    for key in extracted_info:
        extracted_info[key] = extracted_info[key].strip()

    return {"result": extracted_info}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
