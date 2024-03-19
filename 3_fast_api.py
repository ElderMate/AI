from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import uvicorn
import json
import numpy as np

# 토크나이저 로드
with open('./tokenizer/LSTM_tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

# 모델
model_path = './model/LSTM_model.h5'
model = load_model(model_path)

index_to_label = {
    0: '납부 예정',
    1: '결제 거절',
    2: '결제 승인',
    3: '자동 이체',
    4: '미납',
    5: '결제 취소',
    6: '계좌 개설'
}

def get_label_from_index(index):
    return index_to_label.get(index, "Unknown")

def preprocess_text(text):
    # 텍스트를 토큰화하고 패딩 적용
    max_len = 50
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    return padded_sequence

app = FastAPI()

class TextItem(BaseModel):
    text: str

@app.post("/predict/")
async def predict(item: TextItem):
    try:
        preprocessed_text = preprocess_text(item.text)
        prediction = model.predict(preprocessed_text)[0]
        predicted_index = np.argmax(prediction)
        return {"class": get_label_from_index(predicted_index)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
