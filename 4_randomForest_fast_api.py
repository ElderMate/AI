from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

# 랜덤 포레스트 모델과 TF-IDF 벡터라이저 로드
model_path = './model/random_forest_model.pkl'
vectorizer_path = './vectorizer/tfidf_vectorizer.joblib'
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

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
    # 텍스트를 TF-IDF 벡터로 변환
    vectorized_text = vectorizer.transform([text])
    return vectorized_text

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
