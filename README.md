### 프로젝트 실행 방법
- **Python 버전**: 3.10.11로 가상환경 설정
- **라이브러리 설치**: 터미널 창에 pip install -r requirements.txt 입력
- **NER 모델**: 용량 문제로 github에 업로드가 안되어 구글 드라이브에서 직접 다운받아 각 폴더 위치에 삽입
  - kobert_결제거절:
  - kobert_결제승인:
  - kobert_결제취소:
  - kobert_계좌개설:
  - kobert_납부예정:
  - kobert_미납:
  - kobert_자동이체:
- **실행**: 8_AI_fast_api.py 파일 실행

<br>

### API 명세서
<img width="801" alt="스크린샷 2024-04-03 오후 5 39 19" src="https://github.com/ElderMate/AI/assets/110288718/b0ee903e-d012-4f16-add1-a8abb60584dc">

| class_name         | 의미              | response                                                       |
|--------------------|--------------|----------------------------------------------------------------|
| payment_denial     | 결제 거절         | { "METHOD": "", "LOCATION": "", "TIME": "", "COST": "", "REJECTION_REASON": "" } |
| payment_approval   | 결제 승인         | { "METHOD": "", "LOCATION": "", "TIME": "", "COST": "" }       |
| payment_cancellation | 결제 취소         | { "METHOD": "", "LOCATION": "", "TIME": "", "COST": "" }       |
| account_opening    | 계좌 개설         | { "BANK": "", "TYPE": "" }                                     |
| payment_scheduled  | 납부 예정         | { "PAYEE": "", "COST": "", "TIME": "", "PAYMENT_REASON": "" }  |
| non_payment        | 미납              | { "PAYEE": "", "COST": "", "TIME": "" }                        |
| automatic_transfer | 자동 이체         | { "BANK": "", "COMPANY": "" }                                  |
