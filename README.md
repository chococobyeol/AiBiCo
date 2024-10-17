# AiBiCo - AI Bitcoin Trading Bot

AiBiCo는 인공지능을 활용한 비트코인 자동 거래 봇입니다.

## 주요 기능

- OpenAI의 GPT 모델을 활용한 거래 결정
- 업비트 API를 통한 실시간 거래
- 기술적 지표 (RSI, MACD, 볼린저 밴드) 분석
- 공포 탐욕 지수를 활용한 시장 심리 분석
- 거래 내역 및 성과 시각화 대시보드

## 설치 방법

1. 레포지토리를 클론합니다:
   ```
   git clone https://github.com/chococobyeol/AiBiCo.git
   ```

2. 필요한 패키지를 설치합니다:
   ```
   pip install -r requirements.txt
   ```

3. `.env` 파일을 생성하고 필요한 API 키를 설정합니다.

## 사용 방법

1. 자동 거래 봇 실행:
   ```
   python autotrade.py
   ```

2. 대시보드 실행:
   ```
   streamlit run trading_dashboard.py
   ```

## 주의사항

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다. 실제 거래에 사용할 경우 발생하는 손실에 대해 책임지지 않습니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
