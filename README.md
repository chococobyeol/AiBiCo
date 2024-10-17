# AiBiCo - AI Bitcoin Trading Bot

AiBiCo는 인공지능을 활용한 비트코인 자동 거래 봇입니다.

## 프로젝트 구조

- `autotrade.py`: 메인 거래 봇 스크립트
- `trading_dashboard.py`: Streamlit 기반 대시보드
- `next_trade_time.json`: 다음 거래 시간 정보 저장 파일
- `strategies/`: 거래 전략 텍스트 파일 디렉토리
- `requirements.txt`: 필요한 Python 패키지 목록
- `.env`: 환경 변수 설정 파일 (API 키 등)

## 주요 기능

1. 자동 비트코인 거래
   - OpenAI의 GPT 모델을 활용한 거래 결정
   - 업비트 API를 통한 실시간 거래 실행
   - 동적 거래 간격 조정

2. 실시간 대시보드
   - 거래 내역 및 성과 표시
   - 수익률 그래프
   - 필터링 기능

3. 리스크 관리
   - 변동성 및 거래량 기반 거래 간격 조정
   - 시장 조건에 따른 거래 실행

4. 데이터 수집 및 분석
   - 기술적 지표 계산 (RSI, MACD, 볼린저 밴드)
   - 뉴스 데이터 수집 (MediaStack API, CryptoCompare API)
   - 공포/탐욕 지수 분석

## 알고리즘 상세 설명

1. 데이터 수집
   - 업비트 API를 통해 비트코인 가격, 거래량, 호가 정보 수집
   - MediaStack API와 CryptoCompare API를 사용하여 암호화폐 관련 뉴스 수집
   - Alternative.me API를 통해 공포/탐욕 지수 조회

2. 기술적 지표 계산
   - RSI (Relative Strength Index) 계산
   - MACD (Moving Average Convergence Divergence) 계산
   - 볼린저 밴드 계산

3. AI 모델을 통한 거래 결정
   - 수집된 데이터와 계산된 지표를 OpenAI의 GPT 모델에 입력
   - 모델은 '매수', '매도', '홀드' 중 하나의 결정과 거래 비율을 출력

4. 거래 실행
   - AI 모델의 결정에 따라 업비트 API를 통해 실제 거래 실행
   - 거래 실행 전 잔고 확인 및 최소 주문 금액 검증

5. 동적 거래 간격 조정
   - 시장 변동성과 거래량을 기반으로 다음 거래 시간 계산
   - 기본 4시간 간격에서 시장 상황에 따라 1시간에서 8시간 사이로 조정

6. 데이터베이스 저장
   - 모든 거래 내역과 성과를 SQLite 데이터베이스에 저장
   - 거래 반성 및 누적 반성 내용도 함께 저장

7. 대시보드 업데이트
   - Streamlit을 사용하여 실시간으로 거래 내역, 성과, 그래프 등을 표시

## 설치 방법

1. 레포지토리를 클론합니다:
   ```
   git clone https://github.com/chococobyeol/AiBiCo.git
   cd AiBiCo
   ```

2. 가상 환경을 생성하고 활성화합니다:
   ```
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. 필요한 패키지를 설치합니다:
   ```
   pip install -r requirements.txt
   ```

4. `.env` 파일을 생성하고 필요한 API 키를 설정합니다:


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

- 이 프로젝트는 실제 돈을 사용하여 거래합니다. 주의해서 사용하세요.
- API 키를 안전하게 보관하고, 절대 공개하지 마세요.
- 거래 전략을 충분히 테스트한 후 실제 거래에 사용하세요.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
