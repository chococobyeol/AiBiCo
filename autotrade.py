import os
import time
import json
from dotenv import load_dotenv
import pyupbit
from openai import OpenAI
import pandas as pd
import numpy as np
import requests
import sqlite3
from datetime import datetime, timedelta

load_dotenv()

# 전략 파일들 읽기 함수 수정
def read_strategies():
    strategies = ""
    strategy_files = os.listdir('strategies')
    for file in strategy_files:
        if file.endswith('.txt'):
            with open(os.path.join('strategies', file), 'r', encoding='utf-8') as f:
                strategies += f.read() + "\n\n"
    return strategies

# 현재 계정 상태 조회
def get_current_status(upbit):
    krw_balance = upbit.get_balance("KRW")
    btc_balance = upbit.get_balance("BTC")
    btc_price = pyupbit.get_current_price("KRW-BTC")
    
    return {
        "krw_balance": krw_balance,
        "btc_balance": btc_balance,
        "btc_price": btc_price
    }

# 간단한 호가 정보 조회
def get_simplified_orderbook():
    orderbook = pyupbit.get_orderbook("KRW-BTC")
    return {
        "ask_price": orderbook['orderbook_units'][0]['ask_price'],
        "bid_price": orderbook['orderbook_units'][0]['bid_price'],
        "ask_size": orderbook['orderbook_units'][0]['ask_size'],
        "bid_size": orderbook['orderbook_units'][0]['bid_size']
    }

# 차트 데이터 및 기술적 지표 계산
def get_simplified_chart_data():
    df = pyupbit.get_ohlcv("KRW-BTC", count=100, interval="day")
    
    # RSI 계산
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # MACD 계산
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()

    # 볼린저 밴드 계산
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['stddev'] = df['close'].rolling(window=20).std()
    df['upper'] = df['MA20'] + (df['stddev'] * 2)
    df['lower'] = df['MA20'] - (df['stddev'] * 2)

    latest_data = df.iloc[-1].to_dict()
    latest_data.update({
        'rsi': rsi.iloc[-1],
        'macd': macd.iloc[-1],
        'macd_signal': signal.iloc[-1],
        'bb_upper': df['upper'].iloc[-1],
        'bb_lower': df['lower'].iloc[-1]
    })

    return latest_data

# 공포 탐욕 지수 조회
def get_fear_and_greed_index():
    url = "https://api.alternative.me/fng/"
    response = requests.get(url)
    data = response.json()
    return {
        "value": data["data"][0]["value"],
        "classification": data["data"][0]["value_classification"]
    }

# SQLite 데이터베이스 연결 및 테이블 생성
def init_db():
    conn = sqlite3.connect('trading_history.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
    if cursor.fetchone() is None:
        cursor.execute('''
        CREATE TABLE trades
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         timestamp TEXT,
         decision TEXT,
         percentage REAL,
         reason TEXT,
         btc_balance REAL,
         krw_balance REAL,
         btc_avg_buy_price REAL,
         btc_krw_price REAL,
         success INTEGER,
         reflection TEXT)
        ''')
    else:
        cursor.execute("PRAGMA table_info(trades)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'reflection' not in columns:
            cursor.execute('ALTER TABLE trades ADD COLUMN reflection TEXT')
    
    conn.commit()
    return conn

# 거래 데이터 저장 함수 수정
def save_trade(conn, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, success=True, reflection=None):
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute('''
    INSERT INTO trades (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, success, reflection)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, success, reflection))
    conn.commit()

# 최근 거래 데이터 가져오는 함수
def get_recent_trades(conn, limit=10):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,))
    return cursor.fetchall()

# 성과 분석 함수
def analyze_performance(recent_trades, current_price):
    performance = []
    for trade in recent_trades:
        if trade[2] == 'buy':  # decision column
            profit = (current_price - trade[7]) / trade[7] * 100  # (current_price - btc_avg_buy_price) / btc_avg_buy_price * 100
        elif trade[2] == 'sell':
            profit = (trade[8] - current_price) / current_price * 100  # (btc_krw_price - current_price) / current_price * 100
        else:
            profit = 0
        performance.append({
            'decision': trade[2],
            'timestamp': trade[1],
            'profit': profit,
            'reason': trade[4]
        })
    return performance

# 반성 생성 함수
def generate_reflection(performance, strategies):
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"You are an AI trading assistant tasked with analyzing recent trading performance and providing a reflection. Consider the following strategies:\n\n{strategies}\n\nAnalyze the performance data and provide insights on what went well, what could be improved, and how to adjust the strategy for better future performance. Be concise but insightful."
            },
            {
                "role": "user",
                "content": json.dumps(performance)
            }
        ],
        max_tokens=500
    )
    
    return response.choices[0].message.content

# AI 기반 트레이딩 로직
def ai_trading():
    access = os.getenv('UPBIT_ACCESS_KEY')
    secret = os.getenv('UPBIT_SECRET_KEY')
    upbit = pyupbit.Upbit(access, secret)

    conn = init_db()

    current_status = get_current_status(upbit)
    orderbook = get_simplified_orderbook()
    chart_data = get_simplified_chart_data()
    fear_greed_index = get_fear_and_greed_index()

    strategies = read_strategies()
    
    recent_trades = get_recent_trades(conn)
    performance = analyze_performance(recent_trades, current_status['btc_price'])
    reflection = generate_reflection(performance, strategies)

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"You are an expert in Bitcoin investing. Analyze the provided data including RSI, MACD, Bollinger Bands, and the Fear and Greed Index. Decide whether to buy, sell, or hold. Consider market sentiment from the Fear and Greed Index in your decision. Use the following investment strategies:\n\n{strategies}\n\nAlso, consider the following reflection on recent performance:\n\n{reflection}"
            },
            {
                "role": "user",
                "content": json.dumps({
                    "current_status": current_status,
                    "orderbook": orderbook,
                    "chart_data": chart_data,
                    "fear_greed_index": fear_greed_index
                })
            }
        ],
        functions=[
            {
                "name": "make_trading_decision",
                "description": "Make a trading decision based on the analyzed data and given strategy",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "decision": {
                            "type": "string",
                            "enum": ["buy", "sell", "hold"],
                            "description": "The trading decision"
                        },
                        "reason": {
                            "type": "string",
                            "description": "The reason for the trading decision, referencing the strategy and recent performance reflection"
                        }
                    },
                    "required": ["decision", "reason"]
                }
            }
        ],
        function_call={"name": "make_trading_decision"}
    )

    result = response.choices[0].message.function_call.arguments
    data = json.loads(result)
    decision = data['decision']
    reason = data['reason']

    print("\n현재 상태:")
    print(f"KRW 잔고: {current_status['krw_balance']} 원")
    print(f"BTC 잔고: {current_status['btc_balance']} BTC")
    print(f"BTC 현재 가격: {current_status['btc_price']} 원")
    print(f"\nAI 결정: {decision}")
    print(f"결정 이유: {reason}")
    print(f"\n반성: {reflection}")

    if decision == 'buy':
        krw_balance = current_status['krw_balance']
        if krw_balance > 5000:
            buy_amount = krw_balance * 0.9995
            try:
                upbit.buy_market_order("KRW-BTC", buy_amount)
                print(f"\n매수 실행: {buy_amount} 원어치의 BTC 구매")
                print(f"매수 이유: {reason}")
                # 거래 데이터 저장
                btc_balance = upbit.get_balance("BTC")
                krw_balance = upbit.get_balance("KRW")
                btc_avg_buy_price = upbit.get_avg_buy_price("BTC")
                percentage = (buy_amount / current_status['krw_balance']) * 100
                success = True
                save_trade(conn, 'buy', percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, current_status['btc_price'], success, reflection)
            except Exception as e:
                print(f"\n매수 실패: {str(e)}")
                success = False
                save_trade(conn, 'buy', 0, f"매수 실패: {str(e)}", current_status['btc_balance'], krw_balance, current_status['btc_price'], current_status['btc_price'], success, reflection)
        else:
            failure_reason = f"매수 실패: 원화 잔고({krw_balance} 원)가 5000원 미만입니다."
            print(f"\n{failure_reason}")
            success = False
            save_trade(conn, 'buy', 0, failure_reason, current_status['btc_balance'], krw_balance, current_status['btc_price'], current_status['btc_price'], success, reflection)
    elif decision == 'sell':
        btc_balance = current_status['btc_balance']
        btc_price = current_status['btc_price']
        btc_value = btc_balance * btc_price
        if btc_balance > 0 and btc_value > 5000:
            try:
                upbit.sell_market_order("KRW-BTC", btc_balance)
                print(f"\n매도 실행: {btc_balance} BTC 판매 (약 {btc_value} 원)")
                print(f"매도 이유: {reason}")
                # 거래 데이터 저장
                krw_balance = upbit.get_balance("KRW")
                percentage = 100  # 전량 매도
                success = True
                save_trade(conn, 'sell', percentage, reason, 0, krw_balance, 0, btc_price, success, reflection)
            except Exception as e:
                print(f"\n매도 실패: {str(e)}")
                success = False
                save_trade(conn, 'sell', 0, f"매도 실패: {str(e)}", btc_balance, current_status['krw_balance'], current_status['btc_price'], btc_price, success, reflection)
        else:
            failure_reason = f"매도 실패: 비트코인 잔고({btc_balance} BTC, 약 {btc_value} 원)가 0이거나, 원화 가치가 5000원 미만입니다."
            print(f"\n{failure_reason}")
            success = False
            save_trade(conn, 'sell', 0, failure_reason, btc_balance, current_status['krw_balance'], current_status['btc_price'], btc_price, success, reflection)
    elif decision == 'hold':
        print("\n홀딩: 현재 상태를 유지합니다.")
        print(f"홀딩 이유: {reason}")
        # 홀딩 데이터 저장
        btc_balance = current_status['btc_balance']
        krw_balance = current_status['krw_balance']
        btc_avg_buy_price = upbit.get_avg_buy_price("BTC")
        success = True
        save_trade(conn, 'hold', 0, reason, btc_balance, krw_balance, btc_avg_buy_price, current_status['btc_price'], success, reflection)

    conn.close()

if __name__ == "__main__":
    try:
        ai_trading()
    except Exception as e:
        print(f"오류 발생: {e}")
