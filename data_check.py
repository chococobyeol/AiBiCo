import os
from dotenv import load_dotenv
import pyupbit
import json
import pandas as pd
from datetime import datetime

load_dotenv()

def get_current_status(upbit):
    balances = upbit.get_balances()
    krw_balance = upbit.get_balance("KRW")
    btc_balance = upbit.get_balance("BTC")
    btc_price = pyupbit.get_current_price("KRW-BTC")
    
    return {
        "balances": balances,
        "krw_balance": krw_balance,
        "btc_balance": btc_balance,
        "btc_price": btc_price
    }

def get_orderbook():
    return pyupbit.get_orderbook("KRW-BTC")

# json_default 함수를 수정합니다
def json_default(obj):
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

def get_chart_data():
    daily_ohlcv = pyupbit.get_ohlcv("KRW-BTC", count=30, interval="day")
    hourly_ohlcv = pyupbit.get_ohlcv("KRW-BTC", count=24, interval="minute60")
    
    return {
        "daily_ohlcv": daily_ohlcv.reset_index().to_dict(orient='records'),
        "hourly_ohlcv": hourly_ohlcv.reset_index().to_dict(orient='records')
    }

def check_data():
    access = os.getenv('UPBIT_ACCESS_KEY')
    secret = os.getenv('UPBIT_SECRET_KEY')
    upbit = pyupbit.Upbit(access, secret)

    # 1. 현재 투자 상태 가져오기
    current_status = get_current_status(upbit)
    print("현재 투자 상태:")
    print(json.dumps(current_status, indent=2, ensure_ascii=False))
    print("\n" + "="*50 + "\n")

    # 2. 오더북 가져오기
    orderbook = get_orderbook()
    print("오더북 데이터:")
    print(json.dumps(orderbook, indent=2, ensure_ascii=False))
    print("\n" + "="*50 + "\n")

    # 3. 차트 데이터 가져오기
    chart_data = get_chart_data()
    print("차트 데이터:")
    print("일봉 데이터 (30일):")
    print(json.dumps(chart_data['daily_ohlcv'], indent=2, ensure_ascii=False, default=json_default)[:1000] + "...")
    print("\n시간봉 데이터 (24시간):")
    print(json.dumps(chart_data['hourly_ohlcv'], indent=2, ensure_ascii=False, default=json_default)[:1000] + "...")

if __name__ == "__main__":
    check_data()
