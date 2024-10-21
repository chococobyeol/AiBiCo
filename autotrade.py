# autotrade.py

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
import math
import logging
import fcntl
import sys
import signal
import psutil
from scipy import optimize
import logging.handlers
import ta
from ta.utils import dropna

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# 환경 변수에서 API 키 가져오기
MEDIASTACK_API_KEY = os.getenv('MEDIASTACK_API_KEY')
CRYPTOCOMPARE_API_KEY = os.getenv('CRYPTOCOMPARE_API_KEY')

# 뉴스 데이터 캐싱
news_cache = {'mediastack': None, 'cryptocompare': None, 'last_update': None}

# 최소 거래 간격(초 단위)
MIN_TRADE_INTERVAL = 600  # 10분

# 전략 파일 읽기
def read_strategies():
    strategies = ""
    strategy_files = os.listdir('strategies')
    for file in strategy_files:
        if file.endswith('.txt'):
            with open(os.path.join('strategies', file), 'r', encoding='utf-8') as f:
                strategies += f.read()[:500] + "\n\n"  # 각 전략 파일의 처음 500자만 읽기
    return strategies

# 현재 계좌 상태 가져오기
def get_current_status(upbit):
    krw_balance = upbit.get_balance("KRW")
    btc_balance = upbit.get_balance("KRW-BTC")
    btc_price = pyupbit.get_current_price("KRW-BTC")

    return {
        "krw_balance": float(krw_balance),
        "btc_balance": float(btc_balance),
        "btc_price": float(btc_price)
    }

# 간단한 주문서 가져오기
def get_simplified_orderbook():
    try:
        orderbook = pyupbit.get_orderbook("KRW-BTC")
        logging.info(f"Received orderbook: {orderbook}")
        
        if orderbook and isinstance(orderbook, dict) and 'orderbook_units' in orderbook and len(orderbook['orderbook_units']) > 0:
            first_unit = orderbook['orderbook_units'][0]
            return {
                "ask_price": first_unit['ask_price'],
                "bid_price": first_unit['bid_price'],
                "ask_size": first_unit['ask_size'],
                "bid_size": first_unit['bid_size']
            }
        else:
            logging.warning("Orderbook structure is not as expected")
            return {
                "ask_price": 0,
                "bid_price": 0,
                "ask_size": 0,
                "bid_size": 0
            }
    except Exception as e:
        logging.error(f"Error in get_simplified_orderbook: {str(e)}", exc_info=True)
        return {
            "ask_price": 0,
            "bid_price": 0,
            "ask_size": 0,
            "bid_size": 0
        }

# 기술 지표가 포함된 간단한 차트 데이터 가져오기
def get_simplified_chart_data():
    df_daily = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=15)  # 15일로 변경
    df_daily = dropna(df_daily)
    df_daily = add_indicators(df_daily)
    
    df_10min = pyupbit.get_ohlcv("KRW-BTC", interval="minute10", count=72)  # 12 hours of 10-minute data
    df_10min = dropna(df_10min)
    df_10min = add_indicators(df_10min)

    def df_to_dict(df):
        df_dict = df.to_dict(orient='index')
        return {k.isoformat(): v for k, v in df_dict.items()}

    return {
        'daily': df_to_dict(df_daily),
        '10min': df_to_dict(df_10min)
    }

# 공포와 탐욕 지수 가져오기
def get_fear_and_greed_index():
    url = "https://api.alternative.me/fng/"
    try:
        response = requests.get(url)
        data = response.json()
        return {
            "value": data["data"][0]["value"],
            "classification": data["data"][0]["value_classification"]
        }
    except Exception as e:
        logging.error(f"Error fetching Fear and Greed Index: {e}")
        return {"value": None, "classification": None}

# SQLite 데이터베이스 초기화
def init_db():
    conn = sqlite3.connect('trading_history.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades
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
     reflection TEXT,
     total_assets_krw REAL,
     cumulative_reflection TEXT,
     short_term_necessity REAL)
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reflection_summary
    (id INTEGER PRIMARY KEY AUTOINCREMENT,
     summary TEXT,
     timestamp TEXT)
    ''')
    conn.commit()
    return conn

# 거래 데이터 저장
def save_trade(conn, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, success=True, reflection=None, cumulative_reflection=None, short_term_necessity=None):
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()

    # 총 자산(KRW) 계산
    total_assets_krw = krw_balance + (btc_balance * btc_krw_price)

    try:
        cursor.execute('''
        INSERT INTO trades (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price,
                            btc_krw_price, success, reflection, total_assets_krw, cumulative_reflection, short_term_necessity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price,
              int(success), reflection, total_assets_krw, cumulative_reflection, short_term_necessity))
        conn.commit()
        logging.info(f"거래가 성��적으로 저장되었습니다: {decision}, {success}")
    except sqlite3.Error as e:
        logging.error(f"거래 저장 중 오류 발생: {e}")
        conn.rollback()

# 최근 거래 가져오기 함수 수정
def get_recent_trades(conn, days=7, limit=5):  # limit를 5로 변경
    cursor = conn.cursor()
    one_week_ago = (datetime.now() - timedelta(days=days)).isoformat()

    if limit:
        cursor.execute("SELECT * FROM trades WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT ?", (one_week_ago, limit))
    else:
        cursor.execute("SELECT * FROM trades WHERE timestamp >= ? ORDER BY timestamp DESC", (one_week_ago,))

    columns = [column[0] for column in cursor.description]
    trades = [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    # timestamp를 문자열로 변환
    for trade in trades:
        trade['timestamp'] = trade['timestamp']  # 이미 문자열 형태로 저장되어 있음
    
    return trades

# 성과 분석 함수 수정
def analyze_performance(trades, current_price, upbit):
    performance = []
    initial_assets = None
    previous_assets = None

    for trade in trades:
        current_assets = upbit.get_amount('ALL')
        
        if initial_assets is None:
            initial_assets = current_assets
            previous_assets = current_assets
        
        profit_percentage = ((current_assets - previous_assets) / previous_assets) * 100 if previous_assets > 0 else 0
        total_profit_percentage = ((current_assets - initial_assets) / initial_assets) * 100 if initial_assets > 0 else 0
        
        performance.append({
            'decision': trade['decision'],
            'timestamp': trade['timestamp'],
            'profit_percentage': profit_percentage,
            'total_profit_percentage': total_profit_percentage,
            'reason': trade['reason'],
            'success': trade['success'] == 1,
            'total_assets': current_assets
        })
        
        previous_assets = current_assets

    final_assets = upbit.get_amount('ALL')
    final_total_profit_percentage = ((final_assets - initial_assets) / initial_assets) * 100 if initial_assets > 0 else 0

    avg_profit_percentage = sum(p['profit_percentage'] for p in performance) / len(performance) if performance else 0

    return performance, avg_profit_percentage, final_total_profit_percentage, final_assets

# 반성 생성
def generate_reflection(performance, strategies, trades, avg_profit, previous_reflections):
    previous_reflections_text = "\n".join(previous_reflections)

    messages = [
        {
            "role": "system",
            "content": f"You are an AI trading assistant tasked with analyzing trading performance over the past week and providing a reflection. Consider the following strategies:\n\n{strategies}\n\nAnalyze the performance data and trades, then provide insights on what went well, what could be improved, and how to adjust the strategy for better future performance. Be concise but insightful. Consider the average profit of {avg_profit:.2f}% over the past week. Also, consider these previous reflections:\n\n{previous_reflections_text}"
        },
        {
            "role": "user",
            "content": json.dumps({
                "performance": performance,
                "trades": trades,
                "avg_profit": avg_profit
            })
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()

# 변동성 계산
def calculate_volatility(df, window=14):
    df['daily_return'] = df['close'].pct_change()
    volatility = df['daily_return'].rolling(window=window).std() * (252 ** 0.5)
    return volatility.iloc[-1]

def get_volatility_data():
    df = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=30)
    current_volatility = calculate_volatility(df)
    avg_volatility = df['daily_return'].std() * (252 ** 0.5)

    return {
        "current_volatility": current_volatility,
        "average_volatility": avg_volatility
    }

# MediaStack에서 뉴스 가져오기
def get_mediastack_news():
    global news_cache
    current_time = datetime.now()
    if news_cache['mediastack'] is None or (current_time - news_cache['last_update']).total_seconds() > 5400:
        url = f"http://api.mediastack.com/v1/news?access_key={MEDIASTACK_API_KEY}&keywords=bitcoin,cryptocurrency&languages=en&limit=10"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            news = response.json()['data']
            news_cache['mediastack'] = [{'title': item['title'], 'description': item['description']} for item in news]
            news_cache['last_update'] = current_time
        except requests.RequestException as e:
            logging.error(f"MediaStack API error: {str(e)}")
            return []
    return news_cache['mediastack']

# CryptoCompare에서 뉴스 가져오기
def get_cryptocompare_news():
    global news_cache
    current_time = datetime.now()
    if news_cache['cryptocompare'] is None or (current_time - news_cache['last_update']).total_seconds() > 1800:
        url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={CRYPTOCOMPARE_API_KEY}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            news = response.json()['Data']
            news_cache['cryptocompare'] = [{'title': item['title'], 'body': item['body']} for item in news[:10]]
            news_cache['last_update'] = current_time
        except requests.RequestException as e:
            logging.error(f"CryptoCompare API error: {str(e)}")
            return []
    return news_cache['cryptocompare']

def get_news():
    mediastack_news = get_mediastack_news()
    cryptocompare_news = get_cryptocompare_news()

    # 뉴스 데이터 정리
    cleaned_news = []
    for news in mediastack_news + cryptocompare_news:
        cleaned_news.append({
            'title': news.get('title', '')[:100],
            'description': news.get('description', '')[:200] if 'description' in news else news.get('body', '')[:200]
        })

    return cleaned_news[:3]  # 뉴스 헤드라인 수를 3개로 제한

# 반성 요약 가져오기
def get_reflection_summary(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT summary FROM reflection_summary ORDER BY id DESC LIMIT 1")
    result = cursor.fetchone()
    return result[0] if result else None

def update_reflection_summary(conn, new_reflection, previous_summary):
    if previous_summary:
        prompt = f"Previous summary: {previous_summary}\n\nNew reflection to incorporate: {new_reflection}\n\nCreate an updated summary that incorporates the new reflection into the previous summary. The summary should maintain key insights from the previous summary while adding new insights from the latest reflection. Keep the summary concise but insightful."
    else:
        prompt = f"Create a summary of the following reflection: {new_reflection}"

    messages = [
        {"role": "system", "content": "You are an AI assistant tasked with summarizing trading reflections."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300
    )

    updated_summary = response.choices[0].message.content.strip()

    cursor = conn.cursor()
    cursor.execute("INSERT INTO reflection_summary (summary, timestamp) VALUES (?, ?)", (updated_summary, datetime.now().isoformat()))
    conn.commit()

    return updated_summary

def check_balance_for_trade(upbit, decision, percentage):
    current_status = get_current_status(upbit)
    if decision == 'buy':
        available_krw = current_status['krw_balance']
        trade_amount = available_krw * (percentage / 100)
        if trade_amount < 5000:  # 최소 주문 금액
            return False, "Insufficient KRW balance for buy order"
    elif decision == 'sell':
        available_btc = current_status['btc_balance']
        trade_amount = available_btc * (percentage / 100)
        if trade_amount * current_status['btc_price'] < 5000:  # 최소 주문 금액
            return False, "Insufficient BTC balance for sell order"
    return True, ""

# add_indicators 함수 수정
def add_indicators(df):
    # 볼린저 밴드
    indicator_bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    
    # RSI (이미 있음)
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    
    # MACD (이미 있음)
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # 이동평균선 (단기, 장기)
    df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()

    # Stochastic Oscillator 추가
    stoch = ta.momentum.StochasticOscillator(
        high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # Average True Range (ATR) 추가
    df['atr'] = ta.volatility.AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()

    # On-Balance Volume (OBV) 추가
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(
        close=df['close'], volume=df['volume']).on_balance_volume()

    return df

def convert_to_serializable(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return str(obj)  # 다른 모든 타입을 자로 변환

def prepare_data_for_api(data):
    return json.loads(json.dumps(data, default=convert_to_serializable))

# ai_trading 함
def ai_trading():
    logging.info("Starting ai_trading function")

    try:
        upbit = pyupbit.Upbit(os.getenv('UPBIT_ACCESS_KEY'), os.getenv('UPBIT_SECRET_KEY'))
        current_status = get_current_status(upbit)
        try:
            orderbook = get_simplified_orderbook()
            logging.info(f"Simplified orderbook: {orderbook}")
        except Exception as e:
            logging.error(f"Error getting simplified orderbook: {str(e)}", exc_info=True)
            orderbook = {
                "ask_price": 0,
                "bid_price": 0,
                "ask_size": 0,
                "bid_size": 0
            }
        chart_data = get_simplified_chart_data()
        fear_greed_index = get_fear_and_greed_index()
        volatility_data = get_volatility_data()
        news = get_news()

        conn = init_db()

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades")
            trade_count = cursor.fetchone()[0]

            if trade_count == 0:
                logging.info("No trades found. Creating initial trade data.")
                save_trade(conn,
                           'initial',
                           0,
                           "Initial state",
                           current_status['btc_balance'],
                           current_status['krw_balance'],
                           upbit.get_avg_buy_price("KRW-BTC") or 0,
                           current_status['btc_price'],
                           success=True,
                           reflection="Initial trade data",
                           cumulative_reflection="Starting trading")

            recent_trades = get_recent_trades(conn, limit=5)
            current_price = pyupbit.get_current_price("KRW-BTC")
            performance, avg_profit, total_profit_percentage, final_assets = analyze_performance(recent_trades, current_price, upbit)
            strategies = read_strategies()
            previous_summary = get_reflection_summary(conn)

            if len(recent_trades) >= 5:
                last_five_trades = recent_trades[:5]
                reflection = generate_reflection(performance, strategies, last_five_trades, avg_profit, [])
            else:
                reflection = generate_reflection(performance, strategies, recent_trades, avg_profit, [])

            updated_summary = update_reflection_summary(conn, reflection, previous_summary)

            current_btc_balance = current_status['btc_balance']

            system_message = f"""You are an AI trading assistant. Analyze the given market data and make a trading decision.
    Consider the following reflection summary on recent performance:\n\n{updated_summary}
    Current BTC balance: {current_btc_balance}
    IMPORTANT: If the current BTC balance is 0, do not make a 'sell' decision.

    Additionally, evaluate the necessity for short-term trading based on current market conditions.
    The trading interval can be adjusted between 10 minutes (for very short-term trading) and 8 hours (for longer-term trading).
    Provide a short-term trading necessity score from 0.00 to 1.00, where:
    0.00: No need for short-term trading, prefer longer intervals (closer to 8 hours)
    1.00: High necessity for short-term trading, prefer shorter intervals (closer to 10 minutes)
    Base this score on market volatility, recent news impact, and potential short-term opportunities.
    Display the score with two decimal places (e.g., 0.75)."""

            # 모든 데이터를 JSON 직렬화 가능한 형태로 변환
            current_status_serializable = prepare_data_for_api(current_status)
            orderbook_serializable = prepare_data_for_api(orderbook)
            chart_data_serializable = chart_data  # 이미 직렬화 가능한 형태로 변환되어 있음
            fear_greed_index_serializable = prepare_data_for_api(fear_greed_index)
            volatility_data_serializable = prepare_data_for_api(volatility_data)
            news_serializable = prepare_data_for_api(news)
            performance_serializable = prepare_data_for_api(performance)
            recent_trades_serializable = prepare_data_for_api(recent_trades)

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"""
Current investment status: {json.dumps(current_status_serializable)}
Orderbook: {json.dumps(orderbook_serializable)}
Daily OHLCV with indicators (recent 15 days): {json.dumps(chart_data_serializable['daily'])}
10-minute OHLCV with indicators (recent 12 hours): {json.dumps(chart_data_serializable['10min'])}
Recent news headlines: {json.dumps(news_serializable)}
Fear and Greed Index: {json.dumps(fear_greed_index_serializable)}
Volatility data: {json.dumps(volatility_data_serializable)}
Strategies: {strategies}
Recent trades (last 5): {json.dumps(recent_trades_serializable)}
Average profit: {avg_profit}
Reflection: {reflection}

Based on this information, what trading decision should be made? Please provide your decision (buy, sell, or hold), the percentage (0-100), and a detailed reason for your decision. Consider the short-term trends visible in the 10-minute data for more immediate market movements.
                    """}
                ],
                functions=[{
                    "name": "make_trading_decision",
                    "description": "Make a trading decision based on the given market data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "decision": {
                                "type": "string",
                                "enum": ["buy", "sell", "hold"],
                                "description": "The trading decision"
                            },
                            "percentage": {
                                "type": "number",
                                "description": "The percentage of balance to trade"
                            },
                            "reason": {
                                "type": "string",
                                "description": "The reason for the trading decision"
                            },
                            "short_term_necessity": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "The necessity for short-term trading, from 0 to 1"
                            }
                        },
                        "required": ["decision", "percentage", "reason", "short_term_necessity"]
                    }
                }],
                function_call={"name": "make_trading_decision"}
            )

            function_call = response.choices[0].message.function_call
            result = json.loads(function_call.arguments)

        except Exception as e:
            logging.error(f"OpenAI API 오류: {str(e)}")
            raise

        assistant_message = json.dumps(result)
        logging.info(f"AI response: {assistant_message}")

        decision = result['decision']
        percentage = result['percentage']
        reason = result['reason']
        short_term_necessity = result.get('short_term_necessity', 0.5)
        if short_term_necessity is None:
            short_term_necessity = 0.5
        logging.info(f"Short-term trading necessity score: {short_term_necessity}")

        if decision == 'sell' and current_btc_balance == 0:
            decision = 'hold'
            reason = "Changed to hold because current BTC balance is 0"
            logging.info("Decision changed to hold due to zero BTC balance")

        if decision != 'hold':
            balance_ok, message = check_balance_for_trade(upbit, decision, percentage)
            if not balance_ok:
                logging.warning(message)
                save_trade(conn, decision, percentage, reason,
                           current_status['btc_balance'],
                           current_status['krw_balance'],
                           upbit.get_avg_buy_price("KRW-BTC") or 0,
                           current_status['btc_price'],
                           success=False,
                           reflection=message,
                           cumulative_reflection=updated_summary,
                           short_term_necessity=short_term_necessity)
            else:
                if decision == 'buy':
                    trade_amount = current_status['krw_balance'] * (percentage / 100)
                else:  # sell
                    trade_amount = current_status['btc_balance'] * (percentage / 100)
                    trade_amount = round(trade_amount, 8)  # Round to 8 decimal places for BTC

                try:
                    if decision == 'buy':
                        order = upbit.buy_market_order("KRW-BTC", trade_amount)
                    else:  # sell
                        order = upbit.sell_market_order("KRW-BTC", trade_amount)

                    if 'error' in order:
                        raise Exception(f"Order failed: {order['error']['message']}")

                    updated_status = get_current_status(upbit)

                    save_trade(conn,
                               decision,
                               percentage,
                               reason,
                               updated_status['btc_balance'],
                               updated_status['krw_balance'],
                               upbit.get_avg_buy_price("KRW-BTC"),
                               updated_status['btc_price'],
                               success=True,
                               reflection=reflection,
                               cumulative_reflection=updated_summary,
                               short_term_necessity=short_term_necessity)
                    logging.info(f"Trade saved: {decision}, {percentage}%, {reason}")

                    logging.info(f"Trade executed successfully: {decision} {percentage}% of balance. Reason: {reason}")
                except Exception as trade_error:
                    logging.error(f"Trade execution failed: {str(trade_error)}")
                    save_trade(conn,
                               decision,
                               percentage,
                               reason,
                               current_status['btc_balance'],
                               current_status['krw_balance'],
                               upbit.get_avg_buy_price("KRW-BTC"),
                               current_status['btc_price'],
                               success=False,
                               reflection=str(trade_error),
                               cumulative_reflection=updated_summary,
                               short_term_necessity=short_term_necessity)
        else:
            logging.info(f"Decision: Hold. Reason: {reason}")
            save_trade(conn,
                       'hold',
                       0,
                       reason,
                       current_status['btc_balance'],
                       current_status['krw_balance'],
                       upbit.get_avg_buy_price("KRW-BTC"),
                       current_status['btc_price'],
                       success=True,
                       reflection=reflection,
                       cumulative_reflection=updated_summary,
                       short_term_necessity=short_term_necessity)
            logging.info(f"Hold decision saved: {reason}")

        return {'short_term_necessity': short_term_necessity}

    except Exception as e:
        logging.error(f"Error in ai_trading: {str(e)}")
        save_trade(conn,
                   'error',
                   0,
                   str(e),
                   current_status['btc_balance'],
                   current_status['krw_balance'],
                   upbit.get_avg_buy_price("KRW-BTC") or 0,
                   current_status['btc_price'],
                   success=False,
                   reflection="Error in ai_trading function",
                   cumulative_reflection="Error occurred during trading",
                   short_term_necessity=None)
        logging.info("Error trade saved")
        return {'short_term_necessity': None}
    finally:
        if 'conn' in locals():
            conn.close()
        logging.info("Finished ai_trading function")

# Check market volatility
def check_market_volatility():
    df = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=24)
    volatility = (df['high'] - df['low']).mean() / df['open'].mean()
    return volatility

# Check trading volume
def check_trading_volume():
    df = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=24)
    avg_volume = df['volume'].mean()
    return avg_volume

# Updated calculate_trading_interval function
def calculate_trading_interval(volatility, volume, short_term_necessity):
    # Interval range in seconds
    min_interval = 600    # 10 minutes
    max_interval = 28800  # 8 hours

    # Ensure short_term_necessity is not None
    if short_term_necessity is None:
        short_term_necessity = 0.5

    # Normalize volatility and volume to a 0-1 scale
    normalized_volatility = min(volatility / 0.01, 1)
    normalized_volume = min(volume / 150, 1)

    # Assign weights to factors
    combined_factor = (0.7 * short_term_necessity) + (0.15 * normalized_volatility) + (0.15 * normalized_volume)

    # Linearly adjust the interval based on the combined_factor
    interval = max_interval - (combined_factor * (max_interval - min_interval))

    # Ensure the interval is between min_interval and max_interval
    interval = max(min(interval, max_interval), min_interval)

    return interval

def check_market_conditions():
    volatility = check_market_volatility()
    volume = check_trading_volume()

    if volatility > 0.01 or volume > 150:
        return True
    return False

def save_next_trade_time(next_trade_time):
    with open('next_trade_time.json', 'w') as f:
        json.dump({"next_trade_time": next_trade_time.isoformat()}, f)

def load_next_trade_time():
    try:
        with open('next_trade_time.json', 'r') as f:
            data = json.load(f)
            return datetime.fromisoformat(data['next_trade_time'])
    except (FileNotFoundError, KeyError, ValueError):
        return None

def save_last_trade_time(last_trade_time):
    with open('last_trade_time.json', 'w') as f:
        json.dump({"last_trade_time": last_trade_time.isoformat()}, f)

def load_last_trade_time():
    try:
        with open('last_trade_time.json', 'r') as f:
            data = json.load(f)
            return datetime.fromisoformat(data['last_trade_time'])
    except (FileNotFoundError, KeyError, ValueError):
        return None

def acquire_lock(lockfile):
    try:
        fd = open(lockfile, 'w')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fd.write(str(os.getpid()))
        fd.flush()
        return fd
    except IOError:
        return None

def release_lock(fd):
    if fd:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()

def terminate_process(pid):
    try:
        process = psutil.Process(pid)
        process.terminate()
        process.wait(timeout=10)
    except psutil.NoSuchProcess:
        logging.info(f"No process found with PID {pid}")
    except psutil.TimeoutExpired:
        logging.warning(f"Process {pid} did not terminate. Forcing kill...")
        process.kill()
    except Exception as e:
        logging.error(f"Error terminating process {pid}: {e}")

def analyze_historical_conditions():
    df = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=1000)
    volatilities = (df['high'] - df['low']) / df['open']
    volumes = df['volume']

    analysis_logger = logging.getLogger('analysis')
    analysis_logger.setLevel(logging.INFO)
    handler = logging.handlers.RotatingFileHandler(
        'historical_analysis.log', maxBytes=10*1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    analysis_logger.addHandler(handler)

    analysis_logger.info(f"Average volatility: {volatilities.mean()}")
    analysis_logger.info(f"95th percentile volatility: {volatilities.quantile(0.95)}")
    analysis_logger.info(f"Average volume: {volumes.mean()}")
    analysis_logger.info(f"95th percentile volume: {volumes.quantile(0.95)}")

if __name__ == "__main__":
    lockfile = '/tmp/autotrade.lock'
    lock_fd = None

    try:
        lock_fd = acquire_lock(lockfile)

        if lock_fd is None:
            logging.info("Another instance is already running. Attempting to terminate the existing process...")
            try:
                with open(lockfile, 'r') as f:
                    pid = int(f.read().strip())

                terminate_process(pid)

                os.remove(lockfile)

                lock_fd = acquire_lock(lockfile)
                if lock_fd is None:
                    logging.error("Failed to acquire lock even after terminating the previous process. Exiting.")
                    sys.exit(1)
            except Exception as e:
                logging.error(f"Error while trying to terminate the previous process: {e}")
                sys.exit(1)

        logging.info("Starting autotrade script")

        conn = init_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades")
        trade_count = cursor.fetchone()[0]

        if trade_count == 0:
            logging.info("No trades found. Making first trading decision.")
            trade_result = ai_trading()
            last_trade_time = datetime.now()
            short_term_necessity = trade_result.get('short_term_necessity', 0.5)

            volatility = check_market_volatility()
            volume = check_trading_volume()

            interval = calculate_trading_interval(volatility, volume, short_term_necessity)
            next_trade_time = last_trade_time + timedelta(seconds=interval)
            save_next_trade_time(next_trade_time)
            save_last_trade_time(last_trade_time)
            logging.info(f"First trade made at: {last_trade_time}")
            logging.info(f"Next trade scheduled in {interval/60:.2f} minutes at {next_trade_time}")
        else:
            last_trade_time = load_last_trade_time()
            next_trade_time = load_next_trade_time()
            if last_trade_time is None or next_trade_time is None:
                last_trade_time = datetime.now()
                next_trade_time = datetime.now() + timedelta(seconds=MIN_TRADE_INTERVAL)
                save_last_trade_time(last_trade_time)
                save_next_trade_time(next_trade_time)
        conn.close()

        while True:
            try:
                current_time = datetime.now()
                time_since_last_trade = (current_time - last_trade_time).total_seconds()
                time_until_next_trade = (next_trade_time - current_time).total_seconds()

                if time_until_next_trade <= 0:
                    trade_result = ai_trading()
                    last_trade_time = datetime.now()
                    short_term_necessity = trade_result.get('short_term_necessity', 0.5)

                    volatility = check_market_volatility()
                    volume = check_trading_volume()

                    interval = calculate_trading_interval(volatility, volume, short_term_necessity)
                    next_trade_time = current_time + timedelta(seconds=interval)
                    save_next_trade_time(next_trade_time)
                    save_last_trade_time(last_trade_time)

                    logging.info(f"Trade decision made at: {current_time}")
                    logging.info(f"Next trade scheduled in {interval/60:.2f} minutes at {next_trade_time}")
                else:
                    if check_market_conditions() and time_since_last_trade >= MIN_TRADE_INTERVAL:
                        trade_result = ai_trading()
                        last_trade_time = datetime.now()
                        short_term_necessity = trade_result.get('short_term_necessity', 0.5)

                        volatility = check_market_volatility()
                        volume = check_trading_volume()

                        interval = calculate_trading_interval(volatility, volume, short_term_necessity)
                        next_trade_time = current_time + timedelta(seconds=interval)
                        save_next_trade_time(next_trade_time)
                        save_last_trade_time(last_trade_time)

                        logging.info(f"Market conditions met. Trade decision made at: {current_time}")
                        logging.info(f"Next trade scheduled in {interval/60:.2f} minutes at {next_trade_time}")
                    else:
                        logging.info(f"Next trade check in {time_until_next_trade/60:.2f} minutes at {next_trade_time}")

                time.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}", exc_info=True)
                time.sleep(300)  # Retry after 5 minutes if error occurs

    except Exception as e:
        logging.error(f"메인 스크립트에서 예상치 못한 오류 발생: {str(e)}", exc_info=True)
    finally:
        if lock_fd:
            release_lock(lock_fd)
        if os.path.exists(lockfile):
            os.remove(lockfile)
        logging.info("Autotrade 스크립트 종료.")

