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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# API keys from environment variables
MEDIASTACK_API_KEY = os.getenv('MEDIASTACK_API_KEY')
CRYPTOCOMPARE_API_KEY = os.getenv('CRYPTOCOMPARE_API_KEY')

# News data caching
news_cache = {'mediastack': None, 'cryptocompare': None, 'last_update': None}

# MIN_TRADE_INTERVAL in seconds
MIN_TRADE_INTERVAL = 3600  # 1 hour

# Read strategy files
def read_strategies():
    strategies = ""
    strategy_files = os.listdir('strategies')
    for file in strategy_files:
        if file.endswith('.txt'):
            with open(os.path.join('strategies', file), 'r', encoding='utf-8') as f:
                strategies += f.read() + "\n\n"
    return strategies

# Get current account status
def get_current_status(upbit):
    krw_balance = upbit.get_balance("KRW")
    btc_balance = upbit.get_balance("BTC")
    btc_price = pyupbit.get_current_price("KRW-BTC")

    return {
        "krw_balance": krw_balance,
        "btc_balance": btc_balance,
        "btc_price": btc_price
    }

# Get simplified orderbook
def get_simplified_orderbook():
    orderbook = pyupbit.get_orderbook("KRW-BTC")
    return {
        "ask_price": orderbook['orderbook_units'][0]['ask_price'],
        "bid_price": orderbook['orderbook_units'][0]['bid_price'],
        "ask_size": orderbook['orderbook_units'][0]['ask_size'],
        "bid_size": orderbook['orderbook_units'][0]['bid_size']
    }

# Get simplified chart data with technical indicators
def get_simplified_chart_data():
    df = pyupbit.get_ohlcv("KRW-BTC", count=100, interval="day")

    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()

    # Calculate Bollinger Bands
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

# Get Fear and Greed Index
def get_fear_and_greed_index():
    url = "https://api.alternative.me/fng/"
    response = requests.get(url)
    data = response.json()
    return {
        "value": data["data"][0]["value"],
        "classification": data["data"][0]["value_classification"]
    }

# Initialize SQLite database
def init_db():
    db_path = os.path.join(os.path.dirname(__file__), 'trading_history.db')
    logging.info(f"Initializing database at {db_path}")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create trades table
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
         daily_profit REAL,
         total_profit REAL,
         total_assets_krw REAL,
         cumulative_reflection TEXT,
         adjusted_profit REAL,
         twr REAL,
         mwr REAL,
         short_term_necessity REAL)
        ''')

        # Create reflection_summary table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reflection_summary
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         summary TEXT,
         timestamp TEXT)
        ''')

        # Add new columns if they don't exist
        columns_to_add = [
            ('daily_profit', 'REAL'),
            ('total_profit', 'REAL'),
            ('total_assets_krw', 'REAL'),
            ('cumulative_reflection', 'TEXT'),
            ('adjusted_profit', 'REAL'),
            ('twr', 'REAL'),
            ('mwr', 'REAL'),
            ('short_term_necessity', 'REAL')
        ]

        for column_name, column_type in columns_to_add:
            try:
                cursor.execute(f"ALTER TABLE trades ADD COLUMN {column_name} {column_type}")
                logging.info(f"Added column {column_name} to trades table")
            except sqlite3.OperationalError:
                logging.info(f"Column {column_name} already exists in trades table")

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS external_transactions
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         timestamp TEXT,
         type TEXT,
         amount REAL,
         currency TEXT)
        ''')

        conn.commit()
        logging.info("Database initialized successfully")
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database initialization error: {e}")
        raise

# Save trade data
def save_trade(conn, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, success=True, reflection=None, cumulative_reflection=None, adjusted_profit=None, short_term_necessity=None):
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()

    # Get previous trade data
    cursor.execute("SELECT btc_balance, krw_balance, btc_krw_price, total_assets_krw FROM trades ORDER BY timestamp DESC LIMIT 1")
    previous_trade = cursor.fetchone()

    # Calculate total assets in KRW
    total_assets_krw = krw_balance + (btc_balance * btc_krw_price)

    # Calculate profits
    if previous_trade:
        prev_btc_balance, prev_krw_balance, prev_btc_price, prev_total_assets = previous_trade
        prev_total_assets = prev_krw_balance + (prev_btc_balance * prev_btc_price)

        # Daily profit
        daily_profit = (total_assets_krw - prev_total_assets) / prev_total_assets if prev_total_assets != 0 else 0

        # Total profit
        cursor.execute("SELECT total_assets_krw FROM trades ORDER BY timestamp ASC LIMIT 1")
        first_trade = cursor.fetchone()
        if first_trade:
            initial_assets = first_trade[0]
            total_profit = (total_assets_krw - initial_assets) / initial_assets if initial_assets != 0 else 0
        else:
            total_profit = 0
    else:
        daily_profit = 0
        total_profit = 0

    # Calculate TWR and MWR if enough data
    cursor.execute("SELECT COUNT(*) FROM trades")
    trade_count = cursor.fetchone()[0]

    if trade_count > 1:
        twr = calculate_twr(conn)
        mwr = calculate_mwr(conn)
        twr = 0.0 if twr is None else twr
        mwr = 0.0 if mwr is None else mwr
    else:
        twr = 0.0
        mwr = 0.0

    try:
        cursor.execute('''
        INSERT INTO trades (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, success, reflection, daily_profit, total_profit, total_assets_krw, cumulative_reflection, twr, mwr, adjusted_profit, short_term_necessity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, success, reflection, daily_profit, total_profit, total_assets_krw, cumulative_reflection, twr, mwr, adjusted_profit, short_term_necessity))
        conn.commit()
        logging.info(f"Trade saved successfully: {decision}, {success}")
    except sqlite3.Error as e:
        logging.error(f"Error saving trade: {e}")
        conn.rollback()

# Get recent trades
def get_recent_trades(conn, days=7, limit=None):
    cursor = conn.cursor()
    one_week_ago = (datetime.now() - timedelta(days=days)).isoformat()

    if limit:
        cursor.execute("SELECT * FROM trades WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT ?", (one_week_ago, limit))
    else:
        cursor.execute("SELECT * FROM trades WHERE timestamp >= ? ORDER BY timestamp DESC", (one_week_ago,))

    return cursor.fetchall()

# Analyze performance
def analyze_performance(trades, current_price):
    performance = []
    total_profit = 0
    successful_trades = 0
    for trade in trades:
        if trade[9] == 0:  # success column
            performance.append({
                'decision': trade[2],
                'timestamp': trade[1],
                'profit': 0,
                'reason': trade[4],
                'success': False
            })
        else:
            if trade[2] == 'buy':
                profit = (current_price - trade[7]) / trade[7] * 100
            elif trade[2] == 'sell':
                profit = (trade[8] - current_price) / current_price * 100
            else:
                profit = 0
            total_profit += profit
            successful_trades += 1
            performance.append({
                'decision': trade[2],
                'timestamp': trade[1],
                'profit': profit,
                'reason': trade[4],
                'success': True
            })

    avg_profit = total_profit / successful_trades if successful_trades > 0 else 0
    return performance, avg_profit

# Generate reflection
def generate_reflection(performance, strategies, trades, avg_profit, previous_reflections):
    client = OpenAI()

    previous_reflections_text = "\n".join(previous_reflections)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
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
        ],
        max_tokens=500
    )

    return response.choices[0].message.content

# Calculate volatility
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

# Get news from MediaStack
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

# Get news from CryptoCompare
def get_cryptocompare_news():
    global news_cache
    current_time = datetime.now()
    if news_cache['cryptocompare'] is None or (current_time - news_cache['last_update']).total_seconds() > 1800:
        url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={CRYPTOCOMPARE_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            news = response.json()['Data']
            news_cache['cryptocompare'] = [{'title': item['title'], 'body': item['body']} for item in news[:10]]
            news_cache['last_update'] = current_time
        else:
            logging.error(f"CryptoCompare API error: {response.status_code}")
            return []
    return news_cache['cryptocompare']

def get_news():
    mediastack_news = get_mediastack_news()
    cryptocompare_news = get_cryptocompare_news()

    # Clean news data
    cleaned_news = []
    for news in mediastack_news + cryptocompare_news:
        cleaned_news.append({
            'title': news.get('title', '')[:100],
            'description': news.get('description', '')[:200] if 'description' in news else news.get('body', '')[:200]
        })

    return cleaned_news[:5]

# Get reflection summary
def get_reflection_summary(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT summary FROM reflection_summary ORDER BY id DESC LIMIT 1")
    result = cursor.fetchone()
    return result[0] if result else None

def update_reflection_summary(conn, new_reflection, previous_summary):
    client = OpenAI()

    if previous_summary:
        prompt = f"Previous summary: {previous_summary}\n\nNew reflection to incorporate: {new_reflection}\n\nCreate an updated summary that incorporates the new reflection into the previous summary. The summary should maintain key insights from the previous summary while adding new insights from the latest reflection. Keep the summary concise but insightful."
    else:
        prompt = f"Create a summary of the following reflection: {new_reflection}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with summarizing trading reflections."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )

    updated_summary = response.choices[0].message.content

    cursor = conn.cursor()
    cursor.execute("INSERT INTO reflection_summary (summary) VALUES (?)", (updated_summary,))
    conn.commit()

    return updated_summary

def record_external_transaction(conn, type, amount, currency):
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute('''
    INSERT INTO external_transactions (timestamp, type, amount, currency)
    VALUES (?, ?, ?, ?)
    ''', (timestamp, type, amount, currency))
    conn.commit()

def get_external_transactions(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM external_transactions ORDER BY timestamp")
    return cursor.fetchall()

def calculate_adjusted_profit(conn, current_assets):
    external_transactions = get_external_transactions(conn)
    net_external_flow = sum(amount if type == 'deposit' else -amount 
                            for _, _, type, amount, _ in external_transactions)

    cursor = conn.cursor()
    cursor.execute("SELECT total_assets_krw FROM trades ORDER BY timestamp ASC LIMIT 1")
    initial_assets = cursor.fetchone()[0]

    adjusted_profit = (current_assets - initial_assets - net_external_flow) / initial_assets
    return adjusted_profit

def check_balance_for_trade(upbit, decision, percentage):
    current_status = get_current_status(upbit)
    if decision == 'buy':
        available_krw = current_status['krw_balance']
        trade_amount = available_krw * (percentage / 100)
        if trade_amount < 5000:  # Minimum order amount
            return False, "Insufficient KRW balance for buy order"
    elif decision == 'sell':
        available_btc = current_status['btc_balance']
        trade_amount = available_btc * (percentage / 100)
        if trade_amount * current_status['btc_price'] < 5000:  # Minimum order amount
            return False, "Insufficient BTC balance for sell order"
    return True, ""

# ai_trading function
def ai_trading():
    logging.info("Starting ai_trading function")
    client = OpenAI()

    upbit = pyupbit.Upbit(os.getenv('UPBIT_ACCESS_KEY'), os.getenv('UPBIT_SECRET_KEY'))
    current_status = get_current_status(upbit)
    orderbook = get_simplified_orderbook()
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
            
        recent_trades = get_recent_trades(conn, limit=10)
        current_price = pyupbit.get_current_price("KRW-BTC")
        performance, avg_profit = analyze_performance(recent_trades, current_price)
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
        The trading interval can be adjusted between 1 hour (for very short-term trading) and 8 hours (for longer-term trading).
        Provide a short-term trading necessity score from 0 to 1, where:
        0: No need for short-term trading, prefer longer intervals (closer to 8 hours)
        1: High necessity for short-term trading, prefer shorter intervals (closer to 1 hour)
        Base this score on market volatility, recent news impact, and potential short-term opportunities."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": json.dumps({
                    "current_status": current_status,
                    "orderbook": orderbook,
                    "chart_data": chart_data,
                    "fear_greed_index": fear_greed_index,
                    "volatility_data": volatility_data,
                    "news": news
                }, ensure_ascii=False)}
            ],
            functions=[{
                "name": "trading_decision",
                "description": "Make a trading decision based on the given market data and recent performance reflection",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "decision": {
                            "type": "string",
                            "enum": ["buy", "sell", "hold"]
                        },
                        "percentage": {
                            "type": "number",
                            "description": "Percentage of total balance to trade"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Explanation for the trading decision"
                        },
                        "short_term_necessity": {
                            "type": "number",
                            "description": "Score from 0 to 1 indicating the necessity of short-term trading"
                        }
                    },
                    "required": ["decision", "percentage", "reason", "short_term_necessity"]
                }
            }],
            function_call={"name": "trading_decision"}
        )
        result = json.loads(response.choices[0].message.function_call.arguments)
        logging.info(f"AI response: {result}")

        decision = result['decision']
        percentage = result['percentage']
        reason = result['reason']
        short_term_necessity = result['short_term_necessity']
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
                           cumulative_reflection=updated_summary)
            else:
                trade_amount = current_status['krw_balance'] * (percentage / 100) if decision == 'buy' else current_status['btc_balance'] * (percentage / 100)
                
                try:
                    if decision == 'buy':
                        order = upbit.buy_market_order("KRW-BTC", trade_amount)
                    else:  # sell
                        order = upbit.sell_market_order("KRW-BTC", trade_amount)

                    if 'error' in order:
                        raise Exception(f"Order failed: {order['error']['message']}")

                    updated_status = get_current_status(upbit)
                    current_assets = updated_status['krw_balance'] + (updated_status['btc_balance'] * updated_status['btc_price'])
                    adjusted_profit = calculate_adjusted_profit(conn, current_assets)

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
                               adjusted_profit=adjusted_profit,
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
                               cumulative_reflection=updated_summary)
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
    # Base interval in seconds (8 hours)
    base_interval = 28800  # 8 hours

    # Normalize volatility and volume to a 0-1 scale (thresholds remain the same)
    normalized_volatility = min(volatility / 0.01, 1)
    normalized_volume = min(volume / 150, 1)

    # Assign higher weight to short_term_necessity
    combined_factor = (0.7 * short_term_necessity) + (0.15 * normalized_volatility) + (0.15 * normalized_volume)

    # Use an exponential decay function to adjust the interval
    interval = base_interval * (1 - combined_factor ** 2)

    # Ensure the interval is between 1 hour and 8 hours
    return max(min(interval, 28800), 3600)

def check_market_conditions():
    volatility = check_market_volatility()
    volume = check_trading_volume()

    # Thresholds remain unchanged
    if volatility > 0.005 or volume > 100:
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

def calculate_twr(conn):
    cursor = conn.cursor()
    cursor.execute("""
    SELECT trades.timestamp, trades.total_assets_krw, 
           COALESCE(SUM(CASE WHEN type = 'deposit' THEN amount ELSE -amount END), 0) as net_flow
    FROM trades
    LEFT JOIN external_transactions ON trades.timestamp >= external_transactions.timestamp
    GROUP BY trades.timestamp
    ORDER BY trades.timestamp
    """)
    data = cursor.fetchall()

    if len(data) < 2:
        return 0.0

    twr = 1
    for i in range(1, len(data)):
        prev_assets = data[i-1][1]
        curr_assets = data[i][1]
        net_flow = data[i][2]
        
        if prev_assets + net_flow != 0:
            period_return = (curr_assets - net_flow) / (prev_assets + net_flow)
            twr *= period_return

    return (twr - 1) * 100

def calculate_mwr(conn):
    cursor = conn.cursor()
    cursor.execute("""
    SELECT trades.timestamp, trades.total_assets_krw, 
           COALESCE(SUM(CASE WHEN type = 'deposit' THEN amount ELSE -amount END), 0) as net_flow
    FROM trades
    LEFT JOIN external_transactions ON trades.timestamp >= external_transactions.timestamp
    GROUP BY trades.timestamp
    ORDER BY trades.timestamp
    """)
    data = cursor.fetchall()

    dates = [(datetime.fromisoformat(row[0]) - datetime.fromisoformat(data[0][0])).days / 365.0 for row in data]
    cashflows = [-row[1] for row in data]
    cashflows[0] += data[0][1]
    cashflows[1:] = [cf + nf for cf, nf in zip(cashflows[1:], [row[2] for row in data[1:]])]

    def xirr(cashflows, dates):
        def objective(rate):
            return sum([cf / (1 + rate) ** t for cf, t in zip(cashflows, dates)])
        
        return optimize.newton(objective, 0.1)

    try:
        mwr = xirr(cashflows, dates)
        return mwr * 100
    except:
        return 0.0

def get_initial_assets(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT total_assets_krw FROM trades ORDER BY timestamp ASC LIMIT 1")
    result = cursor.fetchone()
    return result[0] if result else 0

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

        os.write(lock_fd.fileno(), str(os.getpid()).encode())
        
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
                logging.error(f"Error in main loop: {e}")
                time.sleep(300)  # Retry after 5 minutes if error occurs

    except Exception as e:
        logging.error(f"Unexpected error in main script: {e}")
    finally:
        if lock_fd:
            release_lock(lock_fd)
        if os.path.exists(lockfile):
            os.remove(lockfile)
        logging.info("Autotrade script terminated.")
