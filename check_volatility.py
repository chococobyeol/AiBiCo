import pyupbit
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# 로깅 설정
logging.basicConfig(filename='volatility.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_volatility(df, window=14):
    df['daily_return'] = df['close'].pct_change()
    volatility = df['daily_return'].rolling(window=window).std() * np.sqrt(365)
    return volatility.iloc[-1]

def get_current_volatility():
    try:
        # 최근 30일간의 일별 데이터 가져오기
        df = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=30)
        
        # 현재 변동성 계산 (14일 기준)
        current_volatility = calculate_volatility(df)
        
        # 평균 변동성 계산 (30일 기준)
        avg_volatility = df['daily_return'].std() * np.sqrt(365)
        
        # 10분 간격 데이터로 단기 변동성 계산
        df_10min = pyupbit.get_ohlcv("KRW-BTC", interval="minute10", count=144)  # 24시간 데이터
        short_term_volatility = calculate_volatility(df_10min, window=144)
        
        # autotrade.py 방식의 변동성 계산
        df_1hour = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=24)
        autotrade_volatility = (df_1hour['high'] - df_1hour['low']).mean() / df_1hour['open'].mean()
        
        return current_volatility, avg_volatility, short_term_volatility, autotrade_volatility
    except Exception as e:
        logging.error(f"Error calculating volatility: {e}")
        return None, None, None, None

def get_average_volume(interval="day", count=14):
    try:
        df = pyupbit.get_ohlcv("KRW-BTC", interval=interval, count=count)
        avg_volume = df['volume'].mean()
        return avg_volume
    except Exception as e:
        logging.error(f"Error getting average volume: {str(e)}")
        return None

def check_trading_volume():
    try:
        # 24시간 평균 거래량
        df_24h = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=24)
        avg_volume_24h = df_24h['volume'].sum()  # 24시간 총 거래량

        # 14일 평균 거래량
        avg_volume_14d = get_average_volume()

        return {
            "avg_volume_24h": avg_volume_24h,
            "avg_volume_14d": avg_volume_14d
        }
    except Exception as e:
        logging.error(f"Error checking trading volume: {e}")
        return None

def get_10min_volatility():
    try:
        # 최근 10분간의 1분 데이터 가져오기
        df = pyupbit.get_ohlcv("KRW-BTC", interval="minute1", count=10)
        
        # 변동성 계산 (고가와 저가의 차이를 시가로 나눔)
        volatility = (df['high'].max() - df['low'].min()) / df['open'].iloc[0]
        
        return volatility
    except Exception as e:
        logging.error(f"Error calculating 10-minute volatility: {e}")
        return None

def get_10min_volume():
    try:
        # 최근 10분간의 1분 데이터 가져오기
        df = pyupbit.get_ohlcv("KRW-BTC", interval="minute1", count=10)
        
        # 10분 동안의 총 거래량 계산
        total_volume = df['volume'].sum()
        
        return total_volume
    except Exception as e:
        logging.error(f"Error calculating 10-minute volume: {e}")
        return None

def check_market_conditions():
    current_volatility, avg_volatility, short_term_volatility, autotrade_volatility = get_current_volatility()
    volume_data = check_trading_volume()
    ten_min_volatility = get_10min_volatility()
    ten_min_volume = get_10min_volume()
    
    if current_volatility is None or volume_data is None or ten_min_volatility is None or ten_min_volume is None:
        logging.error("Failed to get volatility or volume data")
        return False

    logging.info(f"Current Volatility (14-day): {current_volatility:.4f}")
    logging.info(f"Average Volatility (30-day): {avg_volatility:.4f}")
    logging.info(f"Short-term Volatility (24-hour): {short_term_volatility:.4f}")
    logging.info(f"Autotrade Volatility (24-hour): {autotrade_volatility:.4f}")
    logging.info(f"10-minute Volatility: {ten_min_volatility:.4f}")
    logging.info(f"10-minute Volume: {ten_min_volume:.2f}")
    logging.info(f"24-hour Trading Volume: {volume_data['avg_volume_24h']:.2f}")
    logging.info(f"Average Daily Trading Volume (14-day): {volume_data['avg_volume_14d']:.2f}")

    # 추가된 로그
    avg_10min_volume_24h = volume_data['avg_volume_24h'] / 144
    avg_10min_volume_14d = volume_data['avg_volume_14d'] / 2016
    logging.info(f"Average 10-min Volume (24-hour): {avg_10min_volume_24h:.2f}")
    logging.info(f"Average 10-min Volume (14-day): {avg_10min_volume_14d:.2f}")

    # 변동성 조건 확인
    volatility_condition_1 = ten_min_volatility > short_term_volatility * 1.5
    volatility_condition_2 = ten_min_volatility > current_volatility * 3
    logging.info(f"10-min volatility > 24-hour volatility * 1.5: {volatility_condition_1}")
    logging.info(f"10-min volatility > 14-day volatility * 3: {volatility_condition_2}")

    # 거래량 조건 확인
    volume_condition_1 = ten_min_volume > avg_10min_volume_24h * 1.5
    volume_condition_2 = ten_min_volume > avg_10min_volume_14d * 3
    logging.info(f"10-min volume ({ten_min_volume:.2f}) > (24-hour volume / 144) * 1.5 ({avg_10min_volume_24h * 1.5:.2f}): {volume_condition_1}")
    logging.info(f"10-min volume ({ten_min_volume:.2f}) > (14-day volume / 2016) * 3 ({avg_10min_volume_14d * 3:.2f}): {volume_condition_2}")

    # 종합 조건 확인
    market_condition = (volatility_condition_1 or volatility_condition_2) or (volume_condition_1 or volume_condition_2)
    logging.info(f"Overall market condition met: {market_condition}")

    if market_condition:
        logging.warning("Market conditions indicate high volatility or volume!")
        return True
    return False

def main():
    market_conditions_met = check_market_conditions()
    if market_conditions_met:
        logging.info("Market conditions suggest potential trading opportunity.")
    else:
        logging.info("Market conditions are normal.")

if __name__ == "__main__":
    main()
