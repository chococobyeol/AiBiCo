import pyupbit
import pandas as pd
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import numpy as np

def get_chart_data():
    df = pyupbit.get_ohlcv("KRW-BTC", count=100, interval="day")
    
    # RSI 계산
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD 계산
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 볼린저 밴드 계산
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['stddev'] = df['close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (df['stddev'] * 2)
    df['Lower'] = df['MA20'] - (df['stddev'] * 2)

    return df

def visualize_data(df):
    # 날짜 형식 변환
    df['Date'] = df.index.map(mdates.date2num)

    # 서브플롯 생성
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

    # 캔들스틱 차트
    candlestick_ohlc(ax1, df[['Date', 'open', 'high', 'low', 'close']].values, width=0.6, colorup='red', colordown='blue')
    ax1.plot(df['Date'], df['MA20'], label='MA20', color='orange')
    ax1.plot(df['Date'], df['Upper'], label='Upper BB', color='gray', linestyle='--')
    ax1.plot(df['Date'], df['Lower'], label='Lower BB', color='gray', linestyle='--')
    ax1.set_title('Bitcoin Price with Bollinger Bands')
    ax1.legend()

    # MACD
    ax2.plot(df['Date'], df['MACD'], label='MACD', color='blue')
    ax2.plot(df['Date'], df['Signal'], label='Signal', color='orange')
    ax2.bar(df['Date'], df['MACD'] - df['Signal'], label='MACD Histogram')
    ax2.set_title('MACD')
    ax2.legend()

    # RSI
    ax3.plot(df['Date'], df['RSI'], label='RSI', color='purple')
    ax3.axhline(y=70, color='red', linestyle='--')
    ax3.axhline(y=30, color='green', linestyle='--')
    ax3.set_title('RSI')
    ax3.legend()

    # x축 날짜 포맷 설정
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # 날짜 레이블 자동 포맷

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = get_chart_data()
    visualize_data(df)
