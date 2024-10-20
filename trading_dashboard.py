# trading_dashboard.py

# type: ignore
import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import json
from datetime import datetime, timedelta
from scipy import optimize
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 데이터베이스 연결 및 테이블 생성 함수
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
    CREATE TABLE IF NOT EXISTS external_transactions
    (id INTEGER PRIMARY KEY AUTOINCREMENT,
     timestamp TEXT,
     type TEXT,
     amount REAL,
     currency TEXT)
    ''')
    conn.commit()
    return conn

# 거래 데이터 가져오기
def get_trade_data():
    conn = init_db()
    query = """
    SELECT t.*, e.net_external_flow
    FROM trades t
    LEFT JOIN (
        SELECT timestamp,
               SUM(CASE WHEN type = 'deposit' THEN amount ELSE -amount END) OVER (ORDER BY timestamp) as net_external_flow
        FROM external_transactions
    ) e ON t.timestamp >= e.timestamp
    ORDER BY t.timestamp ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['btc_krw_value'] = df['btc_balance'] * df['btc_krw_price']
        df['total_assets_krw'] = df['krw_balance'] + df['btc_krw_value']
        df['total_assets_btc'] = df['btc_balance'] + (df['krw_balance'] / df['btc_krw_price'])
        df['total_assets_btc_formatted'] = df['total_assets_btc'].apply(lambda x: f"{x:.4e}")
        
        # net_external_flow의 NaN 값을 0으로 대체
        df['net_external_flow'] = df['net_external_flow'].fillna(0)
        
    return df

def calculate_twr(df):
    df = df.sort_values('timestamp')
    df['daily_return'] = df['total_assets_krw'].pct_change()
    df['daily_return'] = df['daily_return'].fillna(0)
    twr = np.prod(1 + df['daily_return']) - 1
    return twr * 100  # 백분율로 변환

def get_next_trade_time():
    try:
        with open('next_trade_time.json', 'r') as f:
            data = json.load(f)
            next_trade_time = datetime.fromisoformat(data['next_trade_time'])
            return next_trade_time
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None

def calculate_profit(initial_value, final_value):
    if initial_value == 0:
        return 0
    return (final_value - initial_value) / initial_value * 100

# 수익 계산 함수 수정
def calculate_profits(df):
    if len(df) == 0:
        return 0, 0, 0, 0, 0

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    current_assets = df['total_assets_krw'].iloc[-1]
    initial_assets = df['total_assets_krw'].iloc[0]

    # Total Profit 계산
    total_profit = calculate_profit(initial_assets, current_assets)

    # Latest Day Profit 계산
    yesterday = df['timestamp'].max() - timedelta(days=1)
    yesterday_assets = df[df['timestamp'] <= yesterday]['total_assets_krw'].iloc[-1] if len(df[df['timestamp'] <= yesterday]) > 0 else initial_assets
    latest_day_profit = calculate_profit(yesterday_assets, current_assets)

    # Latest Week Profit 계산
    last_week = df['timestamp'].max() - timedelta(days=7)
    last_week_assets = df[df['timestamp'] <= last_week]['total_assets_krw'].iloc[-1] if len(df[df['timestamp'] <= last_week]) > 0 else initial_assets
    latest_week_profit = calculate_profit(last_week_assets, current_assets)

    # Latest Month Profit 계산
    last_month = df['timestamp'].max() - timedelta(days=30)
    last_month_assets = df[df['timestamp'] <= last_month]['total_assets_krw'].iloc[-1] if len(df[df['timestamp'] <= last_month]) > 0 else initial_assets
    latest_month_profit = calculate_profit(last_month_assets, current_assets)

    # TWR 계산
    twr = calculate_twr(df)

    duration = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days

    logging.info(f"Initial assets: {initial_assets}, Current assets: {current_assets}")
    logging.info(f"Investment duration: {duration} days")
    logging.info(f"TWR: {twr:.5f}%")

    return total_profit, latest_day_profit, latest_week_profit, latest_month_profit, twr

# 메인 대시보드 함수
def main():
    st.set_page_config(layout="wide", page_title="AiBiCo Trading Dashboard", initial_sidebar_state="collapsed")
    
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    .big-font {
        font-size: clamp(16px, 2vw, 24px) !important;
        font-weight: 700 !important;
    }
    .medium-font {
        font-size: clamp(14px, 1.8vw, 20px) !important;
        font-weight: 500 !important;
    }
    .small-font {
        font-size: clamp(12px, 1.6vw, 18px) !important;
        font-weight: 400 !important;
    }
    .value-font {
        font-size: clamp(14px, 1.7vw, 22px) !important;
        font-weight: 500 !important;
    }
    .date-font {
        font-size: clamp(10px, 1.2vw, 16px) !important;
        font-weight: 400 !important;
        line-height: 1.2;
    }
    h1 {
        font-size: clamp(24px, 3vw, 36px) !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 30px;
    }
    h2 {
        font-size: clamp(20px, 2.5vw, 30px) !important;
        font-weight: 600 !important;
        text-align: center;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    h3 {
        font-size: clamp(18px, 2.2vw, 26px) !important;
        font-weight: 500 !important;
        text-align: center;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .stDataFrame {
        font-size: clamp(10px, 1.4vw, 14px) !important;
    }
    .stTable {
        font-size: clamp(10px, 1.4vw, 14px) !important;
    }
    @media (max-width: 768px) {
        .stDataFrame, .stTable {
            font-size: 10px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1>💰 AiBiCo Trading Dashboard</h1>", unsafe_allow_html=True)

    df = get_trade_data()
    
    if len(df) > 0:
        st.sidebar.header("Data Filter")
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        if start_date <= end_date:
            mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
            filtered_df = df.loc[mask]
        else:
            st.sidebar.error("Error: End date must be after start date.")
            filtered_df = df

        # 'initial' 결정을 제외
        filtered_df = filtered_df[filtered_df['decision'] != 'initial']

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.markdown("<p class='big-font'>Total Trades</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='value-font' style='text-align: center;'>{len(filtered_df)}</p>", unsafe_allow_html=True)
        with col2:
            st.markdown("<p class='big-font'>Success Rate</p>", unsafe_allow_html=True)
            success_rate = (filtered_df['success'].sum() / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
            st.markdown(f"<p class='value-font' style='text-align: center;'>{success_rate:.2f}%</p>", unsafe_allow_html=True)
        with col3:
            st.markdown("<p class='big-font'>Total Profit</p>", unsafe_allow_html=True)
            total_profit = calculate_profit(filtered_df['total_assets_krw'].iloc[0], filtered_df['total_assets_krw'].iloc[-1])
            st.markdown(f"<p class='value-font' style='text-align: center;'>{total_profit:.5f}%</p>", unsafe_allow_html=True)
        with col4:
            st.markdown("<p class='big-font'>BTC Price</p>", unsafe_allow_html=True)
            current_price = filtered_df['btc_krw_price'].iloc[-1] if len(filtered_df) > 0 else 0
            st.markdown(f"<p class='value-font' style='text-align: center;'>₩{current_price:,.0f}</p>", unsafe_allow_html=True)
        with col5:
            st.markdown("<p class='big-font'>Trading Period</p>", unsafe_allow_html=True)
            if len(filtered_df) > 0:
                start_date_str = filtered_df['timestamp'].min().strftime('%Y-%m-%d')
                end_date_str = filtered_df['timestamp'].max().strftime('%Y-%m-%d')
                st.markdown(f"<p class='date-font' style='text-align: center;'>{start_date_str}<br>{end_date_str}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='date-font' style='text-align: center;'>No data</p>", unsafe_allow_html=True)
        with col6:
            st.markdown("<p class='big-font'>Next Trade</p>", unsafe_allow_html=True)
            next_trade_time = get_next_trade_time()
            if next_trade_time:
                date = next_trade_time.strftime('%Y-%m-%d')
                time = next_trade_time.strftime('%H:%M:%S')
                st.markdown(f"<p class='date-font' style='text-align: center;'>{date}<br>{time}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='date-font' style='text-align: center;'>No information</p>", unsafe_allow_html=True)

        total_profit, latest_day_profit, latest_week_profit, latest_month_profit, twr = calculate_profits(filtered_df)

        # Profit Summary 섹션 수정
        st.markdown("<h2>Profit Summary</h2>", unsafe_allow_html=True)
        profit_summary = pd.DataFrame({
            'Period': ['Total', 'Latest Day', 'Latest Week', 'Latest Month', 'TWR'],
            'Profit': [
                f"{total_profit:.5f}%",
                f"{latest_day_profit:.5f}%",
                f"{latest_week_profit:.5f}%",
                f"{latest_month_profit:.5f}%",
                f"{twr:.5f}%"
            ]
        })
        st.table(profit_summary)

        st.markdown("<h2>Recent Trade History</h2>", unsafe_allow_html=True)
        if len(filtered_df) > 0:
            df_display = filtered_df[['timestamp', 'decision', 'percentage', 'reason', 
                                      'btc_balance', 'btc_krw_value', 'krw_balance', 
                                      'total_assets_krw', 'total_assets_btc_formatted', 
                                      'btc_krw_price', 'success', 'short_term_necessity']]
            df_display = df_display.sort_values('timestamp', ascending=False)  # 최근 거래가 위로 오도록 정렬
            df_display['success'] = df_display['success'].map({1: 'Success', 0: 'Failure'})
            st.dataframe(df_display, height=400)  # 높이를 조정하여 더 많은 행을 표시
        else:
            st.info("No trade history yet.")

        st.markdown("<h2>Trading Performance</h2>", unsafe_allow_html=True)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Cumulative Profit', 'BTC Price'))
        
        # Cumulative Profit 계산
        df['cumulative_profit'] = (df['total_assets_krw'] / df['total_assets_krw'].iloc[0] - 1) * 100

        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['cumulative_profit'], mode='lines', name='Cumulative Profit'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['btc_krw_price'], mode='lines', name='BTC Price'), row=2, col=1)
        fig.update_layout(height=600, width=1000)
        st.plotly_chart(fig, use_container_width=True)

        # 원형 그래프를 나란히 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3>Trade Decision Distribution</h3>", unsafe_allow_html=True)
            decision_counts = df['decision'].value_counts()
            decision_counts = decision_counts[decision_counts.index.isin(['buy', 'sell', 'hold'])]
            fig = go.Figure(data=[go.Pie(labels=decision_counts.index, values=decision_counts.values)])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("<h3>Trade Success/Failure Ratio</h3>", unsafe_allow_html=True)
            success_counts = df['success'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=['Success', 'Failure'], values=[success_counts.get(1, 0), success_counts.get(0, 0)])])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<h3>BTC Balance Change</h3>", unsafe_allow_html=True)
        fig = go.Figure(data=[go.Scatter(x=df['timestamp'], y=df['btc_balance'], mode='lines')])
        fig.update_layout(title='BTC Balance Over Time', xaxis_title='Timestamp', yaxis_title='BTC Balance')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<h3>KRW Balance Change</h3>", unsafe_allow_html=True)
        fig = go.Figure(data=[go.Scatter(x=df['timestamp'], y=df['krw_balance'], mode='lines')])
        fig.update_layout(title='KRW Balance Over Time', xaxis_title='Timestamp', yaxis_title='KRW Balance')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<h3>BTC Average Buy Price Change</h3>", unsafe_allow_html=True)
        # 0이 아닌 값만 필터링
        non_zero_df = df[df['btc_avg_buy_price'] > 0]
        fig = go.Figure(data=[go.Scatter(x=non_zero_df['timestamp'], y=non_zero_df['btc_avg_buy_price'], mode='lines')])
        fig.update_layout(title='BTC Average Buy Price Over Time', xaxis_title='Timestamp', yaxis_title='BTC Average Buy Price')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<h2>Latest Reflection</h2>", unsafe_allow_html=True)
        if 'reflection' in df.columns and len(df) > 0:
            latest_reflection = df.loc[df['reflection'].notna(), 'reflection'].iloc[-1] if not df['reflection'].isna().all() else "No reflection available."
        else:
            latest_reflection = "No reflection available."
        
        st.markdown(f"<p class='small-font' style='text-align: center;'>{latest_reflection}</p>", unsafe_allow_html=True)

        st.markdown("<h2>Reflection History</h2>", unsafe_allow_html=True)
        if 'reflection' in df.columns and 'cumulative_reflection' in df.columns:
            reflection_df = df[['timestamp', 'decision', 'reflection', 'cumulative_reflection']].dropna(subset=['reflection'])
            if not reflection_df.empty:
                reflection_df = reflection_df.sort_values('timestamp', ascending=False)  # 최근 반성이 위로 오도록 정렬
                st.dataframe(reflection_df, height=300)
            else:
                st.info("No reflections recorded for any trades.")
        else:
            st.info("Reflection or cumulative reflection column does not exist.")

    else:
        st.warning("No trading data available. The auto-trading bot may be running but hasn't made any trades yet, or the bot may not be running.")
        
        # Empty data handling
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.markdown("<p class='big-font'>Total Trades</p>", unsafe_allow_html=True)
            st.markdown("<p class='medium-font' style='text-align: center;'>0</p>", unsafe_allow_html=True)
        with col2:
            st.markdown("<p class='big-font'>Success Rate</p>", unsafe_allow_html=True)
            st.markdown("<p class='medium-font' style='text-align: center;'>0.00%</p>", unsafe_allow_html=True)
        with col3:
            st.markdown("<p class='big-font'>Total Profit</p>", unsafe_allow_html=True)
            st.markdown("<p class='medium-font' style='text-align: center;'>N/A</p>", unsafe_allow_html=True)
        with col4:
            st.markdown("<p class='big-font'>Current BTC Price</p>", unsafe_allow_html=True)
            st.markdown("<p class='medium-font' style='text-align: center;'>No data</p>", unsafe_allow_html=True)
        with col5:
            st.markdown("<p class='big-font'>Trading Period</p>", unsafe_allow_html=True)
            st.markdown("<p class='small-font' style='text-align: center;'>No data</p>", unsafe_allow_html=True)
        with col6:
            st.markdown("<p class='big-font'>Next Trade</p>", unsafe_allow_html=True)
            next_trade_time = get_next_trade_time()
            if next_trade_time:
                st.markdown(f"<p class='small-font' style='text-align: center;'>{next_trade_time.strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='small-font' style='text-align: center;'>No information</p>", unsafe_allow_html=True)

        st.info("Charts and graphs will be displayed here once trading data is generated.")

if __name__ == "__main__":
    main()
