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
     daily_profit REAL,
     total_profit REAL,
     total_assets_krw REAL,
     cumulative_reflection TEXT,
     adjusted_profit REAL,
     twr REAL,
     mwr REAL,
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

        # Adjusted Total Profit 계산 수정
        initial_total_assets = df['total_assets_krw'].iloc[0]
        if initial_total_assets != 0:
            df['adjusted_total_profit'] = (df['total_assets_krw'] - initial_total_assets - df['net_external_flow']) / initial_total_assets
        else:
            df['adjusted_total_profit'] = 0
        
    return df

def calculate_twr(df):
    if 'twr' in df.columns and not df['twr'].isnull().all():
        return df['twr'].iloc[-1]
    else:
        df = df.sort_values('timestamp')
        df['return'] = df['total_assets_krw'].pct_change()
        df['return'] = df['return'].fillna(0)  # 첫 번째 값 NaN 처리
        cumulative_return = (1 + df['return']).prod() - 1
        return cumulative_return * 100  # 백분율로 변환

def calculate_mwr(df):
    if 'mwr' in df.columns and not df['mwr'].isnull().all():
        return df['mwr'].iloc[-1]
    else:
        cashflows = df['total_assets_krw'].diff().fillna(df['total_assets_krw'])
        dates = pd.to_datetime(df['timestamp'])

        def npv(rate):
            total = 0.0
            for cf, date in zip(cashflows, dates):
                days = (date - dates.iloc[0]).days
                total += cf / ((1 + rate) ** (days / 365.0))
            return total

        try:
            result = optimize.newton(npv, 0.1)
            return result * 100  # 백분율로 변환
        except (RuntimeError, OverflowError, ValueError):
            return None

def get_next_trade_time():
    try:
        with open('next_trade_time.json', 'r') as f:
            data = json.load(f)
            next_trade_time = datetime.fromisoformat(data['next_trade_time'])
            return next_trade_time
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None

def calculate_period_profits(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # 일간 수익률 계산
    df['date'] = df['timestamp'].dt.date
    daily_profits = df.groupby('date')['trade_profit'].sum()
    
    # 주간 수익률 계산
    df['week'] = df['timestamp'].dt.to_period('W')
    weekly_profits = df.groupby('week')['trade_profit'].sum()
    
    # 월간 수익률 계산
    df['month'] = df['timestamp'].dt.to_period('M')
    monthly_profits = df.groupby('month')['trade_profit'].sum()
    
    return daily_profits, weekly_profits, monthly_profits

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
        
        if st.sidebar.button('Apply Filter'):
            st.experimental_rerun()

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
            if 'adjusted_total_profit' in filtered_df.columns and len(filtered_df) > 0:
                total_profit = filtered_df['adjusted_total_profit'].iloc[-1] * 100
                st.markdown(f"<p class='value-font' style='text-align: center;'>{total_profit:.2f}%</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='value-font' style='text-align: center;'>N/A</p>", unsafe_allow_html=True)
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

        st.markdown("<h2>Profit Summary</h2>", unsafe_allow_html=True)
        if 'trade_profit' in df.columns and 'total_profit' in df.columns and len(df) > 0:
            latest_data = df.iloc[-1]
            daily_profits, weekly_profits, monthly_profits = calculate_period_profits(df)
            
            # 최근 거래, 일간, 주간, 월간 수익률 계산
            latest_trade_profit = latest_data['trade_profit']
            latest_daily_profit = daily_profits.iloc[-1] if len(daily_profits) > 0 else 0
            latest_weekly_profit = weekly_profits.iloc[-1] if len(weekly_profits) > 0 else 0
            latest_monthly_profit = monthly_profits.iloc[-1] if len(monthly_profits) > 0 else 0
            
            # TWR과 MWR 값 가져오기
            twr_value = latest_data['twr'] if not pd.isnull(latest_data['twr']) else calculate_twr(df)
            mwr_value = latest_data['mwr'] if not pd.isnull(latest_data['mwr']) else calculate_mwr(df)
            
            profit_summary = pd.DataFrame({
                'Period': ['Latest Trade', 'Latest Day', 'Latest Week', 'Latest Month', 'Total', 'TWR', 'MWR'],
                'Profit': [
                    f"{latest_trade_profit*100:.2f}%",
                    f"{latest_daily_profit*100:.2f}%",
                    f"{latest_weekly_profit*100:.2f}%",
                    f"{latest_monthly_profit*100:.2f}%",
                    f"{latest_data['total_profit']*100:.2f}%",
                    f"{twr_value:.2f}%",
                    f"{mwr_value:.2f}%"
                ]
            })
            st.table(profit_summary)
        else:
            st.info("Insufficient profit data.")

        st.markdown("<h2>Recent Trade History</h2>", unsafe_allow_html=True)
        if len(filtered_df) > 0:
            df_display = filtered_df[['timestamp', 'decision', 'percentage', 'reason', 
                                      'btc_balance', 'btc_krw_value', 'krw_balance', 
                                      'total_assets_krw', 'total_assets_btc_formatted', 
                                      'btc_krw_price', 'success', 'short_term_necessity']].head(10)
            df_display = df_display.sort_values('timestamp', ascending=False)  # 최근 거래가 위로 오도록 정렬
            df_display['success'] = df_display['success'].map({1: 'Success', 0: 'Failure'})
            st.dataframe(df_display, height=300)
        else:
            st.info("No trade history yet.")

        st.markdown("<h2>Trading Performance</h2>", unsafe_allow_html=True)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Cumulative Profit', 'BTC Price'))
        
        if 'total_profit' not in df.columns:
            if 'daily_profit' in df.columns:
                df['total_profit'] = df['daily_profit'].cumsum()
            else:
                st.warning("No profit data available. Unable to display chart.")
                return

        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['total_profit'], mode='lines', name='Cumulative Profit'), row=1, col=1)
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
        fig = go.Figure(data=[go.Scatter(x=df['timestamp'], y=df['btc_avg_buy_price'], mode='lines')])
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
