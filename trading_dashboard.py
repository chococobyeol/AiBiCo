# type: ignore
import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import json
from datetime import datetime, timedelta

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
     cumulative_reflection TEXT)
    ''')
    conn.commit()
    return conn

# 거래 데이터 가져오기
def get_trade_data():
    conn = init_db()
    query = "SELECT * FROM trades ORDER BY timestamp DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['btc_krw_value'] = df['btc_balance'] * df['btc_krw_price']
        df['total_assets_btc'] = df['btc_balance'] + (df['krw_balance'] / df['btc_krw_price'])
        df['total_assets_btc_formatted'] = df['total_assets_btc'].apply(lambda x: f"{x:.4e}")
    
    return df

# 성과 계산 함수 제거 (데이터베이스에서 직접 가져오므로 필요 없음)

def get_next_trade_time():
    try:
        with open('next_trade_time.json', 'r') as f:
            data = json.load(f)
            next_trade_time = datetime.fromisoformat(data['next_trade_time'])
            return next_trade_time
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None

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
        font-size: 1.5vw !important;
        font-weight: 700 !important;
    }
    .medium-font {
        font-size: 1.2vw !important;
        font-weight: 500 !important;
    }
    .small-font {
        font-size: 1vw !important;
        font-weight: 400 !important;
    }
    .value-font {
        font-size: 1.1vw !important;
        font-weight: 500 !important;
    }
    h1 {
        font-size: 2.5vw !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 30px;
    }
    h2 {
        font-size: 2vw !important;
        font-weight: 600 !important;
        text-align: center;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    h3 {
        font-size: 1.5vw !important;
        font-weight: 500 !important;
        text-align: center;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .stDataFrame {
        font-size: 0.8vw !important;
    }
    .stTable {
        font-size: 0.8vw !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1>💰 AiBiCo Trading Dashboard</h1>", unsafe_allow_html=True)

    df = get_trade_data()
    
    if len(df) > 0:
        st.sidebar.header("Data Filter")
        start_date = st.sidebar.date_input("Start Date", df['timestamp'].min().date())
        end_date = st.sidebar.date_input("End Date", df['timestamp'].max().date())
        
        if start_date <= end_date:
            mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
            filtered_df = df.loc[mask]
        else:
            st.sidebar.error("Error: End date must be after start date.")
            filtered_df = df

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
            if 'total_profit' in filtered_df.columns and len(filtered_df) > 0:
                total_profit = filtered_df['total_profit'].iloc[0] * 100
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
                start_date = filtered_df['timestamp'].min().strftime('%Y-%m-%d')
                end_date = filtered_df['timestamp'].max().strftime('%Y-%m-%d')
                st.markdown(f"<p class='value-font' style='text-align: center;'>{start_date} ~ {end_date}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='value-font' style='text-align: center;'>No data</p>", unsafe_allow_html=True)
        with col6:
            st.markdown("<p class='big-font'>Next Trade</p>", unsafe_allow_html=True)
            next_trade_time = get_next_trade_time()
            if next_trade_time:
                st.markdown(f"<p class='value-font' style='text-align: center;'>{next_trade_time.strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='value-font' style='text-align: center;'>No information</p>", unsafe_allow_html=True)

        st.markdown("<h2>Profit Summary</h2>", unsafe_allow_html=True)
        if 'daily_profit' in filtered_df.columns and 'total_profit' in filtered_df.columns and len(filtered_df) > 0:
            latest_data = filtered_df.iloc[0]
            weekly_profit = filtered_df['daily_profit'].head(7).sum() if len(filtered_df) >= 7 else filtered_df['daily_profit'].sum()
            monthly_profit = filtered_df['daily_profit'].head(30).sum() if len(filtered_df) >= 30 else filtered_df['daily_profit'].sum()
            profit_summary = pd.DataFrame({
                'Period': ['Daily', 'Weekly', 'Monthly', 'Total'],
                'Profit': [
                    f"{latest_data['daily_profit']*100:.2f}%",
                    f"{weekly_profit*100:.2f}%",
                    f"{monthly_profit*100:.2f}%",
                    f"{latest_data['total_profit']*100:.2f}%"
                ]
            })
            st.table(profit_summary)
        else:
            st.info("Insufficient profit data.")

        st.markdown("<h2>Recent Trade History</h2>", unsafe_allow_html=True)
        if len(filtered_df) > 0:
            df_display = filtered_df[['timestamp', 'decision', 'percentage', 'reason', 
                                      'btc_balance', 'btc_krw_value', 'krw_balance', 
                                      'total_assets_btc_formatted', 
                                      'btc_krw_price', 'success']].head(10)
            df_display['success'] = df_display['success'].map({1: 'Success', 0: 'Failure'})
            st.dataframe(df_display, height=300)
        else:
            st.info("No trade history yet.")

        st.markdown("<h2>Trading Performance</h2>", unsafe_allow_html=True)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Cumulative Profit', 'BTC Price'))
        
        if 'total_profit' not in filtered_df.columns:
            if 'daily_profit' in filtered_df.columns:
                filtered_df['total_profit'] = filtered_df['daily_profit'].cumsum()
            else:
                st.warning("No profit data available. Unable to display chart.")
                return

        fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['total_profit'], mode='lines', name='Cumulative Profit'), row=1, col=1)
        fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['btc_krw_price'], mode='lines', name='BTC Price'), row=2, col=1)
        fig.update_layout(height=600, width=1000)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3>Trade Decision Distribution</h3>", unsafe_allow_html=True)
            decision_counts = filtered_df['decision'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=decision_counts.index, values=decision_counts.values)])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("<h3>Trade Success/Failure Ratio</h3>", unsafe_allow_html=True)
            success_counts = filtered_df['success'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=['Success', 'Failure'], values=[success_counts.get(1, 0), success_counts.get(0, 0)])])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<h3>BTC Balance Change</h3>", unsafe_allow_html=True)
        fig = go.Figure(data=[go.Scatter(x=filtered_df['timestamp'], y=filtered_df['btc_balance'], mode='lines')])
        fig.update_layout(title='BTC Balance Over Time', xaxis_title='Timestamp', yaxis_title='BTC Balance')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<h3>KRW Balance Change</h3>", unsafe_allow_html=True)
        fig = go.Figure(data=[go.Scatter(x=filtered_df['timestamp'], y=filtered_df['krw_balance'], mode='lines')])
        fig.update_layout(title='KRW Balance Over Time', xaxis_title='Timestamp', yaxis_title='KRW Balance')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<h3>BTC Average Buy Price Change</h3>", unsafe_allow_html=True)
        fig = go.Figure(data=[go.Scatter(x=filtered_df['timestamp'], y=filtered_df['btc_avg_buy_price'], mode='lines')])
        fig.update_layout(title='BTC Average Buy Price Over Time', xaxis_title='Timestamp', yaxis_title='BTC Average Buy Price')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<h2>Latest Reflection</h2>", unsafe_allow_html=True)
        if 'reflection' in filtered_df.columns and len(filtered_df) > 0:
            latest_reflection = filtered_df.loc[filtered_df['reflection'].notna(), 'reflection'].iloc[0] if not filtered_df['reflection'].isna().all() else "No reflection available."
        else:
            latest_reflection = "No reflection available."
        
        st.markdown(f"<p class='small-font' style='text-align: center;'>{latest_reflection}</p>", unsafe_allow_html=True)

        st.markdown("<h2>Reflection History</h2>", unsafe_allow_html=True)
        if 'reflection' in filtered_df.columns and 'cumulative_reflection' in filtered_df.columns:
            reflection_df = filtered_df[['timestamp', 'decision', 'reflection', 'cumulative_reflection']].dropna(subset=['reflection'])
            if not reflection_df.empty:
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
