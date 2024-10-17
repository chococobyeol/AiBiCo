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
     total_assets_krw REAL)
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
    st.set_page_config(layout="wide", page_title="비트코인 트레이딩 대시보드", initial_sidebar_state="collapsed")
    
    st.markdown("""
    <style>
    .big-font {font-size:20px !important; font-weight: bold;}
    .medium-font {font-size:16px !important;}
    .small-font {font-size:14px !important;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>비트코인 트레이딩 대시보드</h1>", unsafe_allow_html=True)

    df = get_trade_data()
    
    if len(df) > 0:
        # 성과 계산 부분 제거

        st.sidebar.header("데이터 필터")
        start_date = st.sidebar.date_input("시작 날짜", df['timestamp'].min().date())
        end_date = st.sidebar.date_input("종료 날짜", df['timestamp'].max().date())
        
        if start_date <= end_date:
            mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
            filtered_df = df.loc[mask]
        else:
            st.sidebar.error("Error: 종료 날짜는 시작 날짜 이후여야 합니다.")
            filtered_df = df

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.markdown("<p class='big-font'>총 거래 횟수</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='medium-font'>{len(filtered_df)}</p>", unsafe_allow_html=True)
        with col2:
            st.markdown("<p class='big-font'>성공률</p>", unsafe_allow_html=True)
            success_rate = (filtered_df['success'].sum() / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
            st.markdown(f"<p class='medium-font'>{success_rate:.2f}%</p>", unsafe_allow_html=True)
        with col3:
            st.markdown("<p class='big-font'>누적 수익률</p>", unsafe_allow_html=True)
            if 'total_profit' in filtered_df.columns and len(filtered_df) > 0:
                total_profit = filtered_df['total_profit'].iloc[0] * 100
                st.markdown(f"<p class='medium-font'>{total_profit:.2f}%</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='medium-font'>N/A</p>", unsafe_allow_html=True)
        with col4:
            st.markdown("<p class='big-font'>현재 BTC 가격</p>", unsafe_allow_html=True)
            current_price = filtered_df['btc_krw_price'].iloc[-1] if len(filtered_df) > 0 else 0
            st.markdown(f"<p class='medium-font'>₩{current_price:,.0f}</p>", unsafe_allow_html=True)
        with col5:
            st.markdown("<p class='big-font'>거래 기간</p>", unsafe_allow_html=True)
            if len(filtered_df) > 0:
                start_date = filtered_df['timestamp'].min().strftime('%Y-%m-%d')
                end_date = filtered_df['timestamp'].max().strftime('%Y-%m-%d')
                st.markdown(f"<p class='small-font'>{start_date} ~ {end_date}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='small-font'>데이터 없음</p>", unsafe_allow_html=True)
        with col6:
            st.markdown("<p class='big-font'>다음 거래 예정</p>", unsafe_allow_html=True)
            next_trade_time = get_next_trade_time()
            if next_trade_time:
                st.markdown(f"<p class='small-font'>{next_trade_time.strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='small-font'>정보 없음</p>", unsafe_allow_html=True)

        # 수익률 요약 테이블
        st.markdown("<h2 style='text-align: center;'>수익률 요약</h2>", unsafe_allow_html=True)
        if 'daily_profit' in filtered_df.columns and 'total_profit' in filtered_df.columns and len(filtered_df) > 0:
            latest_data = filtered_df.iloc[0]
            weekly_profit = filtered_df['daily_profit'].head(7).sum() if len(filtered_df) >= 7 else filtered_df['daily_profit'].sum()
            monthly_profit = filtered_df['daily_profit'].head(30).sum() if len(filtered_df) >= 30 else filtered_df['daily_profit'].sum()
            profit_summary = pd.DataFrame({
                '기간': ['일간', '주간', '월간', '누적'],
                '수익률': [
                    f"{latest_data['daily_profit']*100:.2f}%",
                    f"{weekly_profit*100:.2f}%",
                    f"{monthly_profit*100:.2f}%",
                    f"{latest_data['total_profit']*100:.2f}%"
                ]
            })
            st.table(profit_summary)
        else:
            st.info("수익률 데이터가 충분하지 않습니다.")

        st.markdown("<h2 style='text-align: center;'>최근 거래 내역</h2>", unsafe_allow_html=True)
        if len(filtered_df) > 0:
            df_display = filtered_df[['timestamp', 'decision', 'percentage', 'reason', 
                                      'btc_balance', 'btc_krw_value', 'krw_balance', 
                                      'total_assets_btc_formatted', 
                                      'btc_krw_price', 'success']].head(10)
            df_display['success'] = df_display['success'].map({1: '성공', 0: '실패'})
            st.dataframe(df_display, height=300)
        else:
            st.info("아직 거래 내역이 없습니다.")

        st.markdown("<h2 style='text-align: center;'>트레이딩 성과</h2>", unsafe_allow_html=True)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('누적 수익률', 'BTC 가격'))
        
        # 'total_profit' 컬럼이 있는지 확인하고, 없으면 계산
        if 'total_profit' not in filtered_df.columns:
            if 'daily_profit' in filtered_df.columns:
                filtered_df['total_profit'] = filtered_df['daily_profit'].cumsum()
            else:
                st.warning("수익률 데이터가 없습니다. 차트를 표시할 수 없습니다.")
                return

        fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['total_profit'], mode='lines', name='누적 수익률'), row=1, col=1)
        fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['btc_krw_price'], mode='lines', name='BTC 가격'), row=2, col=1)
        fig.update_layout(height=600, width=1000)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3 style='text-align: center;'>거래 결정 분포</h3>", unsafe_allow_html=True)
            decision_counts = filtered_df['decision'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=decision_counts.index, values=decision_counts.values)])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("<h3 style='text-align: center;'>거래 성공/실패 비율</h3>", unsafe_allow_html=True)
            success_counts = filtered_df['success'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=['성공', '실패'], values=[success_counts.get(1, 0), success_counts.get(0, 0)])])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<h3 style='text-align: center;'>BTC 잔고 변화</h3>", unsafe_allow_html=True)
        fig = go.Figure(data=[go.Scatter(x=filtered_df['timestamp'], y=filtered_df['btc_balance'], mode='lines')])
        fig.update_layout(title='BTC Balance Over Time', xaxis_title='Timestamp', yaxis_title='BTC Balance')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<h3 style='text-align: center;'>KRW 잔고 변화</h3>", unsafe_allow_html=True)
        fig = go.Figure(data=[go.Scatter(x=filtered_df['timestamp'], y=filtered_df['krw_balance'], mode='lines')])
        fig.update_layout(title='KRW Balance Over Time', xaxis_title='Timestamp', yaxis_title='KRW Balance')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<h3 style='text-align: center;'>BTC 평균 매수가 변화</h3>", unsafe_allow_html=True)
        fig = go.Figure(data=[go.Scatter(x=filtered_df['timestamp'], y=filtered_df['btc_avg_buy_price'], mode='lines')])
        fig.update_layout(title='BTC Average Buy Price Over Time', xaxis_title='Timestamp', yaxis_title='BTC Average Buy Price')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<h2 style='text-align: center;'>최근 반성 내용</h2>", unsafe_allow_html=True)
        if 'reflection' in filtered_df.columns and len(filtered_df) > 0:
            latest_reflection = filtered_df.loc[filtered_df['reflection'].notna(), 'reflection'].iloc[0] if not filtered_df['reflection'].isna().all() else "반성 내용이 없습니다."
        else:
            latest_reflection = "반성 내용이 없습니다."
        
        st.markdown(f"<p class='small-font'>{latest_reflection}</p>", unsafe_allow_html=True)

    else:
        st.warning("거래 데이터가 없습니다. 자동 거래 봇이 실행 중이지만 아직 거래가 이루어지지 않았거나, 거래 봇이 실행되지 않았을 수 있습니다.")
        
        # 빈 데이터에 대한 처리
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.markdown("<p class='big-font'>총 거래 횟수</p>", unsafe_allow_html=True)
            st.markdown("<p class='medium-font'>0</p>", unsafe_allow_html=True)
        with col2:
            st.markdown("<p class='big-font'>성공률</p>", unsafe_allow_html=True)
            st.markdown("<p class='medium-font'>0.00%</p>", unsafe_allow_html=True)
        with col3:
            st.markdown("<p class='big-font'>누적 수익률</p>", unsafe_allow_html=True)
            st.markdown("<p class='medium-font'>N/A</p>", unsafe_allow_html=True)
        with col4:
            st.markdown("<p class='big-font'>현재 BTC 가격</p>", unsafe_allow_html=True)
            st.markdown("<p class='medium-font'>데이터 없음</p>", unsafe_allow_html=True)
        with col5:
            st.markdown("<p class='big-font'>거래 기간</p>", unsafe_allow_html=True)
            st.markdown("<p class='small-font'>데이터 없음</p>", unsafe_allow_html=True)
        with col6:
            st.markdown("<p class='big-font'>다음 거래 예정</p>", unsafe_allow_html=True)
            next_trade_time = get_next_trade_time()
            if next_trade_time:
                st.markdown(f"<p class='small-font'>{next_trade_time.strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='small-font'>정보 없음</p>", unsafe_allow_html=True)

        st.info("거래 데이터가 생성되면 여기에 차트와 그래프가 표시됩니다.")

if __name__ == "__main__":
    main()
