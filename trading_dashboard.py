# type: ignore
import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 데이터베이스 연결 함수
def get_db_connection():
    conn = sqlite3.connect('trading_history.db')
    return conn

# 거래 데이터 가져오기
def get_trade_data():
    conn = get_db_connection()
    query = "SELECT * FROM trades ORDER BY timestamp DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # timestamp 열을 datetime 형식으로 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    logging.info(f"Query result columns: {df.columns}")
    logging.info(f"Query result shape: {df.shape}")
    
    return df

# 성과 계산 함수
def calculate_performance(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['profit'] = df['btc_krw_price'].pct_change()
    df['cumulative_profit'] = (1 + df['profit']).cumprod() - 1
    return df

# 메인 대시보드 함수
def main():
    st.set_page_config(
        layout="wide",
        page_title="비트코인 트레이딩 대시보드",
        initial_sidebar_state="collapsed"  # 이 줄을 추가합니다
    )
    
    # CSS를 사용하여 글자 크기와 스타일 조정
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:16px !important;
    }
    .small-font {
        font-size:14px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>비트코인 트레이딩 대시보드</h1>", unsafe_allow_html=True)

    # 데이터 로드
    df = get_trade_data()
    df = calculate_performance(df)

    # 사이드바
    st.sidebar.header("데이터 필터")
    start_date = st.sidebar.date_input("시작 날짜", df['timestamp'].min().date())
    end_date = st.sidebar.date_input("종료 날짜", df['timestamp'].max().date())
    
    if start_date <= end_date:
        mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
        filtered_df = df.loc[mask]
    else:
        st.sidebar.error("Error: 종료 날짜는 시작 날짜 이후여야 합니다.")
        filtered_df = df

    # 메인 지표
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("<p class='big-font'>총 거래 횟수</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='medium-font'>{len(filtered_df)}</p>", unsafe_allow_html=True)
    with col2:
        st.markdown("<p class='big-font'>성공률</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='medium-font'>{(filtered_df['success'].mean() * 100):.2f}%</p>", unsafe_allow_html=True)
    with col3:
        st.markdown("<p class='big-font'>총 수익률</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='medium-font'>{(filtered_df['cumulative_profit'].iloc[-1] * 100):.2f}%</p>", unsafe_allow_html=True)
    with col4:
        st.markdown("<p class='big-font'>현재 BTC 가격</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='medium-font'>₩{filtered_df['btc_krw_price'].iloc[-1]:,.0f}</p>", unsafe_allow_html=True)
    with col5:
        st.markdown("<p class='big-font'>거래 기간</p>", unsafe_allow_html=True)
        start_date = filtered_df['timestamp'].min().strftime('%Y-%m-%d')
        end_date = filtered_df['timestamp'].max().strftime('%Y-%m-%d')
        st.markdown(f"<p class='small-font'>{start_date} ~ {end_date}</p>", unsafe_allow_html=True)

    # 최근 거래 내역
    st.markdown("<h2 style='text-align: center;'>최근 거래 내역</h2>", unsafe_allow_html=True)
    st.dataframe(filtered_df[['timestamp', 'decision', 'percentage', 'reason', 'btc_balance', 'krw_balance', 'btc_krw_price', 'success']].head(10), height=300)

    # 차트
    st.markdown("<h2 style='text-align: center;'>트레이딩 성과</h2>", unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('누적 수익률', 'BTC 가격'))
    fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['cumulative_profit'], mode='lines', name='누적 수익률'), row=1, col=1)
    fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['btc_krw_price'], mode='lines', name='BTC 가격'), row=2, col=1)
    fig.update_layout(height=600, width=1000)
    st.plotly_chart(fig, use_container_width=True)

    # 거래 결정 분포와 성공/실패 비율
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

    # BTC 잔고 변화
    st.markdown("<h3 style='text-align: center;'>BTC 잔고 변화</h3>", unsafe_allow_html=True)
    fig = go.Figure(data=[go.Scatter(x=filtered_df['timestamp'], y=filtered_df['btc_balance'], mode='lines')])
    fig.update_layout(title='BTC Balance Over Time', xaxis_title='Timestamp', yaxis_title='BTC Balance')
    st.plotly_chart(fig, use_container_width=True)

    # KRW 잔고 변화
    st.markdown("<h3 style='text-align: center;'>KRW 잔고 변화</h3>", unsafe_allow_html=True)
    fig = go.Figure(data=[go.Scatter(x=filtered_df['timestamp'], y=filtered_df['krw_balance'], mode='lines')])
    fig.update_layout(title='KRW Balance Over Time', xaxis_title='Timestamp', yaxis_title='KRW Balance')
    st.plotly_chart(fig, use_container_width=True)

    # BTC 평균 매수가 변화
    st.markdown("<h3 style='text-align: center;'>BTC 평균 매수가 변화</h3>", unsafe_allow_html=True)
    fig = go.Figure(data=[go.Scatter(x=filtered_df['timestamp'], y=filtered_df['btc_avg_buy_price'], mode='lines')])
    fig.update_layout(title='BTC Average Buy Price Over Time', xaxis_title='Timestamp', yaxis_title='BTC Average Buy Price')
    st.plotly_chart(fig, use_container_width=True)

    # 최근 반성 내용
    st.markdown("<h2 style='text-align: center;'>최근 반성 내용</h2>", unsafe_allow_html=True)
    if 'reflection' in filtered_df.columns:
        latest_reflection = filtered_df.loc[filtered_df['reflection'].notna(), 'reflection'].iloc[0] if not filtered_df['reflection'].isna().all() else "반성 내용이 없습니다."
    else:
        latest_reflection = "반성 내용이 없습니다. ('reflection' 열이 존재하지 않습니다)"
    
    st.markdown(f"<p class='small-font'>{latest_reflection}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
