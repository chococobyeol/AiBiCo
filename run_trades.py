import sqlite3
from autotrade import get_recent_trades

# 데이터베이스 연결 설정
conn = sqlite3.connect('trading_history.db')

# 함수 호출 및 결과 출력
trades = get_recent_trades(conn)
print(trades)

# 연결 닫기
conn.close()