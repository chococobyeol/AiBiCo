import sqlite3
import os
from datetime import datetime

def print_trades():
    # 데이터베이스 파일 경로
    db_path = os.path.join(os.path.dirname(__file__), 'trading_history.db')
    
    try:
        # 데이터베이스 연결
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # trades 테이블의 모든 데이터 조회
        cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC")
        trades = cursor.fetchall()
        
        # 컬럼 이름 가져오기
        cursor.execute("PRAGMA table_info(trades)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # 컬럼 이름 출력
        print("\t".join(columns))
        print("-" * 100)
        
        # 각 거래 데이터 출력
        for trade in trades:
            formatted_trade = []
            for value in trade:
                if isinstance(value, float):
                    formatted_value = f"{value:.8f}"
                elif isinstance(value, (int, str)):
                    formatted_value = str(value)
                elif value is None:
                    formatted_value = "None"
                else:
                    formatted_value = str(value)
                formatted_trade.append(formatted_value)
            
            print("\t".join(formatted_trade))
        
    except sqlite3.Error as e:
        print(f"데이터베이스 오류: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    print_trades()
