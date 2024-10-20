import sqlite3

def remove_profit_columns():
    # 데이터베이스 연결
    conn = sqlite3.connect('trading_history.db')
    cursor = conn.cursor()

    # 기존 테이블의 구조 확인
    cursor.execute("PRAGMA table_info(trades)")
    columns = [column[1] for column in cursor.fetchall()]

    # 삭제할 열 목록
    columns_to_remove = ['daily_profit', 'total_profit', 'adjusted_profit', 'twr', 'mwr', 'trade_profit']

    # 새로운 테이블 구조 생성
    new_columns = [col for col in columns if col not in columns_to_remove]
    new_columns_sql = ', '.join(new_columns)

    # 임시 테이블 생성 및 데이터 복사
    cursor.execute(f"""
    CREATE TABLE trades_temp AS
    SELECT {new_columns_sql}
    FROM trades
    """)

    # 기존 테이블 삭제
    cursor.execute("DROP TABLE trades")

    # 임시 테이블 이름 변경
    cursor.execute("ALTER TABLE trades_temp RENAME TO trades")

    # 변경사항 저장 및 연결 종료
    conn.commit()
    conn.close()

    print("Profit columns have been successfully removed from the trades table.")

if __name__ == "__main__":
    remove_profit_columns()
