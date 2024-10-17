import sqlite3
from tabulate import tabulate

def check_database():
    conn = sqlite3.connect('trading_history.db')
    cursor = conn.cursor()

    # 테이블 구조 확인
    cursor.execute("PRAGMA table_info(trades)")
    columns = cursor.fetchall()
    print("Table Structure:")
    print(tabulate(columns, headers=["ID", "Name", "Type", "NotNull", "DefaultValue", "PK"]))
    print("\n")

    # 데이터 조회
    cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10")
    rows = cursor.fetchall()
    
    if rows:
        headers = [description[0] for description in cursor.description]
        print("Latest 10 Trades:")
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        print("No data found in the trades table.")

    conn.close()

if __name__ == "__main__":
    check_database()
