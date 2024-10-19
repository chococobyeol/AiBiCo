import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_database():
    conn = sqlite3.connect('trading_history.db')
    cursor = conn.cursor()

    try:
        # 1. 'daily_profit'을 'trade_profit'으로 변경
        cursor.execute("PRAGMA table_info(trades)")
        columns = [column[1] for column in cursor.fetchall()]

        if 'daily_profit' in columns and 'trade_profit' not in columns:
            logging.info("Renaming 'daily_profit' to 'trade_profit'")
            cursor.execute("ALTER TABLE trades RENAME COLUMN daily_profit TO trade_profit")
        elif 'daily_profit' in columns and 'trade_profit' in columns:
            logging.info("Both 'daily_profit' and 'trade_profit' exist. Keeping 'trade_profit' and dropping 'daily_profit'")
            cursor.execute("ALTER TABLE trades DROP COLUMN daily_profit")
        elif 'trade_profit' in columns:
            logging.info("'trade_profit' column already exists. No changes needed.")
        else:
            logging.info("Neither 'daily_profit' nor 'trade_profit' found. Adding 'trade_profit' column.")
            cursor.execute("ALTER TABLE trades ADD COLUMN trade_profit REAL")

        conn.commit()
        logging.info("Database update completed successfully")

    except sqlite3.Error as e:
        logging.error(f"An error occurred: {e}")
        conn.rollback()

    finally:
        conn.close()

if __name__ == "__main__":
    update_database()