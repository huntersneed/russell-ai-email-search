import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.getenv('RDS_ENDPOINT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('RDS_PORT', 5432)

def connect_to_db():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        return conn
    except psycopg2.Error as e:
        print(f"Unable to connect to the database: {e}")
        return None

def list_tables(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = cur.fetchall()
            print("Available tables:")
            for table in tables:
                print(table[0])
    except psycopg2.Error as e:
        print(f"Error listing tables: {e}")

def clear_table(conn, table_name):
    try:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {table_name}")
            conn.commit()
            print(f"Table '{table_name}' has been cleared successfully.")
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error clearing table {table_name}: {e}")

if __name__ == "__main__":
    conn = connect_to_db()
    if conn:
        list_tables(conn)
        table_name = input("\nEnter the name of the table you want to clear: ")
        confirm = input(f"Are you sure you want to clear all contents of '{table_name}'? This action cannot be undone. (y/n): ")
        if confirm.lower() == 'y':
            clear_table(conn, table_name)
        else:
            print("Operation cancelled.")
        conn.close()
    else:
        print("Failed to connect to the database.")