import os
import psycopg2
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.getenv('RDS_ENDPOINT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('RDS_PORT', 5432)

TABLE_NAME = "n8n_vectors"

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

def view_table_contents(conn, table_name):
    try:
        with conn.cursor() as cur:
            # Get column names
            cur.execute(f"SELECT * FROM {table_name} LIMIT 0")
            column_names = [desc[0] for desc in cur.description]
            
            # Fetch all rows
            cur.execute(f"SELECT * FROM {table_name}")
            rows = cur.fetchall()
            
            # Print column names
            print(f"\nTable: {table_name}")
            print("Columns:", ", ".join(column_names))
            print("\nTable contents:")
            
            # Print each row
            for row in rows:
                formatted_row = []
                for item in row:
                    if isinstance(item, np.ndarray):
                        # Format vector as a short representation
                        formatted_item = f"Vector({len(item)} dims)"
                    else:
                        formatted_item = str(item)
                    formatted_row.append(formatted_item)
                print(", ".join(formatted_row))
            
            print(f"\nTotal rows: {len(rows)}")
    except psycopg2.Error as e:
        print(f"Error querying the table {table_name}: {e}")

if __name__ == "__main__":
    conn = connect_to_db()
    if conn:
        list_tables(conn)
        table_name = input("\nEnter the name of the table you want to view: ")
        view_table_contents(conn, table_name)
        conn.close()
    else:
        print("Failed to connect to the database.")