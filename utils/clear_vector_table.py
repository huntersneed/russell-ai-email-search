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

def clear_tables(conn, table_names, use_cascade=False):
    try:
        with conn.cursor() as cur:
            if use_cascade:
                cur.execute(f"TRUNCATE TABLE {', '.join(table_names)} CASCADE")
                print(f"Tables {', '.join(table_names)} have been cleared successfully with CASCADE.")
            else:
                for table in reversed(table_names):  # Clear in reverse order to handle foreign key constraints
                    cur.execute(f"TRUNCATE TABLE {table}")
                    print(f"Table '{table}' has been cleared successfully.")
            conn.commit()
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error clearing tables: {e}")

def clear_pgvector_tables():
    try:
        conn = connect_to_db()
        if conn:
            with conn.cursor() as cur:
                # Drop tables completely (more thorough than TRUNCATE)
                cur.execute("""
                    DROP TABLE IF EXISTS langchain_pg_embedding CASCADE;
                    DROP TABLE IF EXISTS langchain_pg_collection CASCADE;
                """)
                conn.commit()
            print("Successfully dropped PGVector tables")
            conn.close()
        else:
            print("Failed to connect to the database.")
    except Exception as e:
        print(f"Error dropping tables: {e}")

if __name__ == "__main__":
    clear_pgvector_tables()
