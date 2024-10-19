import os
import sys
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get database connection details from environment variables
db_host = os.getenv('RDS_ENDPOINT')
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USERNAME')
db_password = os.getenv('DB_PASSWORD')
db_port = os.getenv('RDS_PORT', '5432')  # Default to 5432 if not specified

def check_db_connection():
    try:
        # Attempt to establish a connection to the database
        conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port
        )
        
        # Create a cursor
        cur = conn.cursor()
        
        # Execute a simple query
        cur.execute('SELECT version();')
        
        # Fetch the result
        db_version = cur.fetchone()
        
        # Close the cursor and connection
        cur.close()
        conn.close()
        
        print(f"Successfully connected to the database.")
        print(f"PostgreSQL version: {db_version[0]}")
        
        # Check if pgvector extension is installed
        conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port
        )
        cur = conn.cursor()
        cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector');")
        pgvector_installed = cur.fetchone()[0]
        cur.close()
        conn.close()
        
        if pgvector_installed:
            print("pgvector extension is installed.")
        else:
            print("pgvector extension is NOT installed.")
        
    except psycopg2.Error as e:
        print(f"Unable to connect to the database.")
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    check_db_connection()
