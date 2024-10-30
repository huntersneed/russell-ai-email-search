import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_unique_constraint():
    try:
        # Load environment variables
        load_dotenv()
        
        # Construct database connection string from individual components
        db_username = os.getenv('DB_USERNAME')
        db_password = os.getenv('DB_PASSWORD')
        rds_endpoint = os.getenv('RDS_ENDPOINT')
        rds_port = os.getenv('RDS_PORT')
        db_name = os.getenv('DB_NAME')
        
        # Create the connection string
        db_connection = f"postgresql://{db_username}:{db_password}@{rds_endpoint}:{rds_port}/{db_name}"
        
        # Create database engine
        engine = create_engine(db_connection)
        
        # SQL commands to check if constraint exists and add if it doesn't
        check_constraint_sql = """
        SELECT COUNT(*)
        FROM pg_constraint 
        WHERE conname = 'unique_collection_id_document';
        """
        
        add_constraint_sql = """
        ALTER TABLE langchain_pg_embedding 
        ADD CONSTRAINT unique_collection_id_document 
        UNIQUE (collection_id, document);
        """
        
        # Execute the commands
        with engine.connect() as connection:
            # Check if constraint exists
            result = connection.execute(text(check_constraint_sql))
            constraint_exists = result.scalar()
            
            if constraint_exists:
                logger.info("Unique constraint already exists")
            else:
                logger.info("Adding unique constraint to langchain_pg_embedding table...")
                connection.execute(text(add_constraint_sql))
                connection.commit()
                logger.info("Successfully added unique constraint")
            
    except Exception as e:
        logger.error(f"Error adding unique constraint: {str(e)}")
        raise

if __name__ == "__main__":
    add_unique_constraint()