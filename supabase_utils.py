import os
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from dotenv import load_dotenv

load_dotenv()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

def get_engine():
    """Creates and returns a SQLAlchemy engine for Supabase."""
    return create_engine(SUPABASE_DB_URL)

def upload_file_to_supabase(file, table_name):
    """Uploads an Excel/CSV file as a table to Supabase."""
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            return False, "❌ Unsupported file type"

        engine = get_engine()
        df.to_sql(table_name, con=engine, if_exists="replace", index=False)
        
        schemas = get_all_table_schemas()
        return True, f"✅ File uploaded successfully as table '{table_name}'"
    except Exception as e:
        return False, f"❌ Upload failed: {str(e)}"

def list_tables():
    """Returns a list of all table names in the Supabase database."""
    try:
        engine = get_engine()
        inspector = inspect(engine)
        return inspector.get_table_names()
    except Exception as e:
        return []

def delete_table(table_name):
    """Deletes a specific table from Supabase."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        return True, f"✅ Table '{table_name}' deleted"
    except Exception as e:
        return False, f"❌ Deletion failed: {str(e)}"


def get_all_table_schemas():
    """Fetches schema (column names and types) for all tables in the DB."""
    try:
        engine = get_engine()
        inspector = inspect(engine)
        schemas = {}
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            schemas[table_name] = [
                {"name": col["name"], "type": str(col["type"])} for col in columns
            ]
        return schemas
    except Exception as e:
        return {}