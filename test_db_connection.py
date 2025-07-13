from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
db_url = os.getenv("SUPABASE_DB_URL")

try:
    engine = create_engine(db_url)
    conn = engine.connect()
    print("✅ Connection successful!")
    conn.close()
except Exception as e:
    print("❌ Connection failed:", e)
