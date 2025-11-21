import sqlite3

DATABASE = "ai_data.db"

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Create all your tables here (copy everything from `index.py`)
    ...
    
    conn.commit()
    conn.close()
