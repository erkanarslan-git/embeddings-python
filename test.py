import sqlite3
import numpy as np

DB_NAME = "advertisements.db"

conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

cursor.execute("SELECT embedding FROM embeddings")
rows = cursor.fetchall()

for idx, row in enumerate(rows, start=1):
    embedding = np.frombuffer(row[0], dtype=np.float32)
    print(f"Reklam {idx}: Embedding boyutu: {len(embedding)}")

conn.close()




