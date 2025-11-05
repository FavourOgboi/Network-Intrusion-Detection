import sqlite3
import os
import hashlib

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "users.db")

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user'
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            input_data TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def register_user(username, password, role="user"):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                  (username, hash_password(password), role))
        conn.commit()
        return True, "Registration successful."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()

def verify_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT password_hash, role FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if row and row[0] == hash_password(password):
        return True, row[1]
    return False, None

def save_prediction(user_id, input_data, prediction, confidence):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO predictions (user_id, input_data, prediction, confidence)
            VALUES (?, ?, ?, ?)
        """, (user_id, input_data, prediction, confidence))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error saving prediction: {e}")
    finally:
        conn.close()
