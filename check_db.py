#!/usr/bin/env python3
"""
Check database contents and test login
"""
import sqlite3
from werkzeug.security import check_password_hash

def check_database():
    print("=== Checking users.db ===")
    try:
        conn = sqlite3.connect('users.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check users table
        users = cursor.execute('SELECT id, username, password FROM users').fetchall()
        print(f"Found {len(users)} users:")
        for user in users:
            print(f"  ID: {user['id']}, Username: {user['username']}")
            print(f"  Password hash: {user['password'][:50]}...")

            # Test common passwords
            test_passwords = ['test123', 'admin', 'password', '123456']
            for pwd in test_passwords:
                if check_password_hash(user['password'], pwd):
                    print(f"  ✅ Password '{pwd}' works!")
                    break

        # Check predictions table
        predictions = cursor.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
        print(f"Total predictions: {predictions}")

        conn.close()

    except Exception as e:
        print(f"Error checking users.db: {e}")

    print("\n=== Checking app/users.db ===")
    try:
        conn = sqlite3.connect('app/users.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        users = cursor.execute('SELECT id, username, password FROM users').fetchall()
        print(f"Found {len(users)} users in app/users.db:")
        for user in users:
            print(f"  ID: {user['id']}, Username: {user['username']}")

        conn.close()

    except Exception as e:
        print(f"Error checking app/users.db: {e}")

if __name__ == '__main__':
    check_database()
