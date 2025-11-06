#!/usr/bin/env python3
"""
Test login with existing credentials
"""
import sqlite3
from werkzeug.security import check_password_hash

def test_login():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get admin user
    user = cursor.execute('SELECT * FROM users WHERE username = ?', ('admin',)).fetchone()

    if user:
        print(f"Found user: {user['username']}")
        print(f"User ID: {user['id']}")

        # Test passwords
        test_passwords = ['test123', 'admin', 'password', '123456', 'admin123']

        for password in test_passwords:
            if check_password_hash(user['password'], password):
                print(f"SUCCESS: Password '{password}' works for user '{user['username']}'")
                return password

        print("No matching password found")
    else:
        print("Admin user not found")

    conn.close()
    return None

if __name__ == '__main__':
    working_password = test_login()
    if working_password:
        print(f"\nUse these credentials:")
        print(f"Username: admin")
        print(f"Password: {working_password}")
    else:
        print("\nNo working password found. Need to reset database.")
