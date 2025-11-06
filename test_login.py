#!/usr/bin/env python3
"""
Test script to verify login functionality
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from werkzeug.security import check_password_hash
import sqlite3

def test_login():
    # Connect to database
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all users
    users = cursor.execute('SELECT * FROM users').fetchall()

    print("Users in database:")
    for user in users:
        print(f"ID: {user['id']}, Username: {user['username']}")

    # Test admin login with default password
    admin_user = cursor.execute('SELECT * FROM users WHERE username = ?', ('admin',)).fetchone()

    if admin_user:
        print(f"\nAdmin user found: {admin_user['username']}")
        print(f"Password hash: {admin_user['password'][:50]}...")

        # Test with default password 'test123'
        test_password = 'test123'
        is_valid = check_password_hash(admin_user['password'], test_password)
        print(f"Password 'test123' valid: {is_valid}")

        # Test with common passwords
        common_passwords = ['admin', 'password', '123456', 'admin123']
        for pwd in common_passwords:
            is_valid = check_password_hash(admin_user['password'], pwd)
            if is_valid:
                print(f"Password '{pwd}' is CORRECT!")
                break
    else:
        print("Admin user not found!")

    conn.close()

if __name__ == '__main__':
    test_login()
