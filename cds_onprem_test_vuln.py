import os
import sqlite3


def run_backup(user_input):
    os.system("tar -czf backup.tar.gz " + user_input)


def get_user(username):
    conn = sqlite3.connect("app.db")
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    return conn.execute(query).fetchall()
