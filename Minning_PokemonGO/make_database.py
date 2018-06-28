# Creating the database for holding tweets
import sqlite3
import os

if os.path.isfile('tweets.db'):
    os.remove('tweets.db')
    
conn = sqlite3.connect('tweets.db')

curs = conn.cursor()
curs.execute("CREATE TABLE tweets (id integer primary key, \
                                   tid integer unique,\
                                   username text, \
                                   created_at datatime, \
                                   lang text, \
                                   content text, \
                                   location text, \
                                   source text)")

conn.commit()

conn.close()

