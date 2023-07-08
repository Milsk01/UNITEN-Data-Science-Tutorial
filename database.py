import sqlite3

conn = sqlite3.connect('test.db')

print("Opened database successfully")

conn.execute('''CREATE TABLE RESULTS
        (ID INTEGER PRIMARY KEY AUTOINCREMENT,
         GLUCOSE INT NOT NULL,
         BMI INT NOT NULL,
         AGE INT NOT NULL,
         PREDICTION char(50) NOT NULL,
         CONFIDENCE real NOT NULL,
         CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL)
             ;''')

print("Table created successfully")

conn.close()