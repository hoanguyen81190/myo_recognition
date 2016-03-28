# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 03:51:52 2016

@author: Hoa
"""

import sqlite3

DB_NAME = 'myo_features.db'
TABLE_NAME = 'myo_features'

def createDatabase():
    conn = sqlite3.connect(DB_NAME)

    c = conn.cursor()

    c.execute('''
            CREATE TABLE myo_features
            (sample_no, file_name, gesture_name, gesture_number,
            MAV, MAVR, ZC, WL, ACCC, SPM, SampEn)''')
    conn.commit()
    conn.close()
    
def connectDatabase():
    return sqlite3.connect(DB_NAME)
    
def closeDatabase(conn):
    conn.commit()
    conn.close()
       
def insertDatabase(cursor, *args):
    insert_query = "INSERT INTO myo_features VALUES()"   
    cursor.execute(insert_query)