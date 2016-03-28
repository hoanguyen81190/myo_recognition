# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 03:11:39 2016

@author: Hoa
"""

import mysql.connector
from mysql.connector import errorcode

DB_NAME = 'myo_features'

TABLES = {}
TABLES['myo_features'] = (
    "CREATE TABLE `myo_features` ("
    "  `sample_no` int(11) NOT NULL AUTO_INCREMENT,"
    "  `file_name` varchar(14) NOT NULL,"    
    "  `gesture_name` int(11) NOT NULL,"
    "  `gesture_number` int(11) NOT NULL,"
    "  `MAV` int(11),"
    "  `MAVR` int(11),"
    "  `ZC` int(11),"
    "  `WL` int(11),"
    "  `ACCC` int(11),"
    "  `SPM` int(11),"
    "  `SampEn` int(11),"
    "  PRIMARY KEY (`emp_no`)"
    ") ENGINE=InnoDB")

cnx = mysql.connector.connect(user='hoa')
cursor = cnx.cursor()

def create_database(cursor):
    try:
        cursor.execute(
            "CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(DB_NAME))
    except mysql.connector.Error as err:
        print("Failed creating database: {}".format(err))
        exit(1)
    
try: 
    cnx.database = DB_NAME
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_BAD_DB_ERROR:
        create_database(cursor)
        cnx.database = DB_NAME
    else:
        print(err)
        exit(1)
