# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 03:51:52 2016

@author: Hoa
"""

import sqlite3

DB_NAME = 'myo_features.db'
TABLE_NAME = 'myo_features'
FEATURES = ['MAV', 'MAVR', 'ZC', 'WL', 'ACCC', 'SPM', 'SampEn']

def createDatabase():
    conn = sqlite3.connect(DB_NAME)

    c = conn.cursor()
    
    column_list = ""

    for f in FEATURES:
        for i in range(8):
            column_list += ", " + f + "_" + "s" + str(i)
    
    create_query = "CREATE TABLE " + TABLE_NAME + " (sample_no, file_name, gesture_name, gesture_number, window_number" + column_list + ")"
                   
    c.execute(create_query)
    conn.commit()
    conn.close()
    
def connectDatabase():
    return sqlite3.connect(DB_NAME)
    
def closeDatabase(conn):
    conn.commit()
    conn.close()
       
def insertDatabase(cursor, mav, mavr, zc, wl, accc, spm, sampen):
    mav_str = mavr_str = zc_str = wl_str = accc_str = spm_str = sampen_str = ""
    for i in range(8):
        if mav is not None:
            mav_str += ", " + mav[i]
        else:
            mav_str += ", " 
    for i in range(8):
        if mavr is not None:
            mavr_str += ", " + mavr[i]
        else:
            mavr_str += ", " 
    for i in range(8):
        if zc is not None:
            zc_str += ", " + zc[i]
        else:
            zc_str += ", " 
    for i in range(8):
        if wl is not None:
            wl_str += ", " + wl[i]
        else:
            wl_str += ", " 
    for i in range(8):
        if accc is not None:
            accc_str += ", " + accc[i]
        else:
            accc_str += ", " 
    for i in range(8):
        if spm is not None:
            spm_str += ", " + spm[i]
        else:
            spm_str += ", " 
    for i in range(8):
        if sampen is not None:
            sampen_str += ", " + sampen[i]
        else:
            sampen_str += ", " 
    insert_query = "INSERT INTO myo_features VALUES(" 
    + mav_str + mavr_str + zc_str + wl_str + accc_str + spm_str + sampen_str + ")"
    cursor.execute(insert_query)