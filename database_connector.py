# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 19:24:46 2016

@author: Hoa
"""

from pymongo import MongoClient

client = MongoClient()
client = MongoClient('localhost', 27017)

myo_db = client['myo-features-database']
td_collection = myo_db['time-domain-collection']
fd_collection = myo_db['frequency-domain-collection']

def insertTDFeatureDatabase(*mavs, *mavrs, *zcs, *wls, gesture_type):
    post = {"MAV" : *mavs,
            "MAVR": *mavrs,
            "ZC": *zcs,
            "gesture_type": gesture_type}
    posts = myo_db.posts
    post_id = posts.insert_one(post).inserted_id
    return post_id
    