# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 21:57:59 2016

@author: Hoa
"""
import pandas
import numpy
from datetime import datetime

names_cols = ['timestamp', 'package_number', 'gesture_name', 'gesture_number', 
              's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
              's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16']
df = pandas.read_csv('C:/Users/Hoa/workspace/Myo/rsc/test.csv', sep=',', names=names_cols, skiprows=1)

def time2Date(val):
#    seconds = val/1000
#    sub_seconds = (val % 1000.0) / 1000.0
#    date = datetime.fromtimestamp(seconds + sub_seconds)
    date = datetime.fromtimestamp(val/1000).replace(microsecond = (val % 1000) * 1000)    
    return pandas.to_datetime(date.strftime('%Y-%m-%d %H:%M:%S.%f'))
    

def rollBy(what,basis,window,func, step, *args,**kwargs):
    #note that basis must be sorted in order for this to work properly     
    indexed_what = pandas.Series(what.values,index=basis.values)
    def applyToWindow(val):
        # using slice_indexer rather that what.loc [val:val+window] allows
        # window limits that are not specifically in the index
        indexer = indexed_what.index.slice_indexer(val,val+window,step)
        chunk = indexed_what[indexer]
        return func(chunk,*args,**kwargs)
    rolled = basis.apply(applyToWindow)
    return rolled
    
def f(what, basis, window, func):
    def applyToWindow(val):
        ser = what[(basis >= val) & (basis < val + window)]
        return func(ser)
    return basis.apply(applyToWindow)
    
xyz = df.copy()
xyz['timestamp'] = xyz['timestamp'].apply(time2Date)
xyz = xyz.set_index('timestamp', drop=False)
abc = df.copy()
#xyz.resample('80ms', how='sum')

def testfunction1(data):
    if data is not None: 
        return (numpy.sum(data['s1']+data['s9']))
    else:
        return None
    
def rollBy2(what,basis,window,func,*args,**kwargs):
    #note that basis must be sorted in order for this to work properly
    windows_min = basis.min()
    windows_max = basis.max()
    window_starts = numpy.arange(windows_min, windows_max, window)
    window_starts = pandas.Series(window_starts, index = window_starts)
    indexed_what = pandas.Series(what.values,index=basis.values)
    def applyToWindow(val):
        # using slice_indexer rather that what.loc [val:val+window] allows
        # window limits that are not specifically in the index
        indexer = indexed_what.index.slice_indexer(val,val+window,1)
        chunk = indexed_what[indexer]
        return func(chunk,*args,**kwargs)
    rolled = window_starts.apply(applyToWindow)
    return rolled
    
    
'''
B       business day frequency
C       custom business day frequency (experimental)
D       calendar day frequency
W       weekly frequency
M       month end frequency
BM      business month end frequency
CBM     custom business month end frequency
MS      month start frequency
BMS     business month start frequency
CBMS    custom business month start frequency
Q       quarter end frequency
BQ      business quarter endfrequency
QS      quarter start frequency
BQS     business quarter start frequency
A       year end frequency
BA      business year end frequency
AS      year start frequency
BAS     business year start frequency
BH      business hour frequency
H       hourly frequency
T       minutely frequency
S       secondly frequency
L       milliseonds
U       microseconds
N       nanoseconds
'''