# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 21:57:59 2016

@author: Hoa
"""
import pandas
import numpy
from datetime import datetime
import matplotlib
matplotlib.style.use('ggplot')

names_cols = ['timestamp', 'package_number', 'gesture_name', 'gesture_number', 
              's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
              's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16']

GESTURE_LENGTH = 150
WINDOW_SIZE = 20 #200ms

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
    
def adjacentWindow(basis, window):
    #adjacent disjoint windows
    return (basis[pos:pos + window] for pos in range(0, len(basis), window))

def overlappedWindow(basis, window, increment):
    #adjacent disjoint windows
    return (basis[pos:pos + window] for pos in range(0, len(basis), window-increment))

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
  
xyz = df.copy()
xyz['timestamp'] = xyz['timestamp'].apply(time2Date)
xyz = xyz.set_index('timestamp', drop=False)
abc = df.copy()  



def lengthOfSamples(sequence):
    return sequence.groupby(['gesture_name', 'gesture_number']).size()
    
#####################################
#       feature extraction          #
#####################################
def featureCalculator(sequence, func):
    gesture_groups = sequence.groupby(['gesture_name', 'gesture_number'])
    return gesture_groups.apply(func)
    
#def MAVcalculator1(x):
#    return [[((sum(i['s1'])+sum(i['s9']))/(len(i['s1'])+len(i['s9']))),
#             ((sum(i['s2'])+sum(i['s10']))/(len(i['s2'])+len(i['s10']))),
#             ((sum(i['s3'])+sum(i['s11']))/(len(i['s3'])+len(i['s11']))),
#             ((sum(i['s4'])+sum(i['s12']))/(len(i['s4'])+len(i['s12']))),
#             ((sum(i['s5'])+sum(i['s13']))/(len(i['s5'])+len(i['s13']))),
#             ((sum(i['s6'])+sum(i['s14']))/(len(i['s6'])+len(i['s14']))),
#             ((sum(i['s7'])+sum(i['s15']))/(len(i['s7'])+len(i['s15']))),
#             ((sum(i['s8'])+sum(i['s16']))/(len(i['s8'])+len(i['s16'])))]
#            for i in adjacentDisjointWindow(x[:160], WINDOW_SIZE)]

#MAV
def MAVcalculator(x):
    return [[numpy.mean(numpy.abs(i['s1'])), numpy.mean(numpy.abs(i['s1'])), numpy.mean(numpy.abs(i['s1'])), numpy.mean(numpy.abs(i['s1'])),
            numpy.mean(numpy.abs(i['s1'])), numpy.mean(numpy.abs(i['s1'])), numpy.mean(numpy.abs(i['s1'])), numpy.mean(numpy.abs(i['s1']))]
            for i in adjacentWindow(x[:160], WINDOW_SIZE*2)]  

#zero crossing
def cross(sequence, cross=0, direction='cross'):
    """
    Given a Series returns all the index values where the data values equal 
    the 'cross' value. 

    Direction can be 'rising' (for rising edge), 'falling' (for only falling 
    edge), or 'cross' for both edges
    """
    # Find if values are above or bellow yvalue crossing:
    above=sequence.values > cross
    below=numpy.logical_not(above)
    left_shifted_above = above[1:]
    left_shifted_below = below[1:]
    x_crossings = []
    # Find indexes on left side of crossing point
    if direction == 'rising':
        idxs = (left_shifted_above & below[0:-1]).nonzero()[0]
    elif direction == 'falling':
        idxs = (left_shifted_below & above[0:-1]).nonzero()[0]
    else:
        rising = left_shifted_above & below[0:-1]
        falling = left_shifted_below & above[0:-1]
        idxs = (rising | falling).nonzero()[0]

    # Calculate x crossings with interpolation using formula for a line:
    x1 = sequence.index.values[idxs]
    x2 = sequence.index.values[idxs+1]
    y1 = sequence.values[idxs]
    y2 = sequence.values[idxs+1]
    x_crossings = (cross-y1)*(x2-x1)/(y2-y1) + x1

    return x_crossings
######################################
#       helper                       #
######################################
def getSample(sequence, name, number):
    return sequence[(sequence['gesture_name'] == name) & 
                    (sequence['gesture_number'] == number)]

def transformSensorValues(sequence):
    for index, row in sequence.iterrows():
        new_row = pandas.DataFrame([[row['timestamp'], row['package_number'], row['gesture_name'], row['gesture_number'], 
              row['s9'], row['s10'], row['s11'], row['s12'], row['s13'], row['s14'], row['s15'], row['s16'],
              None, None, None, None, None, None, None, None]], columns=names_cols)
        sequence = sequence.append(new_row)
    sequence.drop(sequence[['s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16']], axis=1, inplace=True)
    return sequence.sort_values(['timestamp'])
    
######################################
#               visualization        #
######################################
#ts.plot()

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