# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 21:57:59 2016

@author: Hoa
"""
import pandas
import numpy
from datetime import datetime
import matplotlib
import glob, os
matplotlib.style.use('ggplot')

names_cols = ['timestamp', 'package_number', 'gesture_name', 'gesture_number', 
              's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
              's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16']
names_cols_short = ['timestamp', 'package_number', 'gesture_name', 'gesture_number', 
              's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']             

GESTURE_LENGTH = 150
WINDOW_SIZE = 40 #200ms
INCREMENT = 20 #100ms
EPSILON = 0.015 #V

df = pandas.read_csv('C:/Users/Hoa/thesis/data/test_transform', sep=',', names=names_cols_short, skiprows=1)

def time2Date(val):
#    seconds = val/1000
#    sub_seconds = (val % 1000.0) / 1000.0
#    date = datetime.fromtimestamp(seconds + sub_seconds)
    date = datetime.fromtimestamp(val/1000).replace(microsecond = (val % 1000) * 1000)    
    return pandas.to_datetime(date.strftime('%Y-%m-%d %H:%M:%S.%f'))

#####################################
#       motion detection            #
#####################################



#####################################
#       data segmentation           #
#####################################    

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

def extractFeatures(sequence, *funcs):
    #festures
    def applyFunctions(x):
        for f in funcs:        
            yield f(x)
    return [i.apply(applyFunctions) for i in overlappedWindow(sequence, WINDOW_SIZE, INCREMENT)]
        
    
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
def MAVcal(x):
    return [numpy.mean(numpy.abs(x['s1'])), numpy.mean(numpy.abs(x['s2'])), 
            numpy.mean(numpy.abs(x['s3'])), numpy.mean(numpy.abs(x['s4'])),
            numpy.mean(numpy.abs(x['s5'])), numpy.mean(numpy.abs(x['s6'])), 
            numpy.mean(numpy.abs(x['s7'])), numpy.mean(numpy.abs(x['s8']))]  

#mean absolute value ratio: the ratio of MAV between channel
#only use: s1/s5, s2/s6, s3/s7, s4/s8
def MAVRcal(x):
    return [numpy.mean(numpy.abs(x['s1'])) / numpy.mean(numpy.abs(x['s5'])), 
            numpy.mean(numpy.abs(x['s2'])) / numpy.mean(numpy.abs(x['s6'])),
            numpy.mean(numpy.abs(x['s3'])) / numpy.mean(numpy.abs(x['s7'])), 
            numpy.mean(numpy.abs(x['s4'])) / numpy.mean(numpy.abs(x['s8']))]

#MAVS
#def MAVScalculator(x):
#    fs = numpy.gradient(x) #calculate first derivative
#    #then count the number of sign change
#    return ....

#zero crossing
#or number of zero crossing
def zeroCrossingRate(x):  
    def zeroCal(xx):    
        s = numpy.sign(xx)  
        s[s==0] = -1     # replace zeros with -1  
        #the delta should be greater than EPSILON, but in this case, 
        #values are integer so we don't need to check this condition.
        return len(numpy.where(numpy.diff(s))[0])/(len(xx)-1)
    return [zeroCal(x['s1']), zeroCal(x['s2']), 
            zeroCal(x['s3']), zeroCal(x['s4']),
            zeroCal(x['s5']), zeroCal(x['s6']), 
            zeroCal(x['s7']), zeroCal(x['s8'])]  

#waveform length: the cumulative length of the EMG signal within the analysis window 
def WLcal(x):
    return [sum(numpy.diff(x['s1'])), sum(numpy.diff(x['s2'])), 
            sum(numpy.diff(x['s3'])), sum(numpy.diff(x['s4'])),
            sum(numpy.diff(x['s5'])), sum(numpy.diff(x['s6'])), 
            sum(numpy.diff(x['s7'])), sum(numpy.diff(x['s8']))]  


#autocorrelation
def autoCorr(x):
    result = numpy.correlate(x, x, mode='full')
    return result[result.size/2:]
    
def acf(x, length=20):
    return numpy.array([1]+[numpy.corrcoef(x[:-i], x[i:]) \
        for i in range(1, length)])

def autocorr(x, t=1):
    return numpy.corrcoef(numpy.array([x[0:len(x)-t], x[t:len(x)]]))

def autocorr1(sequence):
    corr = numpy.correlate(sequence, sequence, mode='same')
    N = len(corr)
    half = corr[N//2:]
    lengths = range(N, N//2, -1)
    half /= lengths
    half /= half[0]
    return half

#spectral power magnitudes
def SPMcalculator(sequence):
    #divide into 4 equal bandwidths
    #performing Fast Fourier Transform
    #taking the average
    return [numpy.average(numpy.fft.fft(i))
        for i in adjacentWindow(sequence, len(sequence)/4)]
    
#sample entropy
def sampEncalculator(sequence):
    return
    
#sample AR model
def ARcalculator(sequence):
    return

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
    
def saveFile(sequence, file_name):
    sequence.to_csv(GIT_DIR+file_name, sep=',', index=False)
    
NUMBER_GROUP = "_number_group"
TAPPING_GROUP = "_tapping_group"
SEMANTIC_GROUP = "_semantic_group"
WRIST = "_wrist"
STAFF_DIR = "staff" 
GIT_DIR = "C:/Users/Hoa/thesis/data/"  
DATA_DIR = "E:/thesis/data/" 
EXPO_DIRS = ["expo_day/mobile_1",
             "expo_day/mobile_2",
             "expo_day/mobile_3",
             "expo_day/mobile_4"]
GROUP_NAMES = [NUMBER_GROUP, TAPPING_GROUP, SEMANTIC_GROUP, WRIST]
def readFiles(mydir, ending):
    os.chdir(mydir)
    return glob.glob("*"+ending)
    
def transformFiles(mydir, ending):
    os.chdir(DATA_DIR+mydir)
    sdir = GIT_DIR+mydir
    if not os.path.exists(sdir):
        os.makedirs(sdir)
    for file in glob.glob("*"+ending):
        sq_file = pandas.read_csv(file, sep=',', names=names_cols, skiprows=1)
        sq_file = transformSensorValues(sq_file)
        sq_file.to_csv(sdir+"/"+file, sep=',', index=False)
######################################
#               visualization        #
######################################
#ts.plot()


fs = getSample(abc, 0, 0)
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