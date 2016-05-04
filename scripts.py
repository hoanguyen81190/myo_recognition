# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 21:57:59 2016

@author: Hoa
"""
import pandas
import numpy
from datetime import datetime
import matplotlib.pyplot as plt
import itertools
import glob, os
import math

#
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.cross_validation import LeaveOneOut
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.learning_curve import learning_curve

#

#matplotlib.style.use('ggplot')

names_cols = ['timestamp', 'package_number', 'gesture_name', 'gesture_number', 
              's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
              's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16']
names_cols_short = ['timestamp', 'package_number', 'gesture_name', 'gesture_number', 
              's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']             

db_cols = ['file_name', 'gesture_name', 'gesture_number', 'window_number', 
           'MAV_s0', 'MAV_s1', 'MAV_s2', 'MAV_s3', 'MAV_s4', 'MAV_s5', 'MAV_s6', 'MAV_s7',
           'MAVR_s05', 'MAVR_s16', 'MAVR_s27', 'MAVR_s38', 
           'ZC_s0', 'ZC_s1', 'ZC_s2', 'ZC_s3', 'ZC_s4', 'ZC_s5', 'ZC_s6', 'ZC_s7', 
           'WL_s0', 'WL_s1', 'WL_s2', 'WL_s3', 'WL_s4', 'WL_s5', 'WL_s6', 'WL_s7', 
           'ACCC_s0', 'ACCC_s1', 'ACCC_s2', 'ACCC_s3', 'ACCC_s4', 'ACCC_s5', 'ACCC_s6', 'ACCC_s7', 
           'SPM_s0', 'SPM_s1', 'SPM_s2', 'SPM_s3', 'SPM_s4', 'SPM_s5', 'SPM_s6', 'SPM_s7', 
           'SampEn_s0', 'SampEn_s1', 'SampEn_s2', 'SampEn_s3', 'SampEn_s4', 'SampEn_s5', 'SampEn_s6', 'SampEn_s7']

GESTURE_LENGTH = 240 #240 points, 5ms/point => 1200 ms
WINDOW_SIZE = 40 #200ms
INCREMENT = 20 #100ms
EPSILON = 0.015 #V

df = pandas.read_csv('C:/Users/Hoa/thesis/data/test', sep=',', names=names_cols_short, skiprows=1)

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
    return (basis[pos:pos + window] for pos in range(int(0), len(basis), int(window)))

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

'''xyz = df.copy()
xyz['timestamp'] = xyz['timestamp'].apply(time2Date)
xyz = xyz.set_index('timestamp', drop=False)
abc = df.copy()'''  



def lengthOfSamples(sequence):
    return sequence.groupby(['gesture_name', 'gesture_number']).size()
    
#####################################
#       feature extraction          #
#####################################
def featureCalculator(sequence, func):
    gesture_groups = sequence.groupby(['gesture_name', 'gesture_number'])
    return gesture_groups.apply(func)

def extractAllSample(seq, funcs, cols, file_name, ending):
    df = pandas.DataFrame(columns = ['file_name', 'gesture_name', 'gesture_number', 'window_number'] + cols)
    groups = seq.groupby(['gesture_name', 'gesture_number'])
    for group, sample in groups:
        windows = extractFeatures1(sample, funcs)
        for i in range(len(windows)):
            temp_df = pandas.DataFrame([[file_name, ending+str(group[0]), group[1], i] + windows[i]],
                                       columns = ['file_name', 'gesture_name', 'gesture_number', 'window_number'] + cols)
            df = df.append(temp_df)
    return df
            
def extractFiles(mydir, ending, funcs, cols, feature_set):
    FEATURE_FILE = 'myo_features_' + mydir
    s_file = GIT_DIR+"/" + FEATURE_FILE + "_" + feature_set
    def helper():
        for file in glob.glob("*"+ending):
            sq_file = pandas.read_csv(file, sep=',', names=names_cols, skiprows=1)
            df = extractAllSample(sq_file, funcs, cols, file, ending)
            if not os.path.isfile(s_file):
                df.to_csv(s_file, header=True, index = False)
            else:
                df.to_csv(s_file, mode = 'a', header = False, index = False)
    dirs = [d for d in os.listdir(GIT_DIR+mydir) if os.path.isdir(os.path.join(GIT_DIR+mydir, d))]
    if len(dirs) != 0:
        for d in dirs:
            os.chdir(GIT_DIR+mydir+"/"+d)
            helper()
    else:
        os.chdir(GIT_DIR+mydir)
        helper()
            
#to simplify the process, gesture length is set to fix size: 1200 ms
def extractFeatures(sequence, funcs):
    #festures
    def applyFunctions(val):
        return [f(val) for f in funcs]
    return [applyFunctions(i) for i in overlappedWindow(sequence[:GESTURE_LENGTH], WINDOW_SIZE, INCREMENT)]
 
def extractFeatures1(sequence, funcs):
    def applyFunctions(val):
        ret_dict = []
        for f in funcs:
            ret_dict += f(val)
        return ret_dict
    return [applyFunctions(i) for i in overlappedWindow(sequence[:GESTURE_LENGTH], WINDOW_SIZE, INCREMENT)]
    
    
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
    
def mav(x):
    return numpy.mean(numpy.abs(x)) if not(x.empty) else 0

def MAVcal(x):
    return [mav(x['s1']), mav(x['s2']), 
            mav(x['s3']), mav(x['s4']),
            mav(x['s5']), mav(x['s6']), 
            mav(x['s7']), mav(x['s8'])]  

#mean absolute value ratio: the ratio of MAV between channel
#only use: s1/s5, s2/s6, s3/s7, s4/s8
def MAVRcal(x):
    ret = []
    for i in range(8):
        for j in range(i, 8):
            if i != j:
                v = (mav(x['s'+str(i+1)])/mav(x['s'+str(j+1)])) if mav(x['s'+str(j+1)]) != 0 else 0
                ret += [v]
    return ret
#    m_s1 = numpy.mean(numpy.abs(x['s1']))
#    m_s2 = numpy.mean(numpy.abs(x['s2']))
#    m_s3 = numpy.mean(numpy.abs(x['s3']))
#    m_s4 = numpy.mean(numpy.abs(x['s4']))
#    m_s5 = numpy.mean(numpy.abs(x['s5']))
#    m_s6 = numpy.mean(numpy.abs(x['s6']))
#    m_s7 = numpy.mean(numpy.abs(x['s7']))
#    m_s8 = numpy.mean(numpy.abs(x['s8']))
#    return [m_s1/m_s5 if m_s5 != 0 else 0, m_s2/m_s6 if m_s6 != 0 else 0,
#            m_s3/m_s7 if m_s7 != 0 else 0, m_s4/m_s8 if m_s8 != 0 else 0]

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
        return len(numpy.where(numpy.diff(s))[0])/(len(xx)-1) if (len(xx) - 1) != 0 else 0
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

#ACCC
def cc(x, y, t=INCREMENT/2, f=1):
    """
    Cross correlation, formula from:
    Application of Autocorrelation and Crosscorrelation
    Analyses in Human Movement
    and Rehabilitation Research
    """
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    N = len(x)
    top = (numpy.array([(x[i]- x.mean())*(y[(N-t+i)%N]-y.mean()) for i in range(N)]).sum())
    bottom = math.sqrt((numpy.array([(x[i]-x.mean())**2 for i in range(N)]).sum())*(numpy.array([(y[i]-y.mean())**2 for i in range(N)]).sum()))
    return top/bottom
    
def ac(x, t=INCREMENT, f=1):
    return cc(x, x, t, f)
    
def acCal(seq):
    return [ac(seq['s1']), ac(seq['s2']), 
            ac(seq['s3']), ac(seq['s4']),
            ac(seq['s5']), ac(seq['s6']), 
            ac(seq['s7']), ac(seq['s8'])]
            
def ccCal(x):
    ret = []
    for i in range(8):
        for j in range(i, 8):
            if i != j:
                v = cc(x['s'+str(i+1)], x['s'+str(j+1)])
                ret += [v]
    return ret

#spectral power magnitudes
def SPMcal(seq):
    #divide into 4 equal bandwidths
    #performing Fast Fourier Transform
    #taking the average
    def spm(x):
        return numpy.average(numpy.fft.fft(x))
    return [spm(seq['s1']), spm(seq['s2']), 
            spm(seq['s3']), spm(seq['s4']),
            spm(seq['s5']), spm(seq['s6']), 
            spm(seq['s7']), spm(seq['s8'])]
    
#sample entropy
#def sampEncalculator(sequence):
#    return
    
#sample AR model
def ARcal(sequence):
    return
    
info_cols = ['file_name', 'gesture_name', 'gesture_number', 'window_number']
tmd_funcs = [MAVcal, MAVRcal, zeroCrossingRate, WLcal]
tmd_cols = ['MAV_s0', 'MAV_s1', 'MAV_s2', 'MAV_s3', 'MAV_s4', 'MAV_s5', 'MAV_s6', 'MAV_s7',
 'MAVR_s12', 'MAVR_s13', 'MAVR_s14', 'MAVR_s15', 'MAVR_s16', 'MAVR_s17', 'MAVR_s18', 
 'MAVR_s23', 'MAVR_s24', 'MAVR_s25', 'MAVR_s26', 'MAVR_s27', 'MAVR_s28', 
 'MAVR_s34', 'MAVR_s35', 'MAVR_s36', 'MAVR_s37', 'MAVR_s38', 
 'MAVR_s45', 'MAVR_s46', 'MAVR_s47', 'MAVR_s48', 
 'MAVR_s56', 'MAVR_s57', 'MAVR_s58', 
 'MAVR_s67', 'MAVR_s68', 
 'MAVR_s78',
 'ZC_s0', 'ZC_s1', 'ZC_s2', 'ZC_s3', 'ZC_s4', 'ZC_s5', 'ZC_s6', 'ZC_s7',
 'WL_s0', 'WL_s1', 'WL_s2', 'WL_s3', 'WL_s4', 'WL_s5', 'WL_s6', 'WL_s7']
tmd_lengths = 52
 
accc_cols = ['AC_s0', 'AC_s1', 'AC_s2', 'AC_s3', 'AC_s4', 'AC_s5', 'AC_s6', 'AC_s7'
             , 'CC_s01', 'CC_s02', 'CC_s03', 'CC_s04', 'CC_s05', 'CC_s06', 'CC_s07'
             , 'CC_s12', 'CC_s13', 'CC_s14', 'CC_s15', 'CC_s16', 'CC_s17'
             , 'CC_s23', 'CC_s24', 'CC_s25', 'CC_s26', 'CC_s27'
             , 'CC_s34', 'CC_s35', 'CC_s36', 'CC_s37'
             , 'CC_s45', 'CC_s46', 'CC_s47'
             , 'CC_s56', 'CC_s57'
             , 'CC_s67']
accc_funcs = [acCal, ccCal]

spm_funcs = [SPMcal]
spm_cols = ['SPM_s0', 'SPM_s1', 'SPM_s2', 'SPM_s3', 'SPM_s4', 'SPM_s5', 'SPM_s6', 'SPM_s7']
sampen_cols = ['SampEn_s0', 'SampEn_s1', 'SampEn_s2', 'SampEn_s3', 'SampEn_s4', 'SampEn_s5', 'SampEn_s6', 'SampEn_s7']



######################################
#       helper                       #
######################################
def getSample(sequence, name, number):
    return sequence[(sequence['gesture_name'] == name) & 
                    (sequence['gesture_number'] == number)]

#def transformSensorValues(sequence):
#    for index, row in sequence.iterrows():
#        new_row = pandas.DataFrame([[row['timestamp'], row['package_number'], row['gesture_name'], row['gesture_number'], 
#              row['s9'], row['s10'], row['s11'], row['s12'], row['s13'], row['s14'], row['s15'], row['s16'],
#              None, None, None, None, None, None, None, None]], columns=names_cols)
#        sequence = sequence.append(new_row)
#    sequence.drop(sequence[['s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16']], axis=1, inplace=True)
#    return sequence.sort_values(['timestamp'])

def transformSensorValues(seq):
    seq = pandas.concat([seq, seq]).sort_index().reset_index(drop=True)
    for i in range(1, len(seq)+1, 2):
        seq.loc[i, 's1'] = seq.loc[i, 's9']
        seq.loc[i, 's2'] = seq.loc[i, 's10']
        seq.loc[i, 's3'] = seq.loc[i, 's11']
        seq.loc[i, 's4'] = seq.loc[i, 's12']
        seq.loc[i, 's5'] = seq.loc[i, 's13']
        seq.loc[i, 's6'] = seq.loc[i, 's14']
        seq.loc[i, 's7'] = seq.loc[i, 's15']
        seq.loc[i, 's8'] = seq.loc[i, 's16']
    seq.drop(['s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16'], axis=1, inplace=True)
 
   
def saveFile(sequence, file_name):
    sequence.to_csv(GIT_DIR+file_name, sep=',', index=False)
    
NUMBER_GROUP = "_number_group"
TAPPING_GROUP = "_tapping_group"
SEMANTIC_GROUP = "_semantic_group"
WRIST = "_wrist"
STAFF_DIR = "staff" 
GIT_DIR = "C:/Users/Hoa/thesis/data/"  
DATA_DIR = "E:/thesis/data/" 
EXPO_DIR = 'expo_day'
EXPO_DIRS = ["expo_day/mobile_1",
             "expo_day/mobile_2",
             "expo_day/mobile_3",
             "expo_day/mobile_4"]
model = "OVR_RFC.pkl"
MODEL_DIR = "C:/Users/Hoa/thesis/models"
NEW_DIR = "new"
HAI_DIR = "hai"
HOA_DIR = "hoa"
HIEN_DIR = "hien"
GROUP_NAMES = [NUMBER_GROUP, TAPPING_GROUP, WRIST]
def readFiles(mydir, ending):
    os.chdir(mydir)
    return glob.glob("*"+ending)
    
def hammingFiles(mydir, ending):
    def f(x):
        x = x.apply(numpy.abs)
        x = x.apply(math.sqrt)
        x = pandas.pandas.rolling_window(x, window=WINDOW_SIZE, win_type='hamming', mean=True)
    os.chdir(GIT_DIR+mydir)
    sdir = GIT_DIR+mydir+"_hamming"
    if not os.path.exists(sdir):
        os.makedirs(sdir)
    for file in glob.glob("*"+ending):
        seq = pandas.read_csv(file, sep=',', names=names_cols, skiprows=1)
        seq['s1'] = f(seq['s1'])
        seq['s2'] = f(seq['s2'])
        seq['s3'] = f(seq['s3'])
        seq['s4'] = f(seq['s4'])
        seq['s5'] = f(seq['s5'])
        seq['s6'] = f(seq['s6'])
        seq['s7'] = f(seq['s7'])
        seq['s8'] = f(seq['s8'])
        seq.to_csv(sdir+"/"+file, sep=',', index=False)
    
def transformFiles(mydir, ending):
    os.chdir(DATA_DIR+mydir)
    sdir = GIT_DIR+mydir
    if not os.path.exists(sdir):
        os.makedirs(sdir)
    for file in glob.glob("*"+ending):
        seq = pandas.read_csv(file, sep=',', names=names_cols, skiprows=1)
        seq = pandas.concat([seq, seq]).sort_index().reset_index(drop=True)
        for i in range(1, len(seq)+1, 2):
            seq.loc[i, 's1'] = seq.loc[i, 's9']
            seq.loc[i, 's2'] = seq.loc[i, 's10']
            seq.loc[i, 's3'] = seq.loc[i, 's11']
            seq.loc[i, 's4'] = seq.loc[i, 's12']
            seq.loc[i, 's5'] = seq.loc[i, 's13']
            seq.loc[i, 's6'] = seq.loc[i, 's14']
            seq.loc[i, 's7'] = seq.loc[i, 's15']
            seq.loc[i, 's8'] = seq.loc[i, 's16']
        seq.drop(['s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16'], axis=1, inplace=True)
 
        seq.to_csv(sdir+"/"+file, sep=',', index=False)

def nameDict(x):
        return {
            "_number_group0" : 1, 
            "_number_group1" : 2,
            "_number_group2" : 3,
            "_number_group3" : 4,
            "_number_group4" : 5,
            "_tapping_group0" : 6,
            "_tapping_group1" : 7,
            "_tapping_group2" : 8,
            "_tapping_group3" : 9,
            "_semantic_group0" : 10,
            "_semantic_group1" : 11,
            "_semantic_group2" : 12,
            "_semantic_group3" : 13,
            "_semantic_group4" : 14,
            "_wrist0" : 15,
            "_number_group0.0" : 1, 
            "_number_group1.0" : 2,
            "_number_group2.0" : 3,
            "_number_group3.0" : 4,
            "_number_group4.0" : 5,
            "_tapping_group0.0" : 6,
            "_tapping_group1.0" : 7,
            "_tapping_group2.0" : 8,
            "_tapping_group3.0" : 9,
            "_semantic_group0.0" : 10,
            "_semantic_group1.0" : 11,
            "_semantic_group2.0" : 12,
            "_semantic_group3.0" : 13,
            "_semantic_group4.0" : 14,
            "_wrist0" : 15
        }[x]

######################################
#               visualization        #
######################################
#ts.plot()

def getXsYs(seq, lengths):
    groups = seq.groupby(['file_name', 'gesture_name', 'gesture_number'])
    Ys = []
    Xs = []
    bucket = numpy.zeros(lengths)
    for g, data in groups:
        Ys += [g[1]]
        dat = data.as_matrix(columns=data.columns[4:])
        for c in range(12 - len(dat)):
            dat = numpy.append(dat, [bucket], axis=0)
        dat = list(itertools.chain(*dat))
        Xs += [dat]
    return (Xs, Ys)

#myo_features_TMD = pandas.read_csv('C:/Users/Hoa/thesis/data/myo_features_TMD', names=info_cols+tmd_cols, skiprows=1)

#fs = getSample(abc, 0, 0)

#
#for g in GROUP_NAMES:
#    extractFiles(EXPO_DIR, g, tmd_funcs, tmd_cols, "TMD")
#for g in GROUP_NAMES:
#    extractFiles(STAFF_DIR, g, tmd_funcs, tmd_cols, "TMD")

#for g in GROUP_NAMES:
#    extractFiles(GIT_DIR + STAFF_DIR, g, spm_funcs, spm_cols, "SPM")

#for d in EXPO_DIRS:
#    for g in GROUP_NAMES:
#        extractFiles(GIT_DIR + d, g, spm_funcs, spm_cols, "SPM")


#
#staff

myo_features_new_TMD = pandas.read_csv('C:/Users/Hoa/thesis/data/myo_features_new_TMD', names=info_cols+tmd_cols, skiprows=1)
myo_features_new_TMD = myo_features_new_TMD.drop_duplicates()
myo_features_new_TMD['gesture_name'] = myo_features_new_TMD['gesture_name'].apply(nameDict)

tmdv_new = getXsYs(myo_features_new_TMD, len(tmd_cols))

myo_features_hai_TMD = pandas.read_csv('C:/Users/Hoa/thesis/data/myo_features_hai_TMD', names=info_cols+tmd_cols, skiprows=1)
myo_features_hai_TMD = myo_features_hai_TMD.drop_duplicates()
myo_features_hai_TMD['gesture_name'] = myo_features_hai_TMD['gesture_name'].apply(nameDict)

tmdv_hai = getXsYs(myo_features_hai_TMD, len(tmd_cols))

myo_features_hoa_TMD = pandas.read_csv('C:/Users/Hoa/thesis/data/myo_features_hoa_TMD', names=info_cols+tmd_cols, skiprows=1)
myo_features_hoa_TMD = myo_features_hoa_TMD.drop_duplicates()
myo_features_hoa_TMD['gesture_name'] = myo_features_hoa_TMD['gesture_name'].apply(nameDict)

tmdv_hoa = getXsYs(myo_features_hoa_TMD, len(tmd_cols))

myo_features_hien_TMD = pandas.read_csv('C:/Users/Hoa/thesis/data/myo_features_hien_TMD', names=info_cols+tmd_cols, skiprows=1)
myo_features_hien_TMD = myo_features_hien_TMD.drop_duplicates()
myo_features_hien_TMD['gesture_name'] = myo_features_hien_TMD['gesture_name'].apply(nameDict)

tmdv_hien = getXsYs(myo_features_hien_TMD, len(tmd_cols))

Xs_tmd = tmdv_new[0] + tmdv_hai[0] + tmdv_hoa[0] + tmdv_hien[0]
Ys_tmd = tmdv_new[1] + tmdv_hai[1] + tmdv_hoa[1] + tmdv_hien[1]

x_hoa = tmdv_hoa[0]

y_hoa = tmdv_hoa[1]

rfc_hoa = RandomForestClassifier(n_estimators=10)

rfc_hoa.fit(x_hoa, y_hoa)

rfc_scores_hoa = cross_validation.cross_val_score(rfc_hoa, x_hoa, y_hoa, cv=5)

# Create classifiers
lr = LogisticRegression()
gnb = GaussianNB()
svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier(n_estimators=100)
classif = OneVsRestClassifier(SVC(kernel='linear'))
#classif.fit(X, Y)

X_tmd = tmdv_hoa[0] + tmdv_hien[0] + tmdv_new[0]
y_tmd = tmdv_hoa[1] + tmdv_hien[1] + tmdv_new[1]
X_train, X_test, y_train, y_test = train_test_split(X_tmd, y_tmd, test_size=0.33, random_state=42)
scores = {}

#ACCC
myo_features_new_ACCC = pandas.read_csv('C:/Users/Hoa/thesis/data/myo_features_new_ACCC', names=info_cols+accc_cols, skiprows=1)
myo_features_new_ACCC = myo_features_new_ACCC.drop_duplicates()
myo_features_new_ACCC['gesture_name'] = myo_features_new_ACCC['gesture_name'].apply(nameDict)

accc_new = getXsYs(myo_features_new_ACCC, len(accc_cols))

myo_features_hai_ACCC = pandas.read_csv('C:/Users/Hoa/thesis/data/myo_features_hai_ACCC', names=info_cols+accc_cols, skiprows=1)
myo_features_hai_ACCC = myo_features_hai_ACCC.drop_duplicates()
myo_features_hai_ACCC['gesture_name'] = myo_features_hai_ACCC['gesture_name'].apply(nameDict)

accc_hai = getXsYs(myo_features_hai_ACCC, len(accc_cols))

myo_features_hoa_ACCC = pandas.read_csv('C:/Users/Hoa/thesis/data/myo_features_hoa_ACCC', names=info_cols+accc_cols, skiprows=1)
myo_features_hoa_ACCC['gesture_name'] = myo_features_hoa_ACCC['gesture_name'].apply(nameDict)
myo_features_hoa_ACCC = myo_features_hoa_ACCC.drop_duplicates()

accc_hoa = getXsYs(myo_features_hoa_ACCC, len(accc_cols))

myo_features_hien_ACCC = pandas.read_csv('C:/Users/Hoa/thesis/data/myo_features_hien_ACCC', names=info_cols+accc_cols, skiprows=1)
myo_features_hien_ACCC = myo_features_hien_ACCC.drop_duplicates()
myo_features_hien_ACCC['gesture_name'] = myo_features_hien_ACCC['gesture_name'].apply(nameDict)

accc_hien = getXsYs(myo_features_hien_ACCC, len(accc_cols))

X_accc = accc_hoa[0] + accc_hien[0] + accc_new[0]
y_accc = accc_hoa[1] + accc_hien[1] + accc_new[1]
X_train, X_test, y_train, y_test = train_test_split(X_accc, y_accc, test_size=0.33, random_state=42)
accc_scores = {}

myo_features_new_SPM = pandas.read_csv('C:/Users/Hoa/thesis/data/myo_features_new_SPM', names=info_cols+spm_cols, skiprows=1, converters={'SPM_s0': complex, 'SPM_s1': complex, 'SPM_s2': complex, 'SPM_s3': complex, 'SPM_s4': complex, 'SPM_s5': complex, 'SPM_s6': complex, 'SPM_s7': complex})
myo_features_new_SPM = myo_features_new_SPM.drop_duplicates()
myo_features_new_SPM['gesture_name'] = myo_features_new_SPM['gesture_name'].apply(nameDict)

spm_new = getXsYs(myo_features_new_SPM, len(spm_cols))

myo_features_hai_SPM = pandas.read_csv('C:/Users/Hoa/thesis/data/myo_features_hai_SPM', names=info_cols+spm_cols, skiprows=1, converters={'SPM_s0': complex, 'SPM_s1': complex, 'SPM_s2': complex, 'SPM_s3': complex, 'SPM_s4': complex, 'SPM_s5': complex, 'SPM_s6': complex, 'SPM_s7': complex})
myo_features_hai_SPM = myo_features_hai_SPM.drop_duplicates()
myo_features_hai_SPM['gesture_name'] = myo_features_hai_SPM['gesture_name'].apply(nameDict)

spm_hai = getXsYs(myo_features_hai_SPM, len(spm_cols))

myo_features_hoa_SPM = pandas.read_csv('C:/Users/Hoa/thesis/data/myo_features_hoa_SPM', names=info_cols+spm_cols, skiprows=1, converters={'SPM_s0': complex, 'SPM_s1': complex, 'SPM_s2': complex, 'SPM_s3': complex, 'SPM_s4': complex, 'SPM_s5': complex, 'SPM_s6': complex, 'SPM_s7': complex})
myo_features_hoa_SPM['gesture_name'] = myo_features_hoa_SPM['gesture_name'].apply(nameDict)
myo_features_hoa_SPM = myo_features_hoa_SPM.drop_duplicates()

spm_hoa = getXsYs(myo_features_hoa_SPM, len(spm_cols))

myo_features_hien_SPM = pandas.read_csv('C:/Users/Hoa/thesis/data/myo_features_hien_SPM', names=info_cols+spm_cols, skiprows=1, converters={'SPM_s0': complex, 'SPM_s1': complex, 'SPM_s2': complex, 'SPM_s3': complex, 'SPM_s4': complex, 'SPM_s5': complex, 'SPM_s6': complex, 'SPM_s7': complex})
myo_features_hien_SPM = myo_features_hien_SPM.drop_duplicates()
myo_features_hien_SPM['gesture_name'] = myo_features_hien_SPM['gesture_name'].apply(nameDict)

spm_hien = getXsYs(myo_features_hien_SPM, len(spm_cols))

#for clf, name in [(lr, 'Logistic'),
#                  (gnb, 'Naive Bayes'),
#                  (svc, 'Support Vector Classification'),
#                  (rfc, 'Random Forest')]:
#    ovr_clf = OneVsRestClassifier(clf)
#    acc = cross_validation.cross_val_score(ovr_clf, X_tmd, Y_tmd, cv=5)
#    scores[name] = acc
#for clf, name in [(lr, 'Logistic'),
#                  (gnb, 'Naive Bayes'),
#                  (svc, 'Support Vector Classification'),
#                  (rfc, 'Random Forest')]:
#    clf.fit(X_train, y_train)
#    y_pred = clf.predict(X_test)
#    acc = accuracy_score(y_test, y_pred)
#    scores[name] = acc

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=numpy.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = numpy.mean(train_scores, axis=1)
    train_scores_std = numpy.std(train_scores, axis=1)
    test_scores_mean = numpy.mean(test_scores, axis=1)
    test_scores_std = numpy.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


'''myo_features_staff_TMD = pandas.read_csv('C:/Users/Hoa/thesis/data/myo_features_staff_TMD', names=info_cols+tmd_cols, skiprows=1)
myo_features_staff_TMD = myo_features_staff_TMD.drop_duplicates()
myo_features_staff_TMD['gesture_name'] = myo_features_staff_TMD['gesture_name'].apply(nameDict)

myo_features_staff_SPM = pandas.read_csv('C:/Users/Hoa/thesis/data/myo_features_staff_SPM', names=info_cols+spm_cols, skiprows=1)
myo_features_staff_SPM = myo_features_staff_SPM.drop_duplicates()
myo_features_staff_SPM['gesture_name'] = myo_features_staff_SPM['gesture_name'].apply(nameDict)


tmdv_staff = getXsYs(myo_features_staff_TMD, len(tmd_cols))
Xs_tmd = tmdv_staff[0]
Ys_tmd = tmdv_staff[1]

spmv_staff = getXsYs(myo_features_staff_SPM, len(spm_cols))
Xs_spm = spmv_staff[0]
Ys_spm = spmv_staff[1]

#expo
myo_features_expo_TMD = pandas.read_csv('C:/Users/Hoa/thesis/data/myo_features_expo_day_TMD', names=info_cols+tmd_cols, skiprows=1)
myo_features_expo_TMD = myo_features_expo_TMD.drop_duplicates()
myo_features_expo_TMD['gesture_name'] = myo_features_expo_TMD['gesture_name'].apply(nameDict)

myo_features_expo_SPM = pandas.read_csv('C:/Users/Hoa/thesis/data/myo_features_expo_day_SPM', names=info_cols+spm_cols, skiprows=1)
myo_features_expo_SPM = myo_features_expo_SPM.drop_duplicates()
myo_features_expo_SPM['gesture_name'] = myo_features_expo_SPM['gesture_name'].apply(nameDict)

tmdv_expo = getXsYs(myo_features_expo_TMD, len(tmd_cols))
Xe_tmd = tmdv_expo[0]
Ye_tmd = tmdv_expo[1]

spmv_expo = getXsYs(myo_features_expo_SPM, len(spm_cols))
Xe_spm = spmv_expo[0]
Ye_spm = spmv_expo[1]

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
confusion_matrix(y_true, y_pred)
accuracy_score(y_test, y_pred)
#
# machine learning
#
#array([ 0.11356467,  0.13015873,  0.12063492,  0.15555556,  0.2031746 ,
# 0.20127796,  0.13782051,  0.09003215,  0.10610932,  0.08064516])

lin_clf = svm.LinearSVC()
lin_clf.fit(X, Y)
lin_scores = cross_validation.cross_val_score(lin_clf, X, Y, cv=5)
#array([ 0.12222222,  0.10047847,  0.15015974,  0.11661342,  0.1088    ])

nb_clf = KNeighborsClassifier(n_neighbors=3)
nb_clf.fit(X, Y)
nb_scores = cross_validation.cross_val_score(nb_clf, X, Y, cv=5)
#array([ 0.07619048,  0.07336523,  0.06549521,  0.0686901 ,  0.0832    ])
#
#
'''
'''
number_0: 1
number_1: 2
number_2: 3
number_3: 4
number_4: 5
tapping_0: 6
tapping_1: 7
tapping_2: 8
tapping_3: 9
semantic_0: 10
semantic_1: 11
semantic_2: 12
semantic_3: 13
semantic_4: 14
wrist: 15
'''

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