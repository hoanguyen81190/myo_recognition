# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 21:57:59 2016

@author: Hoa
"""
import pandas
import numpy
from datetime import datetime
import matplotlib
import itertools
import glob, os

#
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.cross_validation import LeaveOneOut
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
#

matplotlib.style.use('ggplot')

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

df = pandas.read_csv('C:/Users/Hoa/thesis/data/test', sep=',', names=names_cols, skiprows=1)

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


#autocorrelation
def autoCorr(x):
    result = numpy.correlate(x, x, mode='full')
    return result[int(result.size/2):]
    
#def acf(x, length=20):
#    return numpy.array([1]+[numpy.corrcoef(x[:-i], x[i:]) \
#        for i in range(1, length)])

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
    
def acf(series):
    n = len(series)
    data = numpy.asarray(series)
    mean = numpy.mean(data)
    c0 = numpy.sum((data - mean) ** 2) / float(n)

    def r(h):
        acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
        return round(acf_lag, 3)
    x = numpy.arange(n) # Avoiding lag 0 calculation
    acf_coeffs = map(r, x)
    return acf_coeffs
    
#def ACcal(seq):
#    return [autocorr(seq['s1']), autocorr(seq['s2']),
#            autocorr(seq['s3']), autocorr(seq['s4']),
#            autocorr(seq['s5']), autocorr(seq['s6']),
#            autocorr(seq['s7']), autocorr(seq['s8'])]

def ACcal(seq):
    return [autoCorr(seq['s1']), autoCorr(seq['s2']),
            autoCorr(seq['s3']), autoCorr(seq['s4']),
            autoCorr(seq['s5']), autoCorr(seq['s6']),
            autoCorr(seq['s7']), autoCorr(seq['s8'])]


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
 
accc_cols = ['ACCC_s0', 'ACCC_s1', 'ACCC_s2', 'ACCC_s3', 'ACCC_s4', 'ACCC_s5', 'ACCC_s6', 'ACCC_s7']

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
NEW_DIR = "new"
HAI_DIR = "hai"
HOA_DIR = "hoa"
GROUP_NAMES = [NUMBER_GROUP, TAPPING_GROUP, WRIST]
def readFiles(mydir, ending):
    os.chdir(mydir)
    return glob.glob("*"+ending)
    
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
Xs_tmd = tmdv_new[0]
Ys_tmd = tmdv_new[1]

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