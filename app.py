# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:26:34 2016

@author: Hoa
"""

#!flask/bin/python
#!flask/bin/python
from flask import Flask, jsonify, abort
from flask import request
import socket

#features
import numpy
import pandas
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

#machine learning
from sklearn.externals import joblib

app = Flask(__name__)

#=======================================================
GESTURE_LENGTH = 240 #240 points, 5ms/point => 1200 ms
WINDOW_SIZE = 40 #200ms
INCREMENT = 20 #100ms

def overlappedWindow(basis, window, increment):
    #adjacent disjoint windows
    return (basis[pos:pos + window] for pos in range(0, len(basis), window-increment))

def mav(x):
    return numpy.mean(numpy.abs(x)) if not(x.empty) else 0

def MAVcal(x):
    return [mav(x['s1']), mav(x['s2']), 
            mav(x['s3']), mav(x['s4']),
            mav(x['s5']), mav(x['s6']), 
            mav(x['s7']), mav(x['s8'])]  

#mean absolute value ratio: the ratio of MAV between channel
def MAVRcal(x):
    ret = []
    for i in range(8):
        for j in range(i, 8):
            if i != j:
                v = (mav(x['s'+str(i+1)])/mav(x['s'+str(j+1)])) if mav(x['s'+str(j+1)]) != 0 else 0
                ret += [v]
    return ret

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
seq_id = 0
tmd_funcs = [MAVcal, MAVRcal, zeroCrossingRate, WLcal]
def extractSample(sequence):
    seq = pandas.DataFrame.from_csv(StringIO(sequence), sep=',', index_col=False)
        
    seq.to_csv(GIT_DIR+'/seq'+str(seq_id+1), sep=',', index=False)    
    def applyFunctions(val):
        ret_dict = []
        for f in tmd_funcs:
            ret_dict += f(val)
        return ret_dict
    return numpy.asarray([item for sublist in [applyFunctions(i) for i in overlappedWindow(seq[:GESTURE_LENGTH], WINDOW_SIZE, INCREMENT)] for item in sublist]).reshape(1, -1)
GIT_DIR = "C:/Users/Hoa/thesis/" 
model = "models/OVR_RFC.pkl"
#=======================================================
HTTP = '192.168.143.1'

@app.route('/todo/api/v1.0/tasks', methods=['GET'])
def test():
    return jsonify({'connection': 'established'})
@app.route('/todo/api/v1.0/tasks', methods=['POST'])
def predict():
    seq = request.get_json(force=True)['seq']
    gesture = request.get_json(force=True)['gesture']
    features = extractSample(seq)
    clf = joblib.load(GIT_DIR+"/"+model)
    result = str(clf.predict(features)[0])
    return jsonify({'gesture_name': result})

if __name__ == '__main__':
    app.run(host=HTTP,port=5000,debug=True)
#    app.run(debug=True)