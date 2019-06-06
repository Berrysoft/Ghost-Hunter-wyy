import h5py
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

fipt = "answerfile.h5"
fopt = "submissionfile.h5"

opd = [('EventID', '<i8'), ('ChannelID', '<i2'), ('PETime', 'f4'), ('Weight', 'f4')]
tresh = 9
para = [4.70557863824900, 8.59153064595592, 39.5970278962013, 4.17111271556104]

def modelfunc(a1,a2,a3,a4,t):
    if (t <= a2 and t >= a1):
        return a3 * (1 - math.exp(-(t - a1) / a4))
    elif (t > a2):
        return a3 * (math.exp(-(a1 - a2) / a4) - 1) * math.exp(-(t - a1) / a4)
    else:
        return 0

vecmodelfunc = np.vectorize(modelfunc)

def mmp(wr):
    w = np.array(wr[2], dtype=np.int16)

    nothing = np.array(w[range(0,200)], dtype=np.int16)
    nothing[nothing < 970] = 0
    backg = np.average(nothing)
    wr2 = backg - w

    tot = 0
    finalpetime = []
    weigh = []

    if (wr2 < 9).all():
        tot = np.argmax(wr2)
        finalpetime.append(tot - 7)
        weigh.append(1)
    else:
        begining = wr2[range(0,10)] < tresh
        if not begining.all():
            tot = begining[begining == False]
            finalpetime = tot - 6
            weigh = 1
            wr2[range(1,10)] = 0

        while True:
            wr3 = wr2
            wr3[wr3 < tresh] = 0
            tot = np.argwhere(wr3 > 0)
            if len(tot) == 0:
                break
            petime = tot[0][0] - 6

            if len(finalpetime) == 0 or (petime - finalpetime[-1]) < 3 or (petime - finalpetime[-1]) > 5 or wr2[petime] > 12:
                weigh.append(1)
                finalpetime.append(petime)
            if len(finalpetime) > 500:
                raise Exception('too many PEs found, there must be a bug.')

            dis = np.arange(1029)
            wr2 -= vecmodelfunc(petime + para[0], petime + para[1], para[2], para[3], dis)
    
    rst = np.zeros(len(finalpetime), dtype=opd)
    rst['PETime'] = finalpetime
    rst['Weight'] = weigh
    rst['EventID'] = wr[0]
    rst['ChannelID'] = wr[1]

    return rst

#ipt = h5py.File(fipt)
#wr = ipt['Waveform'][100]
#d = mmp(wr)
#plt.plot(wr['Waveform'])
#plt.vlines(d['PETime'],900,1000)
#plt.show()

#
with h5py.File(fipt) as ipt,h5py.File(fopt, "w") as opt:
    dt = np.concatenate([mmp(wr) for wr in ipt['Waveform'][:]])
    opt.create_dataset('Answer', data=dt, compression='gzip')
