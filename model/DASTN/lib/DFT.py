import heapq
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def get_domaintF(data,p):
    y = np.mean(data[:, :, 0].T, axis=0)
    y = y - np.mean(y)
    x=np.arange(len(y))
    n = len(x)
    sample_freq = (n - 1) / (x[-1] - x[0])  # 信号的采样频率
    freqs = fftfreq(n, 1. / sample_freq)[:n // 2]
    amplitudes = 2. / n * np.abs(fft(y)[:n // 2])

    topkA_index=heapq.nlargest(p, range(len(amplitudes)), amplitudes.__getitem__)
    topkA=heapq.nlargest(p,amplitudes)
    topkF=freqs[topkA_index]
    return topkF, topkA

y=np.load('pems04.npz')['data'].transpose(0,2,1)
y2=np.expand_dims(np.load('CD_S.npz')['arr_0'],axis=1)
y3=np.load('pems08.npz')['data'].transpose(0,2,1)
y4=np.expand_dims(np.load('SZ_S.npz')['data'],axis=1)



a,b=get_domaintF(y,500)
a2,b2=get_domaintF(y2,500)
a3,b3=get_domaintF(y3,500)
a4,b4=get_domaintF(y4,500)



fig1 = plt.figure()
ax1 = fig1.add_subplot(411)
rect=plt.Rectangle(
            (91, 0),  
            10,  # width
            60,  # height
            color='yellow',
            alpha=0.5      )
rect2=plt.Rectangle(
            (139, 0),  # (x,y)
            10,  # width
            60,  # height
            color='yellow',
            alpha=0.5      )
rect3=plt.Rectangle(
            (283, 0),  # (x,y)
            10,  # width
            60,  # height
            color='yellow',
            alpha=0.5      )
ax1.add_patch(rect)
ax1.add_patch(rect2)
ax1.add_patch(rect3)


plt.text(x=7,  
         y=40, 
         s='PeMS04',  
         ha='left',  
         va='baseline',  
         )
plt.xlim((0, 400))
plt.stem(1/(a), b,markerfmt='ro') 
plt.ylabel('Amplitude')

ax1 = fig1.add_subplot(412)
rect=plt.Rectangle(
            (91, 0), 
            10,  # width
            60,  # height
            color='yellow',
            alpha=0.5      )
rect2=plt.Rectangle(
            (139, 0),  # (x,y)
            10,  
            60, 
            color='yellow',
            alpha=0.5      )
rect3=plt.Rectangle(
            (283, 0),  
            10,  
            60, 
            color='yellow',
            alpha=0.5      )
ax1.add_patch(rect)
ax1.add_patch(rect2)
ax1.add_patch(rect3)
plt.text(x=7,  
         y=40,  
         s='PeMS08', 
         ha='left', 
         va='baseline',  
         )
plt.xlim((0, 400))
plt.stem(1/(a3), b3,markerfmt='ro') 
plt.ylabel('Amplitude')

ax1 = fig1.add_subplot(413)
rect=plt.Rectangle(
            (45.5, 0),  
            5,  # width
            4,  # height
            color='yellow',
            alpha=0.5      )
rect2=plt.Rectangle(
            (69.5, 0),  # (x,y)
            5,  # width
            4,  # height
            color='yellow',
            alpha=0.5      )
rect3=plt.Rectangle(
            (141.5, 0),  # (x,y)
            5,  # width
            4,  # height
            color='yellow',
            alpha=0.5      )
ax1.add_patch(rect)
ax1.add_patch(rect2)
ax1.add_patch(rect3)
plt.text(x=5,  
         y=2.8,  
         s='CD_S',  
         ha='left', 
         va='baseline', 
         )
plt.xlim((0, 200))
plt.stem(1/(a2), b2,markerfmt='ro') 
plt.ylabel('Amplitude')

ax1 = fig1.add_subplot(414)
rect=plt.Rectangle(
            (45.5, 0),  # (x,y)
            5,  # width
            3.5,  # height
            color='yellow',
            alpha=0.5      )
rect2=plt.Rectangle(
            (69.5, 0),  # (x,y)
            5,  # width
            3.5,  # height
            color='yellow',
            alpha=0.5      )
rect3=plt.Rectangle(
            (141.5, 0),  # (x,y)
            5,  # width
            3.5,  # height
            color='yellow',
            alpha=0.5      )
ax1.add_patch(rect)
ax1.add_patch(rect2)
ax1.add_patch(rect3)
plt.text(x=5,  
         y=2.5, 
         s='SZ_S',  
         ha='left', 
         va='baseline', 
         )
plt.xlim((0, 200))
plt.stem(1/(a4), b4,markerfmt='ro')
plt.ylabel('Amplitude')

plt.xlabel('Period')
plt.savefig('a.png')
plt.show()
'''
y=y-np.mean(y)
n = np.arange(len(y)//2)
#print(data.shape)
#y = np.cos(2*2*np.pi*(n/N)-np.pi/2)#+2*np.cos(2*2*2*np.pi*(n/N)+np.pi/2)
plt.stem(n, y[0:288*10],markerfmt='ro') 
plt.show()


re1 = np.fft.fft(y) 
plt.stem(n,re1[0:288*10]) 
plt.show()
'''

