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


