import numpy as np
import matplotlib.pyplot as plt

# 1D FFT
def fft1d(signal):
    N = len(signal)
    if N <= 1:
        return signal
    else:
        # 递归计算偶数和奇数部分
        even = fft1d(signal[0::2])
        odd = fft1d(signal[1::2])
        
        # 合并结果
        combined = [0] * N
        for k in range(N // 2):
            t = np.exp(-2j * np.pi * k / N) * odd[k]
            combined[k] = even[k] + t
            combined[k + N // 2] = even[k] - t
        return combined

# 2D FFT
def fft2d(image):
    M, N = image.shape
    
    # 对每一行进行 1D FFT
    row_fft = np.zeros((M, N), dtype=complex)
    for i in range(M):
        row_fft[i, :] = fft1d(image[i, :])
    
    # 对每一列进行 1D FFT
    col_fft = np.zeros((M, N), dtype=complex)
    for j in range(N):
        col_fft[:, j] = fft1d(row_fft[:, j])
    
    return col_fft


def AmplitudeSpecturm(image_ft):
    return np.abs(image_ft)

def PhaseSpecturm(image_ft):
    return np.angle(image_ft)

def DispAmplitude(image_ft):
    f_transform_shifted_fast = np.fft.fftshift(image_ft)
    log_amplitude = np.log1p(np.abs(f_transform_shifted_fast))
    plt.imshow(log_amplitude, cmap='gray')
    plt.title('Amplitude Spectrum')
    plt.show()

def DispPhase(image_ft):
    f_transform_shifted_fast = np.fft.fftshift(image_ft)
    plt.imshow(np.angle(f_transform_shifted_fast), cmap='gray')
    plt.title('Phase Spectrum')
    plt.show()
