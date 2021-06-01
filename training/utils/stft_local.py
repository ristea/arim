import numpy as np


def stft(x, Nfft, w, Nstep):
    N = len(x)
    M = len(w)
    Nstep = Nstep

    Nmax = int(N / Nstep)

    S = np.zeros((Nfft, Nmax)).astype(np.complex)
    w = w / sum(w)
    x1 = np.concatenate((np.zeros(M), x, np.zeros(M)), 0)

    for ii in range(1, Nmax):
        w_pad = np.zeros(N + 2 * M)
        w_pad[int((ii - 1) * Nstep + 1 + M / 2): int((ii - 1) * Nstep + M + M / 2) + 1] = w
        s = x1 * w_pad
        s = s[M + 1:]
        S[:, ii] = np.fft.fft(s, Nfft) / 1024

    return S
