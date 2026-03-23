import numpy as np
from scipy.special import jv


def bessel(v, X):
    return ((1j ** (-v)) * jv(v, 1j * X)).real


def stft(x, n_fft=512, win_length=400, hop_length=160, window="hamming"):
    if window == "hanning":
        window = np.hanning(win_length)
    elif window == "hamming":
        window = np.hamming(win_length)
    elif window == "rectangle":
        window = np.ones(win_length)
    return np.array(
        [
            np.fft.rfft(window * x[i : i + win_length], n_fft, axis=0)
            for i in range(0, len(x) - win_length, hop_length)
        ]
    )


def estnoisem(pSpectrum, hop_length):
    eps = 1e-12
    hop_time = float(hop_length)
    if hop_time > 1.0:
        hop_time = hop_time / 16000.0
    hop_time = max(hop_time, eps)

    (nFrames, nFFT2) = np.shape(pSpectrum)  # number of frames and freq bins
    x = np.array(np.zeros((nFrames, nFFT2)))  # initialize output arrays
    xs = np.array(np.zeros((nFrames, nFFT2)))  # will hold std error in the future

    # default algorithm constants
    taca = (
        0.0449  # smoothing time constant for alpha_c = -hop_length/log(0.7) in equ (11)
    )
    tamax = 0.392  # max smoothing time constant in (3) = -hop_length/log(0.96)
    taminh = 0.0133  # min smoothing time constant (upper limit) in (3) = -hop_length/log(0.3)
    tpfall = 0.064  # time constant for P to fall (12)
    tbmax = 0.0717  # max smoothing time constant in (20) = -hop_length/log(0.8)
    qeqmin = 2.0  # minimum value of Qeq (23)
    qeqmax = 14.0  # max value of Qeq per frame
    av = 2.12  # fudge factor for bc calculation (23 + 13 lines)
    td = 1.536  # time to take minimum over
    nu = 8  # number of subwindows
    qith = np.array(
        [0.03, 0.05, 0.06, np.inf], dtype=float
    )  # noise slope thresholds in dB/s
    nsmdb = np.array(
        [47, 31.4, 15.7, 4.1], dtype=float
    )  # maximum permitted +ve noise slope in dB/s

    # derived algorithm constants
    aca = np.exp(-hop_time / taca)  # smoothing constant for alpha_c in equ (11) = 0.7
    acmax = aca  # min value of alpha_c = 0.7 in equ (11) also = 0.7
    amax = np.exp(-hop_time / tamax)  # max smoothing constant in (3) = 0.96
    aminh = np.exp(
        -hop_time / taminh
    )  # min smoothing constant (upper limit) in (3) = 0.3
    bmax = np.exp(-hop_time / tbmax)  # max smoothing constant in (20) = 0.8
    SNRexp = -hop_time / tpfall
    nv = round(td / (hop_time * nu))  # length of each subwindow in frames

    if nv < 4:  # algorithm doesn't work for miniscule frames
        nv = 4
        nu = round(td / (hop_time * nv))
    nd = nu * nv  # length of total window in frames
    (md, hd, dd) = mhvals(nd)  # calculate the constants M(D) and H(D) from Table III
    (mv, hv, dv) = mhvals(nv)  # calculate the constants M(D) and H(D) from Table III
    nsms = np.array([10.0]) ** np.clip(
        nsmdb * nv * hop_time / 10, -50.0, 50.0
    )  # [8 4 2 1.2] in paper
    qeqimax = 1 / qeqmin  # maximum value of Qeq inverse (23)
    qeqimin = 1 / qeqmax  # minumum value of Qeq per frame inverse

    p = pSpectrum[0, :]  # smoothed power spectrum
    ac = 1  # correction factor (9)
    sn2 = p  # estimated noise power
    pb = p  # smoothed noisy speech power (20)
    pb2 = pb**2
    pminu = p
    actmin = np.array(np.ones(nFFT2) * np.inf)  # Running minimum estimate
    actminsub = np.array(np.ones(nFFT2) * np.inf)  # sub-window minimum estimate
    subwc = nv  # force a buffer switch on first loop
    actbuf = np.array(np.ones((nu, nFFT2)) * np.inf)  # buffer to store subwindow minima
    ibuf = 0
    lminflag = np.zeros(nFFT2)  # flag to remember local minimum

    # loop for each frame
    for t in range(0, nFrames):  # we use t instead of lambda in the paper
        pSpectrum_t = pSpectrum[t, :]  # noise speech power spectrum
        acb = (1 + (sum(p) / (sum(pSpectrum_t) + eps) - 1) ** 2) ** (
            -1
        )  # alpha_c-bar(t)  (9)

        tmp = np.array([acb])
        tmp[tmp < acmax] = acmax
        # max_complex(np.array([acb] ),np.array([acmax] ))

        ac = aca * ac + (1 - aca) * tmp  # alpha_c(t)  (10)

        sn2_safe = np.maximum(sn2, eps)
        ah = (
            amax * ac * (1 + (p / sn2_safe - 1) ** 2) ** (-1)
        )  # alpha_hat: smoothing factor per frequency (11)
        SNR = sum(p) / (sum(sn2_safe) + eps)
        SNR = max(float(SNR), eps)

        ah = max_complex(
            ah, min_complex(np.array([aminh]), np.array([SNR**SNRexp]))
        )  # lower limit for alpha_hat (12)

        p = ah * p + (1 - ah) * pSpectrum_t  # smoothed noisy speech power (3)

        b = min_complex(
            ah**2, np.array([bmax])
        )  # smoothing constant for estimating periodogram variance (22 + 2 lines)
        pb = b * pb + (1 - b) * p  # smoothed periodogram (20)
        pb2 = b * pb2 + (1 - b) * p**2  # smoothed periodogram squared (21)

        qeqi_raw = (pb2 - pb**2) / (2 * sn2_safe**2)
        qeqi = max_complex(
            min_complex(qeqi_raw, np.array([qeqimax])), np.array([qeqimin / (t + 1)])
        )  # Qeq inverse (23)
        qeqi_safe = np.maximum(qeqi, eps)
        qeqi_inv = 1.0 / qeqi_safe
        qiav = (
            sum(qeqi) / nFFT2
        )  # Average over all frequencies (23+12 lines) (ignore non-duplication of DC and nyquist terms)
        bc = 1 + av * np.sqrt(qiav)  # bias correction factor (23+11 lines)
        bmind_denom = qeqi_inv - 2 * md
        bmind_denom = np.where(np.abs(bmind_denom) < eps, eps, bmind_denom)
        bmind = (
            1 + 2 * (nd - 1) * (1 - md) / bmind_denom
        )  # we use the signalmplified form (17) instead of (15)
        bminv_denom = qeqi_inv - 2 * mv
        bminv_denom = np.where(np.abs(bminv_denom) < eps, eps, bminv_denom)
        bminv = (
            1 + 2 * (nv - 1) * (1 - mv) / bminv_denom
        )  # same expressignalon but for sub windows
        kmod = (bc * p * bmind) < actmin  # Frequency mask for new minimum

        if any(kmod):
            actmin[kmod] = bc * p[kmod] * bmind[kmod]
            actminsub[kmod] = bc * p[kmod] * bminv[kmod]

        if subwc > 1 and subwc < nv:  # middle of buffer - allow a local minimum
            lminflag = np.logical_or(
                lminflag, kmod
            )  # potential local minimum frequency bins
            pminu = min_complex(actminsub, pminu)
            sn2 = pminu.copy()
        else:
            if subwc >= nv:  # end of buffer - do a buffer switch
                ibuf = 1 + (ibuf % nu)  # increment actbuf storage pointer
                actbuf[ibuf - 1, :] = actmin.copy()  # save sub-window minimum
                pminu = min_complex_mat(actbuf)
                i = np.nonzero(np.array(qiav) < qith)
                nsm = nsms[i[0][0]]  # noise slope max
                lmin = np.logical_and(
                    np.logical_and(
                        np.logical_and(lminflag, np.logical_not(kmod)),
                        actminsub < (nsm * pminu),
                    ),
                    actminsub > pminu,
                )
                if any(lmin):
                    pminu[lmin] = actminsub[lmin]
                    actbuf[:, lmin] = np.ones((nu, 1)) * pminu[lmin]
                lminflag[:] = 0
                actmin[:] = np.inf
                subwc = 0

        subwc = subwc + 1
        x[t, :] = sn2.copy()
        qisq = np.sqrt(qeqi)
        # empirical formula for standard error based on Fig 15 of [2]
        xs_denom = 0.5 * qeqi_inv + nd - 1
        xs_denom = np.where(np.abs(xs_denom) < eps, eps, xs_denom)
        xs[t, :] = sn2 * np.sqrt(
            0.266 * (nd + 100 * qisq) * qisq / (1 + 0.005 * nd + 6 / nd) / xs_denom
        )

    return x


def mhvals(*args):
    eps = 1e-12
    nargin = len(args)

    dmh = np.array(
        [
            [1, 0, 0],
            [2, 0.26, 0.15],
            [5, 0.48, 0.48],
            [8, 0.58, 0.78],
            [10, 0.61, 0.98],
            [15, 0.668, 1.55],
            [20, 0.705, 2],
            [30, 0.762, 2.3],
            [40, 0.8, 2.52],
            [60, 0.841, 3.1],
            [80, 0.865, 3.38],
            [120, 0.89, 4.15],
            [140, 0.9, 4.35],
            [160, 0.91, 4.25],
            [180, 0.92, 3.9],
            [220, 0.93, 4.1],
            [260, 0.935, 4.7],
            [300, 0.94, 5],
        ],
        dtype=float,
    )

    if nargin >= 1:
        d = args[0]
        i = np.nonzero(d <= dmh[:, 0])
        if len(i) == 0:
            i = np.shape(dmh)[0] - 1
            j = max(i - 1, 0)
        else:
            i = i[0][0]
            j = max(i - 1, 0)
        if d == dmh[i, 0]:
            m = dmh[i, 1]
            h = dmh[i, 2]
        else:
            qj = np.sqrt(dmh[i - 1, 0])  # interpolate usignalng sqrt(d)
            qi = np.sqrt(dmh[i, 0])
            q = np.sqrt(d)
            if abs(qj - qi) < eps or abs(qi - qj) < eps or q < eps:
                m = dmh[i, 1]
                h = dmh[i, 2]
            else:
                h = dmh[i, 2] + (q - qi) * (dmh[j, 2] - dmh[i, 2]) / (qj - qi)
                m = dmh[i, 1] + (qi * qj / q - qj) * (dmh[j, 1] - dmh[i, 1]) / (qi - qj)
    else:
        d = dmh[:, 0].copy()
        m = dmh[:, 1].copy()
        h = dmh[:, 2].copy()

    return m, h, d


def max_complex(a, b):
    if len(a) == 1 and len(b) > 1:
        a = np.tile(a, np.shape(b))
    if len(b) == 1 and len(a) > 1:
        b = np.tile(b, np.shape(a))

    i = np.logical_or(np.iscomplex(a), np.iscomplex(b))

    aa = a.copy()
    bb = b.copy()

    if any(i):
        aa[i] = np.absolute(aa[i])
        bb[i] = np.absolute(bb[i])
    if a.dtype == "complex" or b.dtype == "complex":
        cc = np.array(np.zeros(np.shape(a)))
    else:
        cc = np.array(np.zeros(np.shape(a)), dtype=float)

    i = aa > bb
    cc[i] = a[i]
    cc[np.logical_not(i)] = b[np.logical_not(i)]

    return cc


def min_complex(a, b):
    if len(a) == 1 and len(b) > 1:
        a = np.tile(a, np.shape(b))
    if len(b) == 1 and len(a) > 1:
        b = np.tile(b, np.shape(a))

    i = np.logical_or(np.iscomplex(a), np.iscomplex(b))

    aa = a.copy()
    bb = b.copy()

    if any(i):
        aa[i] = np.absolute(aa[i])
        bb[i] = np.absolute(bb[i])

    if a.dtype == "complex" or b.dtype == "complex":
        cc = np.array(np.zeros(np.shape(a)))
    else:
        cc = np.array(np.zeros(np.shape(a)), dtype=float)

    i = aa < bb
    cc[i] = a[i]
    cc[np.logical_not(i)] = b[np.logical_not(i)]

    return cc


def min_complex_mat(a):
    s = np.shape(a)
    m = np.array(np.zeros(s[1]))
    for i in range(0, s[1]):
        j = np.argmin(np.absolute(a[:, i]))
        m[i] = a[j, i]
    return m


class SpectralNoiseReducer:
    """Reusable spectral noise reduction service for STT pipelines."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        window: str = "hamming",
        prop_decrease: float = 1.0,
        gain_floor: float = 0.05,
        eps: float = 1e-10,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = window
        self.prop_decrease = prop_decrease
        self.gain_floor = gain_floor
        self.eps = eps
        self._window = self._build_window(window, win_length)

    @staticmethod
    def _build_window(window: str, win_length: int) -> np.ndarray:
        if window == "hanning":
            return np.hanning(win_length)
        if window == "hamming":
            return np.hamming(win_length)
        if window == "rectangle":
            return np.ones(win_length)
        raise ValueError(f"Unsupported window type: {window}")

    def _istft(self, stft_matrix: np.ndarray, target_length: int) -> np.ndarray:
        n_frames = stft_matrix.shape[0]
        if n_frames == 0:
            return np.zeros(target_length, dtype=np.float32)

        frames = np.fft.irfft(stft_matrix, n=self.n_fft, axis=1)[:, : self.win_length]
        frames = frames * self._window[None, :]

        out_len = self.hop_length * (n_frames - 1) + self.win_length
        signal = np.zeros(out_len, dtype=np.float64)
        window_sum = np.zeros(out_len, dtype=np.float64)

        for index in range(n_frames):
            start = index * self.hop_length
            end = start + self.win_length
            signal[start:end] += frames[index]
            window_sum[start:end] += self._window**2

        valid = window_sum > self.eps
        signal[valid] /= window_sum[valid]

        if out_len < target_length:
            signal = np.pad(signal, (0, target_length - out_len), mode="constant")
        else:
            signal = signal[:target_length]

        return signal.astype(np.float32)

    def denoise_array(self, audio: np.ndarray) -> np.ndarray:
        if audio is None:
            return np.array([], dtype=np.float32)

        audio = np.asarray(audio, dtype=np.float32).flatten()
        if audio.size < self.win_length:
            return audio

        stft_matrix = stft(
            audio,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=self.window,
        )

        if stft_matrix.size == 0:
            return audio

        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)
        power_spectrum = magnitude**2

        noise_power = estnoisem(power_spectrum, self.hop_length)
        gain = 1.0 - self.prop_decrease * (noise_power / (power_spectrum + self.eps))
        gain = np.clip(gain, self.gain_floor, 1.0)

        enhanced_mag = magnitude * gain
        enhanced_stft = enhanced_mag * np.exp(1j * phase)

        return self._istft(enhanced_stft, audio.shape[0])

    def denoise_pcm16(self, pcm_bytes: bytes) -> bytes:
        if not pcm_bytes:
            return b""

        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        enhanced = self.denoise_array(audio_float)
        enhanced_int16 = np.clip(enhanced * 32768.0, -32768, 32767).astype(np.int16)
        return enhanced_int16.tobytes()
