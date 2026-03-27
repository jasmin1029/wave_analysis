"""
岩石超声波形特征分析模块

在 AIC 法拾取初至后，对有效信号段进行定量评估：
1. 波幅参数：最大振幅、峰峰值、RMS 振幅
2. 频率参数：主频、平均频率、频谱重心、带宽
3. 能量参数：总能量、信号能量占比、累积能量曲线
4. 波形复杂度：过零率、峰值因子、波形因子、峭度、偏度、信息熵
5. 衰减参数：包络衰减拟合、品质因子 Q 估算
"""

import numpy as np
from scipy.signal import hilbert
from scipy.optimize import curve_fit


# ====================== 波幅参数 ======================

def amplitude_params(signal):
    """
    计算波幅相关参数

    Returns
    -------
    dict: peak_amplitude, peak_to_peak, rms_amplitude, mean_abs_amplitude
    """
    abs_sig = np.abs(signal)
    return {
        "peak_amplitude": float(np.max(abs_sig)),
        "peak_to_peak": float(np.max(signal) - np.min(signal)),
        "rms_amplitude": float(np.sqrt(np.mean(signal ** 2))),
        "mean_abs_amplitude": float(np.mean(abs_sig)),
    }


# ====================== 频率参数 ======================

def frequency_params(signal, fs):
    """
    通过 FFT 计算频率域参数

    Returns
    -------
    dict: dominant_freq, mean_freq, centroid_freq, bandwidth, freq_axis, spectrum
    """
    n = len(signal)
    windowed = signal * np.hanning(n)

    fft_vals = np.fft.rfft(windowed)
    magnitude = np.abs(fft_vals)
    freq_axis = np.fft.rfftfreq(n, d=1.0 / fs)

    magnitude[0] = 0

    total_power = np.sum(magnitude ** 2)
    if total_power < 1e-30:
        return {
            "dominant_freq": 0.0,
            "mean_freq": 0.0,
            "centroid_freq": 0.0,
            "bandwidth": 0.0,
            "freq_axis": freq_axis,
            "spectrum": magnitude,
        }

    dominant_freq = float(freq_axis[np.argmax(magnitude)])

    power = magnitude ** 2
    centroid_freq = float(np.sum(freq_axis * power) / np.sum(power))

    mean_freq = centroid_freq

    variance = np.sum(((freq_axis - centroid_freq) ** 2) * power) / np.sum(power)
    bandwidth = float(np.sqrt(variance))

    return {
        "dominant_freq": dominant_freq,
        "mean_freq": mean_freq,
        "centroid_freq": centroid_freq,
        "bandwidth": bandwidth,
        "freq_axis": freq_axis,
        "spectrum": magnitude,
    }


# ====================== 能量参数 ======================

def energy_params(signal, fs):
    """
    计算能量相关参数

    Returns
    -------
    dict: total_energy, signal_duration, energy_density, cumulative_energy
    """
    energy = signal ** 2
    total_energy = float(np.sum(energy))
    dt = 1.0 / fs
    duration = len(signal) * dt
    cumulative_energy = np.cumsum(energy)

    t90 = np.searchsorted(cumulative_energy, 0.90 * total_energy)
    t10 = np.searchsorted(cumulative_energy, 0.10 * total_energy)
    rise_time = (t90 - t10) * dt

    return {
        "total_energy": total_energy,
        "signal_duration_s": float(duration),
        "energy_density": float(total_energy / duration) if duration > 0 else 0.0,
        "rise_time_s": float(rise_time),
        "cumulative_energy": cumulative_energy / total_energy,
    }


# ====================== 波形复杂度 ======================

def complexity_params(signal):
    """
    计算波形复杂度相关参数

    Returns
    -------
    dict: zero_crossing_rate, crest_factor, form_factor, kurtosis, skewness, spectral_entropy
    """
    n = len(signal)

    zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
    zero_crossing_rate = float(zero_crossings / n)

    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    mean_abs = np.mean(np.abs(signal))

    crest_factor = float(peak / rms) if rms > 1e-30 else 0.0
    form_factor = float(rms / mean_abs) if mean_abs > 1e-30 else 0.0
    impulse_factor = float(peak / mean_abs) if mean_abs > 1e-30 else 0.0

    mu = np.mean(signal)
    sigma = np.std(signal)
    if sigma > 1e-30:
        centered = (signal - mu) / sigma
        kurtosis = float(np.mean(centered ** 4) - 3.0)
        skewness = float(np.mean(centered ** 3))
    else:
        kurtosis = 0.0
        skewness = 0.0

    fft_vals = np.fft.rfft(signal)
    power = np.abs(fft_vals) ** 2
    power_sum = np.sum(power)
    if power_sum > 1e-30:
        p_norm = power / power_sum
        p_norm = p_norm[p_norm > 0]
        spectral_entropy = float(-np.sum(p_norm * np.log2(p_norm)))
    else:
        spectral_entropy = 0.0

    return {
        "zero_crossing_rate": zero_crossing_rate,
        "crest_factor": crest_factor,
        "form_factor": form_factor,
        "impulse_factor": impulse_factor,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "spectral_entropy": spectral_entropy,
    }


# ====================== 衰减参数 ======================

def _exp_decay(t, a, b):
    return a * np.exp(-b * t)


def attenuation_params(signal, fs):
    """
    通过包络拟合估算衰减参数

    Returns
    -------
    dict: decay_coefficient, quality_factor_Q, envelope
    """
    analytic = hilbert(signal)
    envelope = np.abs(analytic)

    peak_idx = np.argmax(envelope)
    env_after_peak = envelope[peak_idx:]
    n_after = len(env_after_peak)

    if n_after < 10:
        return {
            "decay_coefficient": 0.0,
            "quality_factor_Q": 0.0,
            "envelope": envelope,
        }

    t_after = np.arange(n_after) / fs
    decay_coeff = 0.0
    quality_factor = 0.0

    try:
        p0 = [envelope[peak_idx], 1000.0]
        bounds = ([0, 0], [np.inf, 1e8])
        popt, _ = curve_fit(
            _exp_decay, t_after, env_after_peak,
            p0=p0, bounds=bounds, maxfev=5000,
        )
        decay_coeff = float(popt[1])

        fft_vals = np.fft.rfft(signal)
        magnitude = np.abs(fft_vals)
        freq_axis = np.fft.rfftfreq(len(signal), d=1.0 / fs)
        dominant_freq = freq_axis[np.argmax(magnitude[1:]) + 1]

        if decay_coeff > 1.0:
            quality_factor = float(np.pi * dominant_freq / decay_coeff)
    except (RuntimeError, ValueError):
        pass

    return {
        "decay_coefficient": decay_coeff,
        "quality_factor_Q": quality_factor,
        "envelope": envelope,
    }


# ====================== 综合分析 ======================

def analyze_waveform(full_signal, onset_index, fs):
    """
    综合分析入口：基于 AIC 拾取的初至索引，对有效信号段做全面特征提取

    Parameters
    ----------
    full_signal : 完整波形数组
    onset_index : AIC 法拾取的初至采样点索引
    fs : 采样率 (Hz)

    Returns
    -------
    dict: 包含所有分析结果的字典
    """
    effective_signal = full_signal[onset_index:]
    noise_signal = full_signal[:onset_index] if onset_index > 10 else full_signal[:10]

    amp = amplitude_params(effective_signal)
    freq = frequency_params(effective_signal, fs)
    eng = energy_params(effective_signal, fs)
    comp = complexity_params(effective_signal)
    atten = attenuation_params(effective_signal, fs)

    noise_rms = np.sqrt(np.mean(noise_signal ** 2))
    signal_rms = amp["rms_amplitude"]
    if noise_rms > 1e-30:
        snr_estimated = float(20 * np.log10(signal_rms / noise_rms))
    else:
        snr_estimated = float("inf")

    return {
        "onset_index": onset_index,
        "onset_time_us": onset_index / fs * 1e6,
        "effective_length": len(effective_signal),
        "amplitude": amp,
        "frequency": freq,
        "energy": eng,
        "complexity": comp,
        "attenuation": atten,
        "estimated_snr_db": snr_estimated,
    }


def print_analysis_report(result):
    """将分析结果格式化输出到控制台"""
    print("\n" + "=" * 64)
    print("       岩石超声波形特征分析报告")
    print("=" * 64)

    print(f"\n  AIC 初至位置: 第 {result['onset_index']} 个采样点"
          f" ({result['onset_time_us']:.2f} us)")
    print(f"  有效信号长度: {result['effective_length']} 个采样点")
    print(f"  估算信噪比:   {result['estimated_snr_db']:.1f} dB")

    print("\n" + "-" * 64)
    print("  [波幅参数]")
    amp = result["amplitude"]
    print(f"    最大振幅:       {amp['peak_amplitude']:.6f}")
    print(f"    峰峰值:         {amp['peak_to_peak']:.6f}")
    print(f"    RMS 振幅:       {amp['rms_amplitude']:.6f}")
    print(f"    平均绝对振幅:   {amp['mean_abs_amplitude']:.6f}")

    print("\n" + "-" * 64)
    print("  [频率参数]")
    freq = result["frequency"]
    print(f"    主频:           {freq['dominant_freq'] / 1e3:.2f} kHz")
    print(f"    频谱重心:       {freq['centroid_freq'] / 1e3:.2f} kHz")
    print(f"    频带宽度:       {freq['bandwidth'] / 1e3:.2f} kHz")

    print("\n" + "-" * 64)
    print("  [能量参数]")
    eng = result["energy"]
    print(f"    总能量:         {eng['total_energy']:.6e}")
    print(f"    能量密度:       {eng['energy_density']:.6e} /s")
    print(f"    上升时间:       {eng['rise_time_s'] * 1e6:.2f} us")

    print("\n" + "-" * 64)
    print("  [波形复杂度]")
    comp = result["complexity"]
    print(f"    过零率:         {comp['zero_crossing_rate']:.4f}")
    print(f"    峰值因子:       {comp['crest_factor']:.4f}")
    print(f"    波形因子:       {comp['form_factor']:.4f}")
    print(f"    脉冲因子:       {comp['impulse_factor']:.4f}")
    print(f"    峭度 (超额):    {comp['kurtosis']:.4f}")
    print(f"    偏度:           {comp['skewness']:.4f}")
    print(f"    频谱熵:         {comp['spectral_entropy']:.4f} bits")

    print("\n" + "-" * 64)
    print("  [衰减参数]")
    atten = result["attenuation"]
    print(f"    衰减系数:       {atten['decay_coefficient']:.2f} /s")
    print(f"    品质因子 Q:     {atten['quality_factor_Q']:.2f}")

    print("\n" + "=" * 64)
