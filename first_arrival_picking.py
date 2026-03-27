"""
岩石超声波形初至提取模块

实现了多种常用的初至拾取算法：
1. 阈值法 (Threshold Method)
2. STA/LTA 法 (Short-Term Average / Long-Term Average)
3. AIC 法 (Akaike Information Criterion)
4. 能量比法 (Energy Ratio Method)
5. 改进能量比法 (Modified Energy Ratio Method)
"""

import numpy as np
from scipy.signal import hilbert


def threshold_method(waveform: np.ndarray, threshold_ratio: float = 0.05) -> int:
    """
    阈值法初至拾取

    当信号幅值首次超过「最大幅值 × threshold_ratio」时，认为是初至点。
    简单快速，但对噪声敏感。

    Parameters
    ----------
    waveform : 一维波形数组
    threshold_ratio : 阈值比例 (0~1)，相对于最大绝对幅值

    Returns
    -------
    初至点索引
    """
    abs_wave = np.abs(waveform)
    threshold = np.max(abs_wave) * threshold_ratio
    indices = np.where(abs_wave >= threshold)[0]
    if len(indices) == 0:
        return 0
    return int(indices[0])


def sta_lta_method(
    waveform: np.ndarray,
    sta_len: int = 5,
    lta_len: int = 50,
    trigger_ratio: float = 3.0,
) -> int:
    """
    STA/LTA 法初至拾取

    计算短时窗均值与长时窗均值的比值，比值首次超过触发阈值时为初至。
    是地震学和超声检测中最常用的方法之一。

    Parameters
    ----------
    waveform : 一维波形数组
    sta_len : 短时窗长度（采样点数）
    lta_len : 长时窗长度（采样点数）
    trigger_ratio : 触发比值阈值

    Returns
    -------
    初至点索引
    """
    n = len(waveform)
    if n < lta_len + sta_len:
        return 0

    energy = waveform.astype(np.float64) ** 2
    cum = np.cumsum(energy)
    cum = np.insert(cum, 0, 0.0)

    start = lta_len
    end = n - sta_len
    idx = np.arange(start, end)
    lta_sum = (cum[idx] - cum[idx - lta_len]) / lta_len
    sta_sum = (cum[idx + sta_len] - cum[idx]) / sta_len

    ratio = np.zeros(n)
    safe = lta_sum > 1e-30
    ratio_slice = np.zeros(len(idx))
    ratio_slice[safe] = sta_sum[safe] / lta_sum[safe]
    ratio[start:end] = ratio_slice

    indices = np.where(ratio >= trigger_ratio)[0]
    if len(indices) == 0:
        return 0
    return int(indices[0])


def aic_method(waveform: np.ndarray) -> int:
    """
    AIC 法初至拾取（Akaike Information Criterion）

    对整段波形计算 AIC 函数，其全局最小值对应初至位置。
    适用性好、精度高，是学术研究中广泛使用的方法。

    AIC(k) = k * log(var(x[0:k])) + (N-k-1) * log(var(x[k:N]))

    Parameters
    ----------
    waveform : 一维波形数组

    Returns
    -------
    初至点索引
    """
    aic = compute_aic_curve(waveform)
    valid = np.isfinite(aic)
    if not np.any(valid):
        return 0
    aic_valid = aic.copy()
    aic_valid[~valid] = np.inf
    return int(np.argmin(aic_valid))


def aic_method_two_stage(
    waveform: np.ndarray,
    sta_len: int = 5,
    lta_len: int = 50,
    trigger_ratio: float = 2.0,
    window_expand: int = 50,
) -> int:
    """
    两步法初至拾取：STA/LTA 粗定位 + AIC 精细拾取

    先用 STA/LTA 确定粗略初至区间，再在该区间内用 AIC 法精确定位。
    兼顾效率和精度，推荐用于实际工程。

    Parameters
    ----------
    waveform : 一维波形数组
    sta_len : STA 窗口长度
    lta_len : LTA 窗口长度
    trigger_ratio : STA/LTA 触发比
    window_expand : AIC 搜索窗口向前扩展的点数

    Returns
    -------
    初至点索引
    """
    coarse = sta_lta_method(waveform, sta_len, lta_len, trigger_ratio)
    if coarse == 0:
        return aic_method(waveform)

    start = max(0, coarse - window_expand)
    end = min(len(waveform), coarse + window_expand)
    segment = waveform[start:end]
    local_pick = aic_method(segment)

    return start + local_pick


def energy_ratio_method(waveform: np.ndarray, window_len: int = 10) -> int:
    """
    能量比法初至拾取

    计算前后等长窗口的能量比，最大值位置即初至。

    ER(i) = sum(x[i:i+w]^2) / sum(x[i-w:i]^2)

    Parameters
    ----------
    waveform : 一维波形数组
    window_len : 能量计算窗口长度

    Returns
    -------
    初至点索引
    """
    er = compute_energy_ratio_curve(waveform, window_len)
    return int(np.argmax(er))


def modified_energy_ratio_method(waveform: np.ndarray, window_len: int = 10) -> int:
    """
    改进能量比法初至拾取

    在能量比基础上乘以绝对振幅，增强对初至点的分辨能力。

    MER(i) = |x[i]|^3 * ER(i)

    Parameters
    ----------
    waveform : 一维波形数组
    window_len : 能量计算窗口长度

    Returns
    -------
    初至点索引
    """
    er = compute_energy_ratio_curve(waveform, window_len)
    mer = (np.abs(waveform) ** 3) * er
    return int(np.argmax(mer))


def envelope_threshold_method(
    waveform: np.ndarray, threshold_ratio: float = 0.05
) -> int:
    """
    包络阈值法初至拾取

    通过 Hilbert 变换求包络，然后对包络施加阈值法。
    比直接阈值法更平滑，对高频噪声更鲁棒。

    Parameters
    ----------
    waveform : 一维波形数组
    threshold_ratio : 阈值比例 (0~1)

    Returns
    -------
    初至点索引
    """
    analytic_signal = hilbert(waveform)
    envelope = np.abs(analytic_signal)
    return threshold_method(envelope, threshold_ratio)


# ==================== 辅助函数 ====================


def compute_aic_curve(waveform: np.ndarray) -> np.ndarray:
    """计算完整 AIC 曲线（向量化，O(n) 复杂度）"""
    n = len(waveform)
    aic = np.full(n, np.nan)
    if n < 4:
        return aic

    x = waveform.astype(np.float64)
    cum_sum = np.cumsum(x)
    cum_sq = np.cumsum(x ** 2)

    k = np.arange(1, n - 1, dtype=np.float64)

    mean_before = cum_sum[:-2] / k  # cum_sum[k-1] / k  for k=1..n-2
    var_before = cum_sq[:-2] / k - mean_before ** 2
    var_before = np.maximum(var_before, 1e-20)

    total_sum = cum_sum[-1]
    total_sq = cum_sq[-1]
    n_after = (n - k).astype(np.float64)
    sum_after = total_sum - cum_sum[:-2]
    sq_after = total_sq - cum_sq[:-2]
    mean_after = sum_after / n_after
    var_after = sq_after / n_after - mean_after ** 2
    var_after = np.maximum(var_after, 1e-20)

    aic[1:n - 1] = k * np.log(var_before) + (n - k - 1) * np.log(var_after)
    return aic


def compute_sta_lta_curve(
    waveform: np.ndarray, sta_len: int = 5, lta_len: int = 50
) -> np.ndarray:
    """计算完整 STA/LTA 比值曲线（向量化）"""
    n = len(waveform)
    energy = waveform.astype(np.float64) ** 2
    cum = np.insert(np.cumsum(energy), 0, 0.0)

    ratio = np.zeros(n)
    start, end = lta_len, n - sta_len
    if start >= end:
        return ratio
    idx = np.arange(start, end)
    lta_sum = (cum[idx] - cum[idx - lta_len]) / lta_len
    sta_sum = (cum[idx + sta_len] - cum[idx]) / sta_len
    safe = lta_sum > 1e-30
    ratio_slice = np.zeros(len(idx))
    ratio_slice[safe] = sta_sum[safe] / lta_sum[safe]
    ratio[start:end] = ratio_slice
    return ratio


def compute_energy_ratio_curve(
    waveform: np.ndarray, window_len: int = 10
) -> np.ndarray:
    """计算完整能量比曲线（向量化）"""
    n = len(waveform)
    energy = waveform.astype(np.float64) ** 2
    cum = np.insert(np.cumsum(energy), 0, 0.0)

    er = np.zeros(n)
    start, end = window_len, n - window_len
    if start >= end:
        return er
    idx = np.arange(start, end)
    e_before = cum[idx] - cum[idx - window_len]
    e_after = cum[idx + window_len] - cum[idx]
    safe = e_before > 1e-20
    er_slice = np.zeros(len(idx))
    er_slice[safe] = e_after[safe] / e_before[safe]
    er[start:end] = er_slice
    return er
