"""
岩石超声波形初至提取与特征分析 —— 演示脚本

生成模拟的岩石超声波形，调用多种初至拾取算法并对比可视化，
然后基于 AIC 拾取结果对波形做定量特征分析。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from first_arrival_picking import (
    threshold_method,
    sta_lta_method,
    aic_method,
    aic_method_two_stage,
    energy_ratio_method,
    modified_energy_ratio_method,
    envelope_threshold_method,
    compute_aic_curve,
    compute_sta_lta_curve,
    compute_energy_ratio_curve,
)
from waveform_analysis import analyze_waveform, print_analysis_report

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def generate_synthetic_waveform(
    n_samples: int = 1024,
    fs: float = 1e6,
    arrival_sample: int = 200,
    frequency: float = 100e3,
    snr_db: float = 20.0,
    decay: float = 3000.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    生成模拟岩石超声波形

    Parameters
    ----------
    n_samples : 总采样点数
    fs : 采样率 (Hz)
    arrival_sample : 真实初至位置（采样点索引）
    frequency : 中心频率 (Hz)
    snr_db : 信噪比 (dB)
    decay : 指数衰减系数

    Returns
    -------
    time : 时间轴 (s)
    waveform : 波形数组
    arrival_sample : 真实初至位置
    """
    t = np.arange(n_samples) / fs

    signal = np.zeros(n_samples)
    t_signal = np.arange(n_samples - arrival_sample) / fs
    envelope = np.exp(-decay * t_signal) * (1 - np.exp(-decay * 3 * t_signal))
    signal[arrival_sample:] = envelope * np.sin(2 * np.pi * frequency * t_signal)

    signal_power = np.mean(signal[arrival_sample:] ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(n_samples)

    waveform = signal + noise
    return t, waveform, arrival_sample


def run_all_methods(waveform: np.ndarray) -> dict[str, int]:
    """对波形运行所有初至拾取算法，返回各算法拾取结果"""
    return {
        "阈值法": threshold_method(waveform, threshold_ratio=0.05),
        "STA/LTA 法": sta_lta_method(waveform, sta_len=5, lta_len=50, trigger_ratio=3.0),
        "AIC 法": aic_method(waveform),
        "两步法 (STA/LTA + AIC)": aic_method_two_stage(waveform),
        "能量比法": energy_ratio_method(waveform, window_len=10),
        "改进能量比法": modified_energy_ratio_method(waveform, window_len=10),
        "包络阈值法": envelope_threshold_method(waveform, threshold_ratio=0.05),
    }


def plot_results(
    time: np.ndarray,
    waveform: np.ndarray,
    true_arrival: int,
    picks: dict[str, int],
    fs: float,
):
    """绘制初至拾取结果对比图"""
    colors = [
        "#e74c3c", "#2ecc71", "#3498db", "#9b59b6",
        "#f39c12", "#1abc9c", "#e67e22",
    ]

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), constrained_layout=True)
    fig.suptitle("岩石超声波形初至提取 —— 多方法对比", fontsize=16, fontweight="bold")

    t_us = time * 1e6

    # ----- 子图 1：波形 + 所有拾取标记 -----
    ax = axes[0]
    ax.plot(t_us, waveform, "k-", linewidth=0.6, alpha=0.8, label="波形")
    ax.axvline(true_arrival / fs * 1e6, color="gray", ls="--", lw=2, label=f"真实初至 (#{true_arrival})")

    for i, (name, idx) in enumerate(picks.items()):
        ax.axvline(idx / fs * 1e6, color=colors[i % len(colors)], ls="-", lw=1.2,
                   alpha=0.85, label=f"{name} (#{idx})")

    ax.set_xlabel("时间 (μs)")
    ax.set_ylabel("幅值")
    ax.set_title("波形与各方法拾取结果")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ----- 子图 2：AIC 曲线 -----
    ax = axes[1]
    aic_curve = compute_aic_curve(waveform)
    ax.plot(t_us, aic_curve, "b-", linewidth=0.8)
    aic_pick = picks.get("AIC 法", 0)
    ax.axvline(aic_pick / fs * 1e6, color="#3498db", ls="-", lw=1.5, label=f"AIC 拾取 (#{aic_pick})")
    ax.axvline(true_arrival / fs * 1e6, color="gray", ls="--", lw=1.5, label="真实初至")
    ax.set_xlabel("时间 (μs)")
    ax.set_ylabel("AIC 值")
    ax.set_title("AIC 函数曲线")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ----- 子图 3：STA/LTA 曲线 -----
    ax = axes[2]
    sta_lta_curve = compute_sta_lta_curve(waveform, sta_len=5, lta_len=50)
    ax.plot(t_us, sta_lta_curve, "g-", linewidth=0.8)
    ax.axhline(3.0, color="r", ls=":", lw=1, label="触发阈值=3.0")
    stalta_pick = picks.get("STA/LTA 法", 0)
    ax.axvline(stalta_pick / fs * 1e6, color="#2ecc71", ls="-", lw=1.5, label=f"STA/LTA 拾取 (#{stalta_pick})")
    ax.axvline(true_arrival / fs * 1e6, color="gray", ls="--", lw=1.5, label="真实初至")
    ax.set_xlabel("时间 (μs)")
    ax.set_ylabel("STA/LTA 比值")
    ax.set_title("STA/LTA 比值曲线")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ----- 子图 4：能量比曲线 -----
    ax = axes[3]
    er_curve = compute_energy_ratio_curve(waveform, window_len=10)
    ax.plot(t_us, er_curve, color="#f39c12", linewidth=0.8)
    er_pick = picks.get("能量比法", 0)
    ax.axvline(er_pick / fs * 1e6, color="#f39c12", ls="-", lw=1.5, label=f"能量比拾取 (#{er_pick})")
    ax.axvline(true_arrival / fs * 1e6, color="gray", ls="--", lw=1.5, label="真实初至")
    ax.set_xlabel("时间 (μs)")
    ax.set_ylabel("能量比")
    ax.set_title("能量比曲线")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.savefig("first_arrival_result.png", dpi=150, bbox_inches="tight")
    print("图片已保存: first_arrival_result.png")
    plt.show()


def plot_snr_comparison(fs: float = 1e6):
    """不同信噪比下各算法误差对比"""
    snr_list = [5, 10, 15, 20, 30, 40]
    n_trials = 20
    true_arrival = 200

    method_names = [
        "阈值法", "STA/LTA 法", "AIC 法", "两步法 (STA/LTA + AIC)",
        "能量比法", "改进能量比法", "包络阈值法",
    ]
    errors = {name: [] for name in method_names}

    for snr in snr_list:
        trial_errors = {name: [] for name in method_names}
        for _ in range(n_trials):
            _, waveform, _ = generate_synthetic_waveform(
                arrival_sample=true_arrival, snr_db=snr
            )
            picks = run_all_methods(waveform)
            for name, idx in picks.items():
                trial_errors[name].append(abs(idx - true_arrival))

        for name in method_names:
            errors[name].append(np.mean(trial_errors[name]))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [
        "#e74c3c", "#2ecc71", "#3498db", "#9b59b6",
        "#f39c12", "#1abc9c", "#e67e22",
    ]
    for i, name in enumerate(method_names):
        ax.plot(snr_list, errors[name], "o-", color=colors[i], label=name, linewidth=1.5)

    ax.set_xlabel("信噪比 SNR (dB)", fontsize=12)
    ax.set_ylabel("平均绝对误差（采样点）", fontsize=12)
    ax.set_title("不同信噪比下各初至拾取算法误差对比", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.savefig("snr_comparison.png", dpi=150, bbox_inches="tight")
    print("图片已保存: snr_comparison.png")
    plt.show()


def plot_waveform_analysis(time, waveform, analysis_result, fs):
    """绘制 AIC 初至拾取后的波形特征分析图 (6 子图)"""
    onset = analysis_result["onset_index"]
    t_us = time * 1e6
    eff_signal = waveform[onset:]
    eff_time_us = t_us[onset:]

    freq_data = analysis_result["frequency"]
    eng_data = analysis_result["energy"]
    atten_data = analysis_result["attenuation"]
    comp_data = analysis_result["complexity"]
    amp_data = analysis_result["amplitude"]

    fig, axes = plt.subplots(3, 2, figsize=(15, 14), constrained_layout=True)
    fig.suptitle("AIC 初至拾取后 —— 波形特征定量分析", fontsize=16, fontweight="bold")

    # ---- 子图 1: 波形 + 初至标记 + 包络 ----
    ax = axes[0, 0]
    ax.plot(t_us, waveform, "k-", linewidth=0.5, alpha=0.7, label="原始波形")
    full_envelope = np.zeros_like(waveform)
    full_envelope[onset:] = atten_data["envelope"]
    ax.plot(t_us[onset:], atten_data["envelope"], "r-", linewidth=1.0, alpha=0.8, label="包络线")
    ax.axvline(t_us[onset], color="#3498db", ls="--", lw=2,
               label=f"AIC 初至 ({t_us[onset]:.1f} us)")
    ax.set_xlabel("时间 (us)")
    ax.set_ylabel("振幅")
    ax.set_title("波形、包络与初至位置")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ---- 子图 2: 频谱图 ----
    ax = axes[0, 1]
    freq_khz = freq_data["freq_axis"] / 1e3
    spectrum = freq_data["spectrum"]
    ax.plot(freq_khz, spectrum, color="#e74c3c", linewidth=0.8)
    ax.axvline(freq_data["dominant_freq"] / 1e3, color="#3498db", ls="--", lw=1.5,
               label=f"主频 {freq_data['dominant_freq']/1e3:.1f} kHz")
    ax.axvline(freq_data["centroid_freq"] / 1e3, color="#2ecc71", ls="--", lw=1.5,
               label=f"重心频率 {freq_data['centroid_freq']/1e3:.1f} kHz")
    ax.fill_between(
        freq_khz,
        spectrum,
        alpha=0.15, color="#e74c3c",
    )
    ax.set_xlabel("频率 (kHz)")
    ax.set_ylabel("幅值谱")
    ax.set_title("频谱分析")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, freq_khz[-1] * 0.6)

    # ---- 子图 3: 归一化累积能量曲线 ----
    ax = axes[1, 0]
    cum_energy = eng_data["cumulative_energy"]
    eff_t_us = np.arange(len(cum_energy)) / fs * 1e6
    ax.plot(eff_t_us, cum_energy, color="#9b59b6", linewidth=1.2)
    ax.axhline(0.1, color="gray", ls=":", lw=1, alpha=0.6, label="10%")
    ax.axhline(0.9, color="gray", ls=":", lw=1, alpha=0.6, label="90%")
    ax.fill_between(eff_t_us, cum_energy, alpha=0.1, color="#9b59b6")
    ax.set_xlabel("初至后时间 (us)")
    ax.set_ylabel("归一化累积能量")
    ax.set_title(f"累积能量曲线 (上升时间: {eng_data['rise_time_s']*1e6:.1f} us)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- 子图 4: 包络衰减拟合 ----
    ax = axes[1, 1]
    envelope = atten_data["envelope"]
    peak_idx = np.argmax(envelope)
    env_after = envelope[peak_idx:]
    t_after_us = np.arange(len(env_after)) / fs * 1e6

    ax.plot(t_after_us, env_after, "k-", linewidth=0.8, alpha=0.7, label="实际包络")

    decay_c = atten_data["decay_coefficient"]
    if decay_c > 0:
        t_fit = np.arange(len(env_after)) / fs
        fitted = envelope[peak_idx] * np.exp(-decay_c * t_fit)
        ax.plot(t_after_us, fitted, "r--", linewidth=1.5,
                label=f"拟合: exp(-{decay_c:.0f}t)")

    ax.set_xlabel("峰值后时间 (us)")
    ax.set_ylabel("包络幅值")
    ax.set_title(f"包络衰减拟合 (Q = {atten_data['quality_factor_Q']:.1f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- 子图 5: 波形复杂度雷达图 ----
    ax = axes[2, 0]
    ax.set_aspect("equal")
    categories = ["过零率", "峰值因子", "波形因子", "脉冲因子", "峭度", "频谱熵"]
    raw_values = [
        comp_data["zero_crossing_rate"],
        comp_data["crest_factor"],
        comp_data["form_factor"],
        comp_data["impulse_factor"],
        abs(comp_data["kurtosis"]),
        comp_data["spectral_entropy"],
    ]
    max_vals = [max(v, 1e-10) for v in raw_values]
    global_max = max(max_vals)
    normalized = [v / global_max for v in raw_values]

    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    normalized_closed = normalized + [normalized[0]]
    angles_closed = angles + [angles[0]]

    ax.clear()
    ax = fig.add_subplot(3, 2, 5, polar=True)
    ax.plot(angles_closed, normalized_closed, "o-", color="#e74c3c", linewidth=1.5, markersize=5)
    ax.fill(angles_closed, normalized_closed, alpha=0.15, color="#e74c3c")
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_title("波形复杂度指标 (归一化)", fontsize=11, pad=15)

    # ---- 子图 6: 参数汇总表 ----
    ax = axes[2, 1]
    ax.axis("off")
    table_data = [
        ["参数类别", "指标", "数值"],
        ["波幅", "最大振幅", f"{amp_data['peak_amplitude']:.6f}"],
        ["波幅", "峰峰值", f"{amp_data['peak_to_peak']:.6f}"],
        ["波幅", "RMS 振幅", f"{amp_data['rms_amplitude']:.6f}"],
        ["频率", "主频", f"{freq_data['dominant_freq']/1e3:.2f} kHz"],
        ["频率", "频谱重心", f"{freq_data['centroid_freq']/1e3:.2f} kHz"],
        ["频率", "带宽", f"{freq_data['bandwidth']/1e3:.2f} kHz"],
        ["能量", "总能量", f"{eng_data['total_energy']:.4e}"],
        ["能量", "上升时间", f"{eng_data['rise_time_s']*1e6:.2f} us"],
        ["复杂度", "过零率", f"{comp_data['zero_crossing_rate']:.4f}"],
        ["复杂度", "峭度", f"{comp_data['kurtosis']:.4f}"],
        ["复杂度", "频谱熵", f"{comp_data['spectral_entropy']:.2f} bits"],
        ["衰减", "衰减系数", f"{atten_data['decay_coefficient']:.1f} /s"],
        ["衰减", "品质因子 Q", f"{atten_data['quality_factor_Q']:.1f}"],
        ["信噪比", "估算 SNR", f"{analysis_result['estimated_snr_db']:.1f} dB"],
    ]

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for i in range(3):
        table[0, i].set_facecolor("#2c3e50")
        table[0, i].set_text_props(color="white", fontweight="bold")
    for row in range(1, len(table_data)):
        if row % 2 == 0:
            for col in range(3):
                table[row, col].set_facecolor("#f0f3f5")
    ax.set_title("特征参数汇总", fontsize=11, pad=15)

    plt.savefig("waveform_analysis_result.png", dpi=150, bbox_inches="tight")
    print("分析图已保存: waveform_analysis_result.png")
    plt.show()


def main():
    print("=" * 60)
    print("  岩石超声波形初至提取与特征分析")
    print("=" * 60)

    fs = 1e6
    np.random.seed(42)
    time, waveform, true_arrival = generate_synthetic_waveform(
        n_samples=1024, fs=fs, arrival_sample=200,
        frequency=100e3, snr_db=20.0
    )

    picks = run_all_methods(waveform)

    print(f"\n真实初至位置: 第 {true_arrival} 个采样点 ({true_arrival / fs * 1e6:.1f} μs)")
    print("-" * 50)

    for name, idx in picks.items():
        error = idx - true_arrival
        print(f"  {name:20s}  →  第 {idx:4d} 点  ({idx / fs * 1e6:7.1f} μs)  误差: {error:+d} 点")

    print("-" * 50)
    print("\n正在绘制初至拾取结果对比图...")
    plot_results(time, waveform, true_arrival, picks, fs)

    print("\n正在进行不同信噪比对比分析...")
    plot_snr_comparison(fs)

    # ===== AIC 初至后的波形特征分析 =====
    aic_onset = picks["AIC 法"]
    print(f"\n基于 AIC 拾取结果 (第 {aic_onset} 点) 进行波形特征分析...")
    analysis_result = analyze_waveform(waveform, aic_onset, fs)
    print_analysis_report(analysis_result)

    print("\n正在绘制波形特征分析图...")
    plot_waveform_analysis(time, waveform, analysis_result, fs)

    print("\n全部完成！")


if __name__ == "__main__":
    main()
