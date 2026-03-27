"""
从项目文件夹读取示波器/采集 CSV（Time, Voltage），进行 AIC 初至拾取与波形特征分析。

默认处理当前目录下所有 *.csv，排除 aic_summary.csv；结果写入 aic_summary.csv 并保存图像。
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from first_arrival_picking import aic_method_two_stage, compute_aic_curve
from waveform_analysis import analyze_waveform, print_analysis_report

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent


def _find_time_voltage_columns(header: list[str]) -> tuple[int, int]:
    """根据表头定位时间与电压列索引。"""
    time_idx = voltage_idx = None
    for i, h in enumerate(header):
        h_low = h.strip().lower()
        if time_idx is None and ("time" in h_low or h_low.startswith("t")):
            time_idx = i
        if voltage_idx is None and ("volt" in h_low or "ch" in h_low or h_low in ("v", "y")):
            voltage_idx = i
    if time_idx is None:
        time_idx = 0
    if voltage_idx is None:
        voltage_idx = 1 if len(header) > 1 else 0
    return time_idx, voltage_idx


def load_scope_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取两列及以上数值 CSV，返回 time, voltage（float）。"""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"空文件: {path}")

    start = 0
    header = rows[0]
    try:
        float(header[0])
    except (ValueError, TypeError):
        start = 1

    data_rows = rows[start:]
    if not data_rows:
        raise ValueError(f"无数据行: {path}")

    if start == 1:
        t_i, v_i = _find_time_voltage_columns(header)
    else:
        t_i, v_i = 0, 1

    times: list[float] = []
    volts: list[float] = []
    for parts in data_rows:
        if len(parts) <= max(t_i, v_i):
            continue
        try:
            times.append(float(parts[t_i]))
            volts.append(float(parts[v_i]))
        except ValueError:
            continue

    if len(times) < 8:
        raise ValueError(f"有效数据点过少: {path}")

    return np.asarray(times, dtype=np.float64), np.asarray(volts, dtype=np.float64)


def detect_time_unit(time_arr: np.ndarray) -> tuple[np.ndarray, str]:
    """
    自动检测时间列单位。Tektronix 等示波器 CSV 的时间列标注为 Time(s)，
    但对于超声数据，值域 ±数百 通常表示微秒。
    判据：dt 中值 > 1e-3 且值域跨度 > 10 时，视为 μs 并转换为 s。
    """
    dt = np.median(np.abs(np.diff(time_arr)))
    span = np.ptp(time_arr)

    if dt > 1e-3 and span > 10:
        return time_arr * 1e-6, "μs→s"
    return time_arr, "s"


def estimate_fs(time_s: np.ndarray) -> float:
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("无法从时间列估计采样间隔")
    return float(1.0 / np.median(dt))


def preprocess_voltage(v: np.ndarray) -> np.ndarray:
    """去直流（均值）。"""
    return v - np.mean(v)


def plot_csv_pick(
    path: Path,
    time_s: np.ndarray,
    waveform: np.ndarray,
    onset_idx: int,
    fs: float,
    out_png: Path,
) -> None:
    t_us = time_s * 1e6
    aic_curve = compute_aic_curve(waveform)
    aic_min, aic_max = np.nanmin(aic_curve), np.nanmax(aic_curve)
    span = aic_max - aic_min
    if not np.isfinite(span) or span < 1e-30:
        aic_norm = np.zeros_like(aic_curve)
    else:
        aic_norm = (aic_curve - aic_min) / span

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(f"AIC 初至拾取 — {path.name}", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(t_us, waveform, "k-", lw=0.5, alpha=0.85, label="电压 (去直流)")
    ax.axvline(time_s[onset_idx] * 1e6, color="#3498db", ls="--", lw=2, label=f"AIC 初至 t={time_s[onset_idx]*1e6:.2f} μs")
    ax.set_xlabel("时间 (μs)")
    ax.set_ylabel("电压 (V)")
    ax.set_title("波形与初至")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(t_us, aic_curve, "b-", lw=0.7, label="AIC")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(t_us, aic_norm, color="gray", lw=0.4, alpha=0.6, label="AIC 归一化")
    ax2.axvline(time_s[onset_idx] * 1e6, color="#3498db", ls="--", lw=2)
    ax2.set_xlabel("时间 (μs)")
    ax2.set_ylabel("AIC 值")
    ax2.set_title("AIC 曲线与拾取位置")
    ax2.grid(True, alpha=0.3)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_waveform_analysis_figure(
    time_s: np.ndarray,
    waveform: np.ndarray,
    analysis_result: dict,
    fs: float,
    out_png: Path,
    title_suffix: str = "",
) -> None:
    """与 main.py 中逻辑一致的 6 子图分析图（略去重复导入 main）。"""
    onset = analysis_result["onset_index"]
    t_us = time_s * 1e6
    freq_data = analysis_result["frequency"]
    eng_data = analysis_result["energy"]
    atten_data = analysis_result["attenuation"]
    comp_data = analysis_result["complexity"]
    amp_data = analysis_result["amplitude"]

    fig, axes = plt.subplots(3, 2, figsize=(15, 14), constrained_layout=True)
    fig.suptitle("AIC 初至后 — 波形特征分析" + title_suffix, fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(t_us, waveform, "k-", linewidth=0.5, alpha=0.7, label="波形")
    ax.plot(t_us[onset:], atten_data["envelope"], "r-", linewidth=1.0, alpha=0.8, label="包络线")
    ax.axvline(t_us[onset], color="#3498db", ls="--", lw=2, label=f"AIC 初至 ({t_us[onset]:.1f} μs)")
    ax.set_xlabel("时间 (μs)")
    ax.set_ylabel("振幅")
    ax.set_title("波形、包络与初至位置")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    freq_khz = freq_data["freq_axis"] / 1e3
    spectrum = freq_data["spectrum"]
    ax.plot(freq_khz, spectrum, color="#e74c3c", linewidth=0.8)
    ax.axvline(
        freq_data["dominant_freq"] / 1e3,
        color="#3498db",
        ls="--",
        lw=1.5,
        label=f"主频 {freq_data['dominant_freq']/1e3:.1f} kHz",
    )
    ax.axvline(
        freq_data["centroid_freq"] / 1e3,
        color="#2ecc71",
        ls="--",
        lw=1.5,
        label=f"重心 {freq_data['centroid_freq']/1e3:.1f} kHz",
    )
    ax.fill_between(freq_khz, spectrum, alpha=0.15, color="#e74c3c")
    ax.set_xlabel("频率 (kHz)")
    ax.set_ylabel("幅值谱")
    ax.set_title("频谱分析")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, freq_khz[-1] * 0.6)

    ax = axes[1, 0]
    cum_energy = eng_data["cumulative_energy"]
    eff_t_us = np.arange(len(cum_energy)) / fs * 1e6
    ax.plot(eff_t_us, cum_energy, color="#9b59b6", linewidth=1.2)
    ax.axhline(0.1, color="gray", ls=":", lw=1, alpha=0.6)
    ax.axhline(0.9, color="gray", ls=":", lw=1, alpha=0.6)
    ax.fill_between(eff_t_us, cum_energy, alpha=0.1, color="#9b59b6")
    ax.set_xlabel("初至后时间 (μs)")
    ax.set_ylabel("归一化累积能量")
    ax.set_title(f"累积能量 (上升时间: {eng_data['rise_time_s']*1e6:.1f} μs)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    envelope = atten_data["envelope"]
    peak_idx = int(np.argmax(envelope))
    env_after = envelope[peak_idx:]
    t_after_us = np.arange(len(env_after)) / fs * 1e6
    ax.plot(t_after_us, env_after, "k-", linewidth=0.8, alpha=0.7, label="实际包络")
    decay_c = atten_data["decay_coefficient"]
    if decay_c > 0:
        t_fit = np.arange(len(env_after)) / fs
        fitted = envelope[peak_idx] * np.exp(-decay_c * t_fit)
        ax.plot(t_after_us, fitted, "r--", linewidth=1.5, label=f"拟合 exp(-{decay_c:.0f}t)")
    ax.set_xlabel("峰值后时间 (μs)")
    ax.set_ylabel("包络幅值")
    ax.set_title(f"包络衰减 (Q = {atten_data['quality_factor_Q']:.1f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    axes[2, 0].remove()
    ax_polar = fig.add_subplot(3, 2, 5, polar=True)
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
    ax_polar.plot(angles_closed, normalized_closed, "o-", color="#e74c3c", linewidth=1.5, markersize=5)
    ax_polar.fill(angles_closed, normalized_closed, alpha=0.15, color="#e74c3c")
    ax_polar.set_xticks(angles)
    ax_polar.set_xticklabels(categories, fontsize=9)
    ax_polar.set_title("波形复杂度 (归一化)", fontsize=11, pad=15)

    ax = axes[2, 1]
    ax.axis("off")
    table_data = [
        ["参数类别", "指标", "数值"],
        ["波幅", "最大振幅", f"{amp_data['peak_amplitude']:.6f}"],
        ["波幅", "峰峰值", f"{amp_data['peak_to_peak']:.6f}"],
        ["波幅", "RMS", f"{amp_data['rms_amplitude']:.6f}"],
        ["频率", "主频", f"{freq_data['dominant_freq']/1e3:.2f} kHz"],
        ["频率", "频谱重心", f"{freq_data['centroid_freq']/1e3:.2f} kHz"],
        ["频率", "带宽", f"{freq_data['bandwidth']/1e3:.2f} kHz"],
        ["能量", "总能量", f"{eng_data['total_energy']:.4e}"],
        ["能量", "上升时间", f"{eng_data['rise_time_s']*1e6:.2f} μs"],
        ["复杂度", "过零率", f"{comp_data['zero_crossing_rate']:.4f}"],
        ["复杂度", "峭度", f"{comp_data['kurtosis']:.4f}"],
        ["复杂度", "频谱熵", f"{comp_data['spectral_entropy']:.2f} bits"],
        ["衰减", "衰减系数", f"{atten_data['decay_coefficient']:.1f} /s"],
        ["衰减", "Q", f"{atten_data['quality_factor_Q']:.1f}"],
        ["SNR", "估算", f"{analysis_result['estimated_snr_db']:.1f} dB"],
    ]
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for j in range(3):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for row in range(1, len(table_data)):
        if row % 2 == 0:
            for col in range(3):
                table[row, col].set_facecolor("#f0f3f5")
    ax.set_title("特征参数汇总", fontsize=11, pad=15)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def safe_stem(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE).strip("_") or "waveform"


def iter_waveform_csvs(folder: Path) -> list[Path]:
    skip = {"aic_summary.csv"}
    paths = sorted(folder.glob("*.csv"))
    return [p for p in paths if p.name.lower() not in {s.lower() for s in skip}]


def find_excitation_end(waveform: np.ndarray, fs: float,
                        threshold_ratio: float = 0.10,
                        quiet_duration_s: float = 5e-6) -> int:
    """
    检测激励脉冲结束位置。
    找到高振幅区结束后、信号回到噪声水平的采样点索引，
    再加一段静默缓冲区，确保完全跳过激励残余。
    """
    abs_w = np.abs(waveform)
    peak = np.max(abs_w)

    noise_est = np.median(abs_w[:min(100, len(abs_w) // 10)])
    thresh = max(noise_est * 5, peak * threshold_ratio)

    above = np.where(abs_w > thresh)[0]
    if len(above) == 0:
        return 0

    last_above = above[-1]
    for i in range(above[0], len(abs_w)):
        if abs_w[i] <= thresh:
            last_above = i
            quiet_samples = int(quiet_duration_s * fs)
            remaining = abs_w[i:i + quiet_samples]
            if len(remaining) == quiet_samples and np.all(remaining <= thresh):
                break

    buffer_samples = int(quiet_duration_s * fs)
    end_idx = min(last_above + buffer_samples, len(waveform) - 1)
    return end_idx


def process_one_csv(csv_path: Path, out_dir: Path) -> dict:
    time_raw, voltage = load_scope_csv(csv_path)
    time_s, unit_note = detect_time_unit(time_raw)
    if unit_note != "s":
        print(f"  [时间单位自动修正] 原始值域 {time_raw[0]:.2f}~{time_raw[-1]:.2f}，"
              f"判定为 μs，已转换为秒")
    fs = estimate_fs(time_s)
    print(f"  采样率: {fs/1e6:.3f} MHz  ({len(time_s)} 点)")
    waveform = preprocess_voltage(voltage)

    excitation_end = find_excitation_end(waveform, fs)
    if excitation_end > 0:
        print(f"  [激励脉冲检测] 跳过前 {excitation_end} 点"
              f" (至 {time_s[excitation_end]*1e6:.2f} μs)，在后续信号上拾取初至")

    search_waveform = waveform[excitation_end:]
    onset_local = aic_method_two_stage(
        search_waveform,
        sta_len=max(5, int(fs * 2e-6)),
        lta_len=max(50, int(fs * 20e-6)),
        trigger_ratio=3.0,
        window_expand=max(50, int(fs * 20e-6)),
    )
    onset_idx = int(np.clip(excitation_end + onset_local, 0, len(waveform) - 1))

    analysis = analyze_waveform(waveform, onset_idx, fs)

    stem = safe_stem(csv_path.stem)
    pick_png = out_dir / f"{stem}_aic_pick.png"
    analysis_png = out_dir / f"{stem}_waveform_analysis.png"

    plot_csv_pick(csv_path, time_s, waveform, onset_idx, fs, pick_png)
    plot_waveform_analysis_figure(
        time_s, waveform, analysis, fs, analysis_png, title_suffix=f" — {csv_path.name}"
    )

    print_analysis_report(analysis)

    row = {
        "file": csv_path.name,
        "n_samples": len(waveform),
        "fs_Hz": fs,
        "dt_median_s": 1.0 / fs,
        "onset_index": onset_idx,
        "onset_time_s": float(time_s[onset_idx]),
        "onset_time_us_from_start": float((time_s[onset_idx] - time_s[0]) * 1e6),
        "dominant_freq_Hz": analysis["frequency"]["dominant_freq"],
        "centroid_freq_Hz": analysis["frequency"]["centroid_freq"],
        "bandwidth_Hz": analysis["frequency"]["bandwidth"],
        "peak_amplitude": analysis["amplitude"]["peak_amplitude"],
        "peak_to_peak": analysis["amplitude"]["peak_to_peak"],
        "rms_amplitude": analysis["amplitude"]["rms_amplitude"],
        "total_energy": analysis["energy"]["total_energy"],
        "rise_time_us": analysis["energy"]["rise_time_s"] * 1e6,
        "estimated_snr_db": analysis["estimated_snr_db"],
        "pick_plot": str(pick_png.name),
        "analysis_plot": str(analysis_png.name),
    }
    return row


def write_summary_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    folder = ROOT
    out_dir = folder / "csv_processing_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = iter_waveform_csvs(folder)
    if not csv_files:
        print(f"未在 {folder} 找到可处理的 CSV（已排除 aic_summary.csv）。")
        return

    print(f"将处理 {len(csv_files)} 个文件，输出目录: {out_dir}")
    rows: list[dict] = []

    for p in csv_files:
        print(f"\n--- {p.name} ---")
        try:
            row = process_one_csv(p, out_dir)
            rows.append(row)
        except Exception as e:
            print(f"处理失败: {e}")
            rows.append({"file": p.name, "error": str(e)})

    summary_path = folder / "aic_summary.csv"
    # 仅写入成功行带完整字段；若有 error 行，统一字段
    ok_rows = [r for r in rows if "error" not in r]
    if ok_rows:
        write_summary_csv(ok_rows, summary_path)
        print(f"\n汇总已写入: {summary_path}")
    print("完成。")


if __name__ == "__main__":
    main()
