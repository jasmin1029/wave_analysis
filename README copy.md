# 岩石超声波形初至提取

基于 Python 实现的多种岩石超声波形初至（First Arrival）拾取算法，适用于岩石力学实验中的超声波传播时间测量。

## 算法列表

| 算法 | 说明 | 适用场景 |
|------|------|----------|
| **阈值法** | 幅值首次超过阈值即为初至 | 高信噪比、快速估算 |
| **STA/LTA 法** | 短时窗/长时窗能量比触发 | 通用场景，工程常用 |
| **AIC 法** | Akaike 信息准则最小值定位 | 高精度需求，学术研究 |
| **两步法** | STA/LTA 粗定 + AIC 精拾 | 兼顾效率与精度，推荐 |
| **能量比法** | 前后窗口能量比最大值 | 噪声适中的场景 |
| **改进能量比法** | 能量比 × 振幅增强 | 初至不明显的波形 |
| **包络阈值法** | Hilbert 包络 + 阈值 | 高频噪声环境 |

## 项目结构

```
rock_waveform_analysis/
├── first_arrival_picking.py   # 核心算法模块
├── main.py                    # 演示脚本（模拟波形 + 对比可视化）
├── requirements.txt           # 依赖列表
└── README.md
```

## 安装与运行

```bash
# 安装依赖
pip install -r requirements.txt

# 运行演示
python main.py
```

## 输出

运行 `main.py` 后会：

1. 生成模拟的岩石超声波形（含噪声）
2. 调用全部 7 种算法进行初至拾取
3. 在控制台打印各算法的拾取结果和误差
4. 输出 4 子图对比图 `first_arrival_result.png`
5. 输出不同信噪比下的算法误差对比图 `snr_comparison.png`

## 使用自己的数据

在 `main.py` 中替换 `generate_synthetic_waveform()` 为自己的数据加载逻辑即可：

```python
import numpy as np
from first_arrival_picking import threshold_method, aic_method

# 加载你的波形数据
waveform = np.loadtxt("your_waveform.txt")

# 选择算法
pick = aic_method(waveform)
print(f"初至位置: 第 {pick} 个采样点")
```

## 参考文献

- Maeda, N. (1985). A method for reading and checking phase time in auto-processing system of seismic wave data. *Zisin*, 38(3), 365-379.
- Allen, R.V. (1978). Automatic earthquake recognition and timing from single traces. *BSSA*, 68(5), 1521-1532.
- Wong, J. et al. (2009). Automatic time-picking of first arrivals on noisy microseismic data. *CSEG Convention*.
