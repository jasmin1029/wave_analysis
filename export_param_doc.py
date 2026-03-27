"""
用标准库生成波形特征参数说明 Word 文档（.docx 本质是 ZIP+XML）。
无需 python-docx 依赖。
"""
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

OUT_DIR = Path(__file__).resolve().parent / "batch_output"

SECTIONS = [
    {
        "title": "一、波幅参数",
        "items": [
            ("最大振幅 (V)", "peak_amplitude",
             "有效信号段中电压绝对值的最大值。反映接收信号的最强响应强度，与岩石的声阻抗匹配度和耦合质量直接相关。振幅越大，说明超声波在传播过程中衰减越小。"),
            ("峰峰值 (V)", "peak_to_peak",
             "信号最大值与最小值之差。代表波形的完整振动幅度，比单侧最大振幅更全面地反映信号强度。岩石损伤加剧时峰峰值通常下降。"),
            ("RMS振幅 (V)", "rms_amplitude",
             "均方根振幅，即信号平方均值的平方根。代表信号的等效平均强度（等效'能量水平'），比峰值更稳定、更适合不同波形之间的对比。"),
        ],
    },
    {
        "title": "二、频率参数",
        "items": [
            ("主频 (kHz)", "dominant_freq",
             "频谱中幅值最大的频率分量。反映超声波在岩石中传播后的最主要振动频率。岩石中裂纹发育时，高频成分被优先吸收，主频会向低频偏移（'频率降低效应'）。"),
            ("频谱重心 (kHz)", "centroid_freq",
             "以幅值功率为权重的频率加权平均值。比主频更稳定，反映信号能量在频域上的整体分布中心。重心下移通常意味着岩石内部结构劣化（裂纹、孔隙增加）。"),
            ("带宽 (kHz)", "bandwidth",
             "频谱能量分布的标准差，衡量信号频率的分散程度。带宽窄说明信号集中在某一频段（介质均匀），带宽宽说明信号成分复杂（散射、多次反射较多）。"),
        ],
    },
    {
        "title": "三、能量参数",
        "items": [
            ("总能量", "total_energy",
             "有效信号段所有采样点电压平方之和（Σv²）。反映接收到的超声波的总能量大小。岩石损伤越严重，能量衰减越多，总能量越低。"),
            ("上升时间 (μs)", "rise_time",
             "累积能量从 10% 增长到 90% 所经历的时间。反映能量的释放快慢。上升时间短说明信号能量集中在前段（脉冲尖锐），长则说明信号拖尾严重（散射、多路径传播）。"),
        ],
    },
    {
        "title": "四、波形复杂度参数",
        "items": [
            ("过零率", "zero_crossing_rate",
             "信号穿越零线的次数与总采样点数的比值。反映波形的振荡频繁程度。过零率高说明高频成分丰富，低则低频主导。岩石损伤导致高频吸收后过零率会下降。"),
            ("峰值因子", "crest_factor",
             "峰值与 RMS 的比值（Peak / RMS）。衡量波形的尖锐程度。峰值因子大说明存在突出的尖峰（信号集中、脉冲性强），小说明信号幅值比较均匀。"),
            ("波形因子", "form_factor",
             "RMS 与平均绝对幅值的比值（RMS / |v̄|）。反映波形的形状特征，正弦波约为 1.11。偏离越大说明波形越不规则。"),
            ("峭度 (kurtosis)", "kurtosis",
             "四阶中心矩标准化后减 3（超额峭度）。衡量波形分布的尾部厚薄。正值表示存在极端大振幅的采样（尖峰多），负值表示振幅分布比正态更平坦均匀。接近 0 类似高斯噪声。"),
            ("偏度 (skewness)", "skewness",
             "三阶中心矩标准化值。衡量波形分布的不对称性。正偏表示正向振幅偏大，负偏表示负向偏大。接近 0 说明波形正负对称。非对称可能暗示非线性传播效应。"),
            ("频谱熵 (bits)", "spectral_entropy",
             "频谱功率归一化后计算的信息熵（-Σpᵢlog₂pᵢ）。衡量频域能量分布的均匀/集中程度。熵高说明能量分散在很多频率上（信号复杂），熵低说明能量集中在少数几个频率上（信号简单纯净）。"),
        ],
    },
    {
        "title": "五、衰减参数",
        "items": [
            ("衰减系数 (/s)", "decay_coefficient",
             "对信号包络线（Hilbert 变换求得）进行指数拟合 A·e^(-βt) 所得的 β 值。反映超声波在岩石中的能量衰减速率。系数越大说明衰减越快（岩石吸收强、散射多）。"),
            ("品质因子 Q", "quality_factor_Q",
             "Q = πf₀/β，其中 f₀ 为主频。反映岩石的储能与耗能之比。Q 高说明岩石弹性好、能量损耗小（致密完整），Q 低说明岩石内部损伤大、耗散严重。Q 是岩石力学中评价介质完整性的重要指标。"),
        ],
    },
    {
        "title": "六、信噪比",
        "items": [
            ("信噪比 SNR (dB)", "estimated_snr_db",
             "20·log₁₀(RMS_signal / RMS_noise)，其中 noise 取初至前的噪声段。衡量有效信号相对于背景噪声的强弱比。SNR 高说明信号质量好、拾取可靠；SNR 低（<10 dB）说明信号接近噪声水平，拾取结果需谨慎对待。"),
        ],
    },
]

ENGINEERING_SUMMARY = (
    "在岩石超声加载实验中，随着应力增加/裂纹发育，各参数的典型变化趋势如下：\n\n"
    "• 波速下降 → 裂纹使声波绕行，路径变长\n"
    "• 振幅、能量下降 → 裂纹界面散射和吸收增加\n"
    "• 主频、频谱重心下移 → 高频被优先吸收\n"
    "• 频谱熵升高 → 散射使频率成分更分散\n"
    "• Q 值下降 → 耗散增大\n"
    "• 衰减系数增大 → 包络衰减更快"
)


def make_docx(out_path: Path):
    """生成 .docx (Office Open XML) 文件"""

    CONTENT_TYPES = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '</Types>')

    RELS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '</Relationships>')

    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

    def p(text, bold=False, size=None, heading=None, color=None, space_after=None):
        """生成一个段落 XML"""
        ppr_parts = []
        if heading:
            ppr_parts.append(f'<w:pStyle w:val="{heading}"/>')
        if space_after is not None:
            ppr_parts.append(f'<w:spacing w:after="{space_after}"/>')

        rpr_parts = []
        if bold:
            rpr_parts.append('<w:b/>')
        if size:
            rpr_parts.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
        if color:
            rpr_parts.append(f'<w:color w:val="{color}"/>')

        rpr_xml = f'<w:rPr>{"".join(rpr_parts)}</w:rPr>' if rpr_parts else ''
        ppr_xml = f'<w:pPr>{"".join(ppr_parts)}</w:pPr>' if ppr_parts else ''

        lines = text.split('\n')
        runs = []
        for i, line in enumerate(lines):
            runs.append(f'<w:r>{rpr_xml}<w:t xml:space="preserve">{escape(line)}</w:t></w:r>')
            if i < len(lines) - 1:
                runs.append('<w:r><w:br/></w:r>')

        return f'<w:p>{ppr_xml}{"".join(runs)}</w:p>'

    def table_row(cells, bold=False, shade=None):
        """生成表格行"""
        tc_list = []
        for c in cells:
            shd = f'<w:shd w:val="clear" w:fill="{shade}"/>' if shade else ''
            rpr = '<w:rPr><w:b/><w:color w:val="FFFFFF"/></w:rPr>' if bold else ''
            tc_list.append(
                f'<w:tc><w:tcPr>{shd}</w:tcPr>'
                f'<w:p><w:r>{rpr}<w:t xml:space="preserve">{escape(c)}</w:t></w:r></w:p>'
                f'</w:tc>'
            )
        return f'<w:tr>{"".join(tc_list)}</w:tr>'

    body_parts = []

    body_parts.append(p("超声波形特征参数说明", bold=True, size=36, space_after=200))
    body_parts.append(p("本文档对批量波形分析中提取的每一项特征参数进行详细说明，"
                        "包括定义、计算方法和工程意义。", size=21, space_after=200))

    for section in SECTIONS:
        body_parts.append(p(section["title"], bold=True, size=28, color="2C3E50", space_after=100))

        col_widths = '<w:tblGrid><w:gridCol w:w="2500"/><w:gridCol w:w="2000"/><w:gridCol w:w="5500"/></w:tblGrid>'
        tbl_props = ('<w:tblPr>'
                     '<w:tblStyle w:val="TableGrid"/>'
                     '<w:tblW w:w="10000" w:type="dxa"/>'
                     '<w:tblBorders>'
                     '<w:top w:val="single" w:sz="4" w:color="CCCCCC"/>'
                     '<w:left w:val="single" w:sz="4" w:color="CCCCCC"/>'
                     '<w:bottom w:val="single" w:sz="4" w:color="CCCCCC"/>'
                     '<w:right w:val="single" w:sz="4" w:color="CCCCCC"/>'
                     '<w:insideH w:val="single" w:sz="4" w:color="CCCCCC"/>'
                     '<w:insideV w:val="single" w:sz="4" w:color="CCCCCC"/>'
                     '</w:tblBorders>'
                     '</w:tblPr>')

        rows = [table_row(["参数名称", "代码标识", "含义与工程解释"], bold=True, shade="2C3E50")]
        for i, (name, code, desc) in enumerate(section["items"]):
            shade = "F0F3F5" if i % 2 == 1 else None
            rows.append(table_row([name, code, desc], shade=shade))

        body_parts.append(f'<w:tbl>{tbl_props}{col_widths}{"".join(rows)}</w:tbl>')
        body_parts.append(p("", space_after=200))

    body_parts.append(p("七、工程意义总结", bold=True, size=28, color="2C3E50", space_after=100))
    body_parts.append(p(ENGINEERING_SUMMARY, size=21, space_after=100))

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{W}">'
        f'<w:body>{"".join(body_parts)}'
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>'
        '</w:sectPr></w:body></w:document>'
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', CONTENT_TYPES)
        zf.writestr('_rels/.rels', RELS)
        zf.writestr('word/document.xml', document_xml)

    print(f"Word 文档已保存: {out_path}")
    print(f"  共 {sum(len(s['items']) for s in SECTIONS)} 个参数")


if __name__ == "__main__":
    out = OUT_DIR / "波形特征参数说明.docx"
    make_docx(out)
