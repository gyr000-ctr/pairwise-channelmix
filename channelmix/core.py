"""channelmix 核心 API。

声道矩阵混音，包含：

- 单位矩阵原样透传（不做任何归一化）
- 矩阵维度 / 声道数 / 非有限样本校验
- 下混响度归一（简化 BS.1770：均方能量匹配，省略 K 加权与门限）
- 削波保护（峰值限幅 + 硬削波兜底，保证输出落在 [-1, 1]）
- 确定性、无状态（线程安全）、不修改输入

双后端：优先使用 numpy（若环境可用），否则回退到纯标准库实现；
两条路径的算法与数值口径完全一致。
"""

import math
import threading

try:  # numpy 为可选加速后端，缺失时纯标准库回退
    import numpy as _np
except Exception:  # pragma: no cover - 取决于运行环境
    _np = None

# 矩阵维度上限，防止异常输入导致资源耗尽
MAX_CHANNELS = 64


def _is_sequence(value):
    return isinstance(value, (list, tuple))


def _validate_matrix(matrix):
    """校验矩阵并返回 (float 元组矩阵, 行数, 列数)。"""
    if not _is_sequence(matrix) or len(matrix) == 0:
        raise ValueError("matrix must be a non-empty sequence of rows")
    rows = len(matrix)
    cols = None
    normalized = []
    for row in matrix:
        if not _is_sequence(row):
            raise ValueError("matrix row must be a sequence")
        if cols is None:
            cols = len(row)
            if cols == 0:
                raise ValueError("matrix row must be non-empty")
        elif len(row) != cols:
            raise ValueError("matrix rows have inconsistent lengths")
        out_row = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("matrix entries must be real numbers")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("matrix entries must be finite")
            out_row.append(value)
        normalized.append(tuple(out_row))
    if rows > MAX_CHANNELS or cols > MAX_CHANNELS:
        raise ValueError(
            "matrix dimension exceeds MAX_CHANNELS (%d)" % MAX_CHANNELS
        )
    return tuple(normalized), rows, cols


def _is_identity(matrix, rows, cols):
    if rows != cols:
        return False
    for r in range(rows):
        row = matrix[r]
        for c in range(cols):
            if row[c] != (1.0 if r == c else 0.0):
                return False
    return True


def _build_pure_mixer(matrix, rows, cols):
    """生成专用混音函数，避免 apply 热路径上的动态下标开销。"""
    params = ", ".join("c%d" % j for j in range(cols))
    binds = ", ".join("a%d" % j for j in range(cols))
    src_lines = ["def mix(%s):" % params, "    return ["]
    for r in range(rows):
        terms = " + ".join(
            "(%r)*a%d" % (matrix[r][j], j) for j in range(cols)
        )
        src_lines.append(
            "        [%s for %s in zip(%s)]," % (terms, binds, params)
        )
    src_lines.append("    ]")
    namespace = {}
    exec("\n".join(src_lines), namespace)
    return namespace["mix"]


class ChannelMix:
    """固定混音矩阵的可复用混音器（构造后不可变，可并发调用）。"""

    def __init__(self, matrix):
        """按矩阵构造：matrix 为 R 行 C 列实数矩阵（R 输出，C 输入）。"""
        validated, rows, cols = _validate_matrix(matrix)
        self._matrix = validated
        self._rows = rows
        self._cols = cols
        self._identity = _is_identity(validated, rows, cols)
        self._lock = threading.Lock()  # 仅守护惰性缓存，apply 路径无写状态
        self._pure_mixer = None
        self._np_matrix = None
        if _np is not None:
            arr = _np.asarray(validated, dtype=_np.float64, order="C")
            arr.setflags(write=False)
            self._np_matrix = arr

    # ------------------------------------------------------------------
    # 输入校验
    # ------------------------------------------------------------------
    def _validate_channels(self, channels):
        if not _is_sequence(channels):
            raise ValueError("channels must be a sequence of channel arrays")
        if len(channels) == 0:
            return 0, 0
        if len(channels) != self._cols:
            raise ValueError(
                "expected %d input channels, got %d"
                % (self._cols, len(channels))
            )
        length = None
        for ch in channels:
            if not _is_sequence(ch):
                raise ValueError("each channel must be a sequence")
            if length is None:
                length = len(ch)
            elif len(ch) != length:
                raise ValueError("channels have inconsistent lengths")
        return len(channels), length

    # ------------------------------------------------------------------
    # 响度 / 削波
    # ------------------------------------------------------------------
    @staticmethod
    def _protection_gain(in_per_ch, out_per_ch, peak, downmix):
        """简化 BS.1770 下混响度补偿与峰值削波保护，取更保守者。

        - 峰值增益 1/peak：硬上限，任何矩阵超峰值时都生效；
        - 响度补偿 sqrt(in/out)：仅在下混（R < C）且每路平均功率
          因相干叠加而上升时启用，把下混后功率拉回下混前水平；
          非下混矩阵（纯增益、上混）不做响度处理，只受峰值约束。
        增益永不超过 1.0（只衰减不提升）；峰值低于 1.0 时整体不启用，
        低电平通路逐样本保持不变。
        """
        if peak < 1.0:
            return 1.0
        gain_peak = 1.0 / peak
        if downmix and in_per_ch > 0.0 and out_per_ch > in_per_ch:
            gain_loudness = math.sqrt(in_per_ch / out_per_ch)
            return min(1.0, gain_peak, gain_loudness)
        return min(1.0, gain_peak)

    @staticmethod
    def _apply_gain_clip(row, gain):
        """增益与硬削波合并为单遍处理。"""
        if gain == 1.0:
            return [
                1.0 if x > 1.0 else (-1.0 if x < -1.0 else x) for x in row
            ]
        out = []
        append = out.append
        for x in row:
            y = gain * x
            if y > 1.0:
                y = 1.0
            elif y < -1.0:
                y = -1.0
            append(y)
        return out

    # ------------------------------------------------------------------
    # 后端
    # ------------------------------------------------------------------
    def _apply_numpy(self, channels, frames):
        try:
            x = _np.asarray(channels, dtype=_np.float64, order="C")
        except (TypeError, ValueError):
            raise ValueError("channel samples must be real numbers")
        if x.ndim != 2 or x.shape[0] != self._cols:
            raise ValueError("invalid channel layout")
        if not _np.isfinite(x).all():
            raise ValueError("channel samples must be finite")
        y = self._np_matrix @ x  # 形状 (R, N)，每次调用都是新数组
        if frames == 0:
            return [[] for _ in range(self._rows)]
        peak = float(_np.max(_np.abs(y)))
        # 每路平均功率口径 = 总平方和 / 总样本数（mean 对整个 R×N 矩阵），
        # 与纯 Python 回退路径完全一致
        in_per_ch = float((x * x).mean())
        out_per_ch = float((y * y).mean())
        gain = self._protection_gain(
            in_per_ch, out_per_ch, peak, self._rows < self._cols
        )
        if gain != 1.0:
            y = y * gain
        _np.clip(y, -1.0, 1.0, out=y)
        return [row.tolist() for row in y]

    def _apply_pure(self, channels, frames):
        if self._pure_mixer is None:
            with self._lock:
                if self._pure_mixer is None:
                    self._pure_mixer = _build_pure_mixer(
                        self._matrix, self._rows, self._cols
                    )
        # 输入能量与非有限样本校验合并为单遍（每路平均功率口径，
        # 与 numpy 路径一致：总平方和 / 声道数 / 帧数）
        input_sq = 0.0
        for ch in channels:
            acc = 0.0
            for x in ch:
                if not isinstance(x, (int, float)):
                    raise ValueError("channel samples must be real numbers")
                if not math.isfinite(x):
                    raise ValueError("channel samples must be finite")
                acc += x * x
            input_sq += acc
        in_per_ch = input_sq / (self._cols * frames)
        mixed = self._pure_mixer(*channels)
        # 全局口径（与 numpy 路径一致）：跨所有输出行汇总能量与峰值，
        # 整段共用一个保护增益，保证各输出声道间的比例关系不被破坏
        output_sq = 0.0
        peak = 0.0
        for row in mixed:
            row_sq = 0.0
            for x in row:
                ax = -x if x < 0.0 else x
                if ax > peak:
                    peak = ax
                row_sq += x * x
            output_sq += row_sq
        out_per_ch = output_sq / (self._rows * frames)
        gain = self._protection_gain(
            in_per_ch, out_per_ch, peak, self._rows < self._cols
        )
        return [self._apply_gain_clip(row, gain) for row in mixed]

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def apply(self, channels):
        """对 C 路输入（每路等长样本序列）应用矩阵，返回 R 路 list[list[float]]。"""
        n_channels, frames = self._validate_channels(channels)
        if n_channels == 0:
            return []

        # 单位矩阵原样透传：拷贝输出、不做归一化，同时完成有限性校验
        if self._identity:
            result = []
            for ch in channels:
                copy = []
                for x in ch:
                    if not isinstance(x, (int, float)):
                        raise ValueError(
                            "channel samples must be real numbers"
                        )
                    if not math.isfinite(x):
                        raise ValueError("channel samples must be finite")
                    copy.append(x)
                result.append(copy)
            return result

        if frames == 0:
            return [[] for _ in range(self._rows)]

        if _np is not None:
            return self._apply_numpy(channels, frames)
        return self._apply_pure(channels, frames)
