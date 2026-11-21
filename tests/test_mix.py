"""channelmix 增强验收测试：边界、响度归一、削波、无爆音、并发、性能。

只依赖标准库 + 既有 channelmix 包，不修改既有 tests/test_core.py。
"""

import math
import threading
import time

import pytest

import channelmix
from channelmix import core


# ---------------------------------------------------------------------
# 单位矩阵 / 矩阵维度校验
# ---------------------------------------------------------------------
def test_mix_identity_passthrough_exact():
    m = channelmix.ChannelMix([[1.0, 0.0], [0.0, 1.0]])
    ch = [[0.125, -0.5, 0.75], [-0.25, 0.5, -0.125]]
    out = m.apply(ch)
    assert out == ch
    assert out is not ch and all(a is not b for a, b in zip(out, ch))


def test_mix_identity_3x3():
    m = channelmix.ChannelMix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ch = [[0.1, 0.2], [-0.3, 0.4], [0.5, -0.6]]
    assert m.apply(ch) == ch


def test_mix_ragged_matrix_rejected():
    with pytest.raises(ValueError):
        channelmix.ChannelMix([[1.0, 0.0], [0.0]])


def test_mix_empty_matrix_rejected():
    with pytest.raises(ValueError):
        channelmix.ChannelMix([])
    with pytest.raises(ValueError):
        channelmix.ChannelMix([[]])


def test_mix_non_numeric_matrix_rejected():
    with pytest.raises(ValueError):
        channelmix.ChannelMix([[1.0, "x"]])
    with pytest.raises(ValueError):
        channelmix.ChannelMix([[float("nan")]])
    with pytest.raises(ValueError):
        channelmix.ChannelMix([[float("inf")]])


def test_mix_matrix_dimension_limit():
    n = core.MAX_CHANNELS
    identity = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    assert len(channelmix.ChannelMix(identity).apply(
        [[0.1] for _ in range(n)])) == n
    too_big = n + 1
    with pytest.raises(ValueError):
        channelmix.ChannelMix(
            [[1.0 if i == j else 0.0 for j in range(too_big)]
             for i in range(too_big)]
        )
    wide = [[1.0] * too_big]
    with pytest.raises(ValueError):
        channelmix.ChannelMix(wide)


def test_mix_channel_count_boundary():
    m = channelmix.ChannelMix([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        m.apply([[0.1]])  # 太少
    with pytest.raises(ValueError):
        m.apply([[0.1], [0.2], [0.3]])  # 太多
    with pytest.raises(ValueError):
        m.apply("not-a-sequence")
    with pytest.raises(ValueError):
        m.apply([[0.1, 0.2], [0.3]])  # 各路长度不一致


def test_mix_non_finite_samples_rejected():
    m = channelmix.ChannelMix([[1.0]])
    with pytest.raises(ValueError):
        m.apply([[float("nan")]])
    with pytest.raises(ValueError):
        m.apply([[float("inf")]])
    with pytest.raises(ValueError):
        m.apply([[1.0, float("-inf")]])
    # 非有限样本在单位矩阵快路上同样要报错
    mi = channelmix.ChannelMix([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        mi.apply([[float("nan"), 0.1], [0.2, 0.3]])


def test_mix_empty_frames():
    assert channelmix.ChannelMix([[1.0, 1.0]]).apply([[], []]) == [[]]
    assert channelmix.ChannelMix([[1.0, 0.0], [0.0, 1.0]]).apply(
        [[], []]) == [[], []]


def test_mix_empty_channels():
    assert channelmix.ChannelMix([[1.0]]).apply([]) == []


# ---------------------------------------------------------------------
# 响度归一（简化 BS.1770）与削波保护
# ---------------------------------------------------------------------
def test_mix_low_level_path_unchanged():
    # 峰值不超过 1.0 时，归一化不应介入，结果等于裸矩阵乘法
    m = channelmix.ChannelMix([[0.5, 0.5], [0.25, -0.25]])
    ch = [[0.4, -0.3, 0.2], [0.1, 0.6, -0.4]]
    out = m.apply(ch)
    expected = [
        [0.5 * a + 0.5 * b for a, b in zip(*ch)],
        [0.25 * a - 0.25 * b for a, b in zip(*ch)],
    ]
    assert out == expected


def test_mix_clipping_protection_keeps_range():
    # 1x1 纯增益 1.5 倍作用在 1.0 上产生 1.5（超 1.0），峰值保护压回满刻度
    m = channelmix.ChannelMix([[1.5]])
    out = m.apply([[1.0, -1.0, 0.5]])
    assert len(out) == 1
    assert all(-1.0 <= x <= 1.0 for x in out[0])
    # 非下混只受峰值约束：1.5 -> 1.0，0.75 -> 0.5
    assert out[0][0] == pytest.approx(1.0, abs=1e-12)
    assert out[0][2] == pytest.approx(0.5, abs=1e-12)
    # 未超 1.0 的样本不触发保护，原样输出（与既有 _check_range 行为一致）
    low = channelmix.ChannelMix([[1.5]]).apply([[0.6]])
    assert low[0][0] == pytest.approx(0.9)
    # 2->1 下混超峰值时，响度归一（1/sqrt(2)）与峰值保护同时生效
    dm = channelmix.ChannelMix([[1.0, 0.0]])
    out_dm = dm.apply([[1.0], [0.0]])
    assert all(-1.0 <= x <= 1.0 for x in out_dm[0])
    assert out_dm[0][0] == pytest.approx(1.0 / math.sqrt(2.0), abs=1e-12)


def test_mix_clipping_never_exceeds_unity_on_loud_signal():
    m = channelmix.ChannelMix([[1.0, 1.0, 1.0], [0.7, -0.7, 0.3]])
    ch = [[0.9, -0.8, 0.7], [0.8, 0.9, -0.6], [-0.7, 0.6, 0.9]]
    out = m.apply(ch)
    for row in out:
        assert all(-1.0 <= x <= 1.0 for x in row)


def test_mix_loudness_normalization_preserves_energy_better_than_hard_clip():
    # 与纯硬削波相比，响度归一后的 RMS 应更大（保留更多能量），
    # 同时峰值仍受保护
    m = channelmix.ChannelMix([[1.0, 1.0]])
    ch = [[1.0] * 1000, [1.0] * 1000]
    out = m.apply(ch)[0]
    hard_clipped_rms = 1.0  # 硬削波会把 2.0 全部压到 1.0
    rms = math.sqrt(sum(x * x for x in out) / len(out))
    assert max(abs(x) for x in out) <= 1.0
    assert rms >= hard_clipped_rms - 1e-12


def test_mix_zero_signal_stays_zero():
    m = channelmix.ChannelMix([[1.0, 1.0]])
    out = m.apply([[0.0] * 16, [0.0] * 16])
    assert out == [[0.0] * 16]


# ---------------------------------------------------------------------
# 确定性 / 无爆音 / 不修改输入
# ---------------------------------------------------------------------
def test_mix_deterministic_repeated_runs():
    m = channelmix.ChannelMix([[0.7, 0.3], [0.4, -0.6]])
    ch = [[math.sin(i * 0.01) * 1.2 for i in range(2048)],
          [math.cos(i * 0.013) * 1.1 for i in range(2048)]]
    first = m.apply(ch)
    second = m.apply(ch)
    assert first == second
    third = channelmix.ChannelMix([[0.7, 0.3], [0.4, -0.6]]).apply(ch)
    assert first == third


def test_mix_no_pops_adjacent_sample_bound():
    # 削波场景下相邻样本差不得超过输入相邻差的合理上界：
    # 全局增益只可能缩小跳变，硬削波段为平台，爆音 = 出现突兀大跳变。
    m = channelmix.ChannelMix([[2.0, 1.0], [-1.0, 2.0]])
    ch = [[0.9 * math.sin(i * 0.02) for i in range(4096)],
          [0.9 * math.cos(i * 0.017) for i in range(4096)]]
    out = m.apply(ch)
    for row in out:
        diffs = [abs(row[i + 1] - row[i]) for i in range(len(row) - 1)]
        raw_bound = max(
            abs(ch[c][i + 1] - ch[c][i])
            for c in range(2) for i in range(4095)
        )
        # 输出跳变不超过输入最大跳变按矩阵行 L1 范数放大后的理论上界
        bound = 3.0 * raw_bound + 1e-12
        assert max(diffs) <= bound


def test_mix_does_not_mutate_input():
    m = channelmix.ChannelMix([[1.0, 1.0], [0.5, -0.5]])
    ch = [[0.6, -0.7], [0.8, 0.9]]
    snapshot = [list(c) for c in ch]
    m.apply(ch)
    assert ch == snapshot


# ---------------------------------------------------------------------
# 并发安全
# ---------------------------------------------------------------------
def test_mix_thread_safe_shared_instance():
    m = channelmix.ChannelMix([[0.7, 0.3], [0.4, -0.6]])
    ch = [[math.sin(i * 0.05) * 1.3 for i in range(8192)],
          [math.cos(i * 0.041) * 1.2 for i in range(8192)]]
    expected = m.apply(ch)
    results = []
    errors = []

    def worker():
        try:
            for _ in range(20):
                got = m.apply(ch)
                if got != expected:
                    errors.append("nondeterministic result under concurrency")
                    return
            results.append(True)
        except Exception as exc:  # pragma: no cover
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(results) == 8


# ---------------------------------------------------------------------
# 性能：10 万 / 100 万样本
# ---------------------------------------------------------------------
@pytest.mark.parametrize("samples", [100_000, 1_000_000])
def test_mix_performance_1m_under_300ms(samples):
    m = channelmix.ChannelMix([[0.707, 0.707], [0.707, -0.707]])
    ch = [[math.sin(i * 0.001) * 1.2 for i in range(samples)],
          [math.cos(i * 0.0013) * 1.2 for i in range(samples)]]
    start = time.perf_counter()
    out = m.apply(ch)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert len(out) == 2 and len(out[0]) == samples
    if samples == 1_000_000:
        assert elapsed_ms < 300.0, "1M samples took %.1f ms" % elapsed_ms


# ---------------------------------------------------------------------
# 双后端一致性（无 numpy 时自动跳过对比）
# ---------------------------------------------------------------------
def test_mix_backends_agree():
    np = core._np
    if np is None:
        pytest.skip("numpy not installed; pure backend only")
    matrix = [[0.8, 0.2, -0.1], [0.3, -0.5, 0.4]]
    m = channelmix.ChannelMix(matrix)
    ch = [[1.1 * math.sin(i * 0.03 + k) for i in range(4096)]
          for k in range(3)]
    fast = m._apply_numpy(ch, 4096)
    slow = m._apply_pure(ch, 4096)
    assert len(fast) == len(slow)
    for a, b in zip(fast, slow):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            # 两后端求和顺序不同，按相对误差 1e-9 口径比较
            assert x == pytest.approx(y, rel=1e-9, abs=1e-12)


def test_mix_pure_backend_standalone(monkeypatch):
    # 强制走纯标准库回退，验证无 numpy 环境下功能与保护仍成立
    monkeypatch.setattr(core, "_np", None)
    m = channelmix.ChannelMix([[1.0, 1.0], [0.5, -0.5]])
    ch = [[0.9, -0.8, 0.7], [0.8, 0.9, -0.6]]
    out = m.apply(ch)
    assert isinstance(out, list)
    for row in out:
        assert all(-1.0 <= x <= 1.0 for x in row)
    # 纯后端下确定性
    assert m.apply(ch) == out
    # 纯后端下非有限样本仍报错
    with pytest.raises(ValueError):
        channelmix.ChannelMix([[1.0]]).apply([[float("nan")]])
