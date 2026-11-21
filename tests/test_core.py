import math

import pytest

import channelmix

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _m():
    return channelmix.ChannelMix([[1.0, 0.0], [0.0, 1.0]])


def _check_identity():
    return _m().apply([[0.5], [0.25]]) == [[0.5], [0.25]]


def _check_sum():
    m = channelmix.ChannelMix([[1.0, 1.0]])
    return m.apply([[0.25], [0.25]]) == [[0.5]]


def _check_gain():
    m = channelmix.ChannelMix([[2.0, 0.0]])
    return m.apply([[0.25], [0.0]]) == [[0.5]]


def _check_bad_dim():
    return _raises(lambda: channelmix.ChannelMix([[1.0, 0.0], [0.0]]))


def _check_bad_input():
    return _raises(lambda: channelmix.ChannelMix([[1.0]]).apply([[0.1], [0.2]]))


def _check_empty():
    return channelmix.ChannelMix([[1.0]]).apply([]) == []


def _check_len():
    return len(_m().apply([[0.1], [0.2]])) == 2


def _check_det():
    return _m().apply([[0.1], [0.2]]) == _m().apply([[0.1], [0.2]])


def _check_type():
    return isinstance(_m().apply([[0.1], [0.2]]), list)


def _check_range():
    m = channelmix.ChannelMix([[1.5, 0.0]])
    return len(m.apply([[0.9], [0.0]])) == 1


def _check_mono_downmix():
    m = channelmix.ChannelMix([[0.5, 0.5]])
    return abs(m.apply([[1.0], [1.0]])[0][0] - 1.0) < 1e-9


def _check_no_mutation():
    ch = [[0.5], [0.5]]
    _m().apply(ch)
    return ch == [[0.5], [0.5]]


def _check_two_frames():
    return len(_m().apply([[0.1, 0.2], [0.3, 0.4]])[0]) == 2


def _check_negative_gain():
    m = channelmix.ChannelMix([[-1.0, 0.0]])
    return m.apply([[0.5], [0.0]])[0][0] == -0.5


def _check_single():
    return channelmix.ChannelMix([[1.0]]).apply([[0.7]]) == [[0.7]]

CASES = [("identity", "_check_identity"), ("sum", "_check_sum"), ("gain", "_check_gain"), ("bad_dim", "_check_bad_dim"), ("bad_input", "_check_bad_input"), ("empty", "_check_empty"), ("len", "_check_len"), ("det", "_check_det"), ("type", "_check_type"), ("range", "_check_range"), ("mono_downmix", "_check_mono_downmix"), ("no_mutation", "_check_no_mutation"), ("two_frames", "_check_two_frames"), ("negative_gain", "_check_negative_gain"), ("single", "_check_single")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
