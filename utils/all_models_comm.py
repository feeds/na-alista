import utils.algorithms_comm as algo

"""
Different configurations of the algorithms, to be used in run.py
"""


# standard NA-ALISTA

def NA_ALISTA_UR_128(m, n, s, k, p, forward_op, backward_op, L):
    return algo.NA_ALISTA(m, n, k, forward_op, backward_op, s, p, lstm_input='c_b', lstm_hidden=128)


# all competitors

def AGLISTA(m, n, s, k, p, forward_op, backward_op, L):
    return algo.AGLISTA(m, n, k, forward_op, backward_op, s, p)


def ALISTA(m, n, s, k, p, forward_op, backward_op, L):
    return algo.ALISTA(m, n, k, forward_op, backward_op, s, p)


def ALISTA_AT(m, n, s, k, p, forward_op, backward_op, L):
    return algo.ALISTA_AT(m, n, k, forward_op, backward_op, s, p)


def ISTA(m, n, s, k, p, forward_op, backward_op, L):
    return algo.ISTA(m, n, k, forward_op, backward_op, L, lmbda=0.4)


def FISTA(m, n, s, k, p, forward_op, backward_op, L):
    return algo.FISTA(m, n, k, forward_op, backward_op, L, lmbda=0.4)


# NA-LISTA variations

def NA_ALISTA_U_128(m, n, s, k, p, forward_op, backward_op, L):
    return algo.NA_ALISTA(m, n, k, forward_op, backward_op, s, p, lstm_input='c', lstm_hidden=128)


def NA_ALISTA_R_128(m, n, s, k, p, forward_op, backward_op, L):
    return algo.NA_ALISTA(m, n, k, forward_op, backward_op, s, p, lstm_input='b', lstm_hidden=128)


def NA_ALISTA_UR_16(m, n, s, k, p, forward_op, backward_op, L):
    return algo.NA_ALISTA(m, n, k, forward_op, backward_op, s, p, lstm_input='c_b', lstm_hidden=16)


def NA_ALISTA_UR_32(m, n, s, k, p, forward_op, backward_op, L):
    return algo.NA_ALISTA(m, n, k, forward_op, backward_op, s, p, lstm_input='c_b', lstm_hidden=32)


def NA_ALISTA_UR_64(m, n, s, k, p, forward_op, backward_op, L):
    return algo.NA_ALISTA(m, n, k, forward_op, backward_op, s, p, lstm_input='c_b', lstm_hidden=64)


def NA_ALISTA_UR_256(m, n, s, k, p, forward_op, backward_op, L):
    return algo.NA_ALISTA(m, n, k, forward_op, backward_op, s, p, lstm_input='c_b', lstm_hidden=256)
