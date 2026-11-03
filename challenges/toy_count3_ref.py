"""Programa del reto toy-42 (count3): output 'S' si count de unos == 3, else 'N'.
Arquitectura toy_bench nueva (Q/K/V): N = z[0] canal via tok/V1, softmax uniforme.
"""
import torch


def set_weights(net, task):
    net.tok.data.fill_(0.0)
    net.tok.data[2, 0] = 1.0  # V1
    net.pos.data.fill_(0.0)
    net.K.data.fill_(0.0)
    net.Q.data.fill_(0.0)
    net.V.data.fill_(0.0)
    net.V.data[0, 0] = 1.0
    net.W1.data.fill_(0.0)
    net.b1.data.fill_(0.0)
    net.W1.data[0, 0] = 11.0
    net.b1.data[0] = -2.0
    net.W1.data[1, 0] = 11.0
    net.b1.data[1] = -3.0
    net.W1.data[2, 0] = 11.0
    net.b1.data[2] = -4.0
    net.W2.data.fill_(0.0)
    net.Wskip.data.fill_(0.0)
    net.W2.data[0, 0] = 1.0
    net.W2.data[0, 1] = -2.0
    net.W2.data[0, 2] = 1.0
    net.b2.data.fill_(0.0)
    net.Wout.data.fill_(0.0)
    net.Wout.data[4, 0] = 6.0   # S
    net.Wout.data[5, 0] = -6.0  # N
    net.bout.data.fill_(-13.0)
    net.bout.data[4] = -2.0
    net.bout.data[5] = 2.0
    return net
