# Programa del reto toy-43 (geq3): output 'H' si count de unos >= 3, else 'N'


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
    net.b1.data[0] = -2.0  # A = relu(N-2)
    net.W1.data[1, 0] = 11.0
    net.b1.data[1] = -3.0  # B = relu(N-3)
    net.W2.data.fill_(0.0)
    net.Wskip.data.fill_(0.0)
    net.W2.data[0, 0] = 1.0
    net.W2.data[0, 1] = -1.0
    net.b2.data.fill_(0.0)
    net.Wout.data.fill_(0.0)
    net.Wout.data[6, 0] = 6.0   # H
    net.Wout.data[5, 0] = -6.0  # N
    net.bout.data.fill_(-13.0)
    net.bout.data[6] = -2.0
    net.bout.data[5] = 2.0
    return net
