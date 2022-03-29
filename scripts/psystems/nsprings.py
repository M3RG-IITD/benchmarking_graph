import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def get_fully_connected_senders_and_receivers(
    num_particles: int, self_edges: bool = False,
):
    """Returns senders and receivers for fully connected particles."""
    particle_indices = np.arange(num_particles)
    senders, receivers = np.meshgrid(particle_indices, particle_indices)
    senders, receivers = senders.flatten(), receivers.flatten()
    if not self_edges:
        mask = senders != receivers
        senders, receivers = senders[mask], receivers[mask]
    return senders, receivers


def get_fully_edge_order(N):
    out = []
    for j in range(N):
        for i in range(N):
            if i == j:
                pass
            else:
                if j > i:
                    out += [i*(N-1) + j-1]
                else:
                    out += [i*(N-1) + j]
    return np.array(out)


def get_connections(a, b):
    senders = []
    receivers = []

    def func(i, a, b):
        return [j*b+i for j in range(a)]

    for i in range(a):
        senders += list(range(i*b, (i + 1)*b - 1))
        receivers += list(range(i*b+1, (i + 1)*b))
    for i in range(b):
        senders += func(i, a, b)[:-1]
        receivers += func(i, a, b)[1:]
    return jnp.array(senders+receivers, dtype=int).flatten(), jnp.array(receivers+senders, dtype=int).flatten()


def edge_order(N):
    return jnp.array(list(range(N//2, N)) + list(range(N//2)), dtype=int)


def get_init(N, *args, **kwargs):
    a = int(np.sqrt(N) - 0.1) + 1
    b = int(N/a - 0.1) + 1
    return N, get_init_spring(a, b, *args, **kwargs)


def get_count(s, i):
    c = 0
    for item in s:
        if item == i:
            c += 1
    return c


def check(i, j, senders, receivers):
    bool = True
    for it1, it2 in zip(senders, receivers):
        if (it1 == i) and (it2 == j):
            bool = False
            break
        if (it2 == i) and (it1 == j):
            bool = False
            break
    return bool


def get_init_ab(a, b, L=1, dim=2):
    R = jnp.array(np.random.rand(a, dim))*L*2
    V = jnp.array(np.random.rand(*R.shape)) / 10
    V = V - V.mean(axis=0)

    senders = []
    receivers = []
    for i in range(a):
        c = get_count(senders, i)
        if c >= b:
            pass
        else:
            neigh = b-c
            s = ((R - R[i])**2).sum(axis=1)
            ind = np.argsort(s)
            new = []
            for j in ind:
                if check(i, j, senders, receivers) and (neigh > 0) and j != i:
                    new += [j]
                    neigh -= 1

            senders += new + [i]*len(new)
            receivers += [i]*len(new) + new
            print(i, senders, receivers)

    return R, V, jnp.array(senders, dtype=int), jnp.array(receivers, dtype=int)


def get_init_spring(a, b, L=1, dim=2, grid=True):
    if grid:
        def rand():
            return np.random.rand()
        Rs = []
        for i in range(a):
            for j in range(b):
                l = 0.2*L*(rand()-0.5)
                if dim == 2:
                    Rs += [[i*L-l, j*L-l]]
                else:
                    Rs += [[i*L-l, j*L-l, 0.0]]
        R = jnp.array(Rs, dtype=float)
        V = np.random.rand(*R.shape) / 10
        V = V - V.mean(axis=0)
        return R, V
    else:
        R = jnp.array(np.random.rand(a*b, dim))*L*2
        V = jnp.array(np.random.randn(*R.shape)) / 100
        V = V - V.mean(axis=0)
        return R, V


def plot_conf(R, senders, receivers, s=500, **kwargs):
    plt.scatter(R[:, 0], R[:, 1], s=s/np.sqrt(len(R)), **kwargs)
    Ri = R[senders]
    Rf = R[receivers]
    for a, b in zip(Ri, Rf):
        plt.plot([a[0], b[0]], [a[1], b[1]])
    plt.show()


def chain(N, L=2, dim=2):
    R = jnp.array(np.random.rand(N, dim))*L
    V = jnp.array(np.random.rand(*R.shape)) / 10
    V = V - V.mean(axis=0)

    senders = [N-1] + list(range(0, N-1))
    receivers = list(range(N))

    return R, V, jnp.array(senders+receivers, dtype=int), jnp.array(receivers+senders, dtype=int)
