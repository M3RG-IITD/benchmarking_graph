import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

def get_fully_connected_senders_and_receivers(num_particles: int, self_edges: bool = False,):
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

def get_init_conf(train = True):
    R = [
        [1.0, 
        0.0,
        0.0,],
        [9.0,
        0.0,
        0.0,],
        [11.0,
        0.0,
        0.0,],
        [-1.0,
        0.0,
        0.0,],]

    V = [[0.0,
        0.05,
        0.0,],
        [0.0,
        -0.05,
        0.0,],
        [0.0,
        0.65,
        0.0,],
        [0.0,
        -0.65,
        0.0],]

    if (not train):
        return [(-jnp.array(R), jnp.array(V))]

    return [(jnp.array(R), jnp.array(V))]

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

def plot_conf(R, senders, receivers, s=500, **kwargs):
    plt.scatter(R[:, 0], R[:, 1], s=s/np.sqrt(len(R)), **kwargs)
    Ri = R[senders]
    Rf = R[receivers]
    for a, b in zip(Ri, Rf):
        plt.plot([a[0], b[0]], [a[1], b[1]])
    plt.show()



