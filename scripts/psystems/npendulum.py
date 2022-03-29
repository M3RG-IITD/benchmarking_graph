import jax.numpy as jnp
import numpy as np


def pendulum_connections(P):
    return (jnp.array([i for i in range(P-1)] + [i for i in range(1, P)], dtype=int),
            jnp.array([i for i in range(1, P)] + [i for i in range(P-1)], dtype=int))


def edge_order(P):
    N = (P-1)
    return jnp.array(jnp.hstack([jnp.array(range(N, 2*N)), jnp.array(range(N))]), dtype=int)


def get_θ(angles=(0, 360)):
    return np.random.choice(np.arange(*angles))


def get_L():
    # return np.random.choice(np.arange(1, 2, 0.1))
    return np.random.choice([1.0])


def get_init(P, *args, **kwargs):
    return get_init_pendulum(P, *args, **kwargs)


def get_init_pendulum(P, dim=2, **kwargs):
    Ls = [get_L() for i in range(P)]
    θs = [get_θ(**kwargs) for i in range(P)]
    last = [0.0, 0.0]
    pos = []
    for l, θ in zip(Ls, θs):
        last = [last[0] + l*np.sin(θ), last[1] - l*np.cos(θ)]
        if dim == 2:
            pos += [last]
        else:
            pos += [last+[0.0]]
    R = jnp.array(pos)
    return R, 0*R


def PEF(R, g=10.0, mass=jnp.array([1.0])):
    if len(mass) != len(R):
        mass = mass[0]*jnp.ones((len(R)))
    out = (mass*g*R[:, 1]).sum()
    return out


def hconstraints(R, l=jnp.array([1.0])):
    if len(l) != len(R):
        l = l[0]*jnp.ones((len(R)))
    out = jnp.square(R - jnp.vstack([0*R[:1], R[:-1]])).sum(axis=1) - l**2
    return out
