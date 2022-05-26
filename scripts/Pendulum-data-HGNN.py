################################################
################## IMPORT ######################
################################################

import sys
import fire
import os
from datetime import datetime
from functools import partial, wraps
from psystems.npendulum import get_init

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental import ode
# from shadow.plot import panel
import matplotlib.pyplot as plt

MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8

import src
from jax.config import config
from src import lnn
from src.graph import *
from src.md import *
from src.models import MSE, initialize_mlp
from src.nve import nve
from src.utils import *
from src.hamiltonian import *


config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)


def ps(*args):
    for i in args:
        print(i.shape)


# N = 1
# dim = 2
# nconfig = 100
# saveat = 10
# ifdrag = 0
# dt = 0.01
# runs = 1000


def main(N=2, dim=2, nconfig=1000, saveat=10, ifdrag=0, dt=0.01, runs=100):

    tag = f"{N}-Pendulum-data"
    seed = 42
    out_dir = f"../data-efficiency"
    rname = False
    rstring = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") if rname else "2"
    filename_prefix = f"{out_dir}/{tag}/{rstring}/"

    def _filename(name):
        file = f"{filename_prefix}/{name}"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        filename = f"{filename_prefix}/{name}".replace("//", "/")
        print("===", filename, "===")
        return filename

    def OUT(f):
        @wraps(f)
        def func(file, *args, **kwargs):
            return f(_filename(file), *args, **kwargs)
        return func

    ################################################
    ################## CONFIG ######################
    ################################################

    np.random.seed(seed)
    key = random.PRNGKey(seed)

    init_confs = [get_init(N, dim=dim) for i in range(nconfig)]

    print("Saving init configs...")
    savefile = OUT(src.io.savefile)
    savefile(f"initial-configs_{ifdrag}.pkl", init_confs)

    ################################################
    ################## SYSTEM ######################
    ################################################

    def drag(x, p, params):
        return -0.1 * (p*p).sum()

    def KE(p):
        return (p*p).sum()/2

    def V(x, params):
        return 10*x[:, 1].sum()

    def hamiltonian(x, p, params): return KE(p) + V(x, params)

    def phi(x):
        X = jnp.vstack([x[:1, :]*0, x])
        return jnp.square(X[:-1, :] - X[1:, :]).sum(axis=1) - 1.0

    constraints = get_constraints(N, dim, phi)

    zdot, lamda_force = get_zdot_lambda(
        N, dim, hamiltonian, drag=None, constraints=constraints)

    def zdot_func(z, t):
        x, p = jnp.split(z, 2)
        return zdot(x, p, None)

    def get_z(x, p):
        return jnp.vstack([x, p])

    ################################################
    ############### DATA GENERATION ################
    ################################################

    def zz(out, ind=None):
        if ind is None:
            x, p = jnp.split(out, 2, axis=1)
            return x, p
        else:
            return jnp.split(out, 2, axis=1)[ind]

    t = jnp.linspace(0.0, runs*dt, runs)

    print("Data generation ...")
    ind = 0
    dataset_states = []
    for x, p in init_confs:
        z_out = ode.odeint(zdot_func, get_z(x, p), t)
        xout, pout = zz(z_out)
        zdot_out = jax.vmap(zdot, in_axes=(0, 0, None))(xout, pout, None)
        ind += 1
        print(f"{ind}/{len(init_confs)}", end='\r')
        # my_state = States()
        # my_state.position = xout
        # my_state.velocity = pout
        # my_state.force = zdot_out
        # my_state.mass = jnp.ones(xout.shape[0])
        # model_states = my_state
        model_states = z_out, zdot_out
        dataset_states += [model_states]
        if ind % saveat == 0:
            print(f"{ind} / {len(init_confs)}")
            print("Saving datafile...")
            savefile(f"model_states_{ifdrag}.pkl", dataset_states)

    print("Saving datafile...")
    savefile(f"model_states_{ifdrag}.pkl", dataset_states)

    print("plotting traj and Forces...")

    ind = 0
    for states in dataset_states:
        z_out, _ = states
        xout, pout = zz(z_out)
        # xout = states.position
        # pout = states.velocity
        ind += 1
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        for i in range(N):
            axs[0].scatter(xout[:, i, 0], xout[:, i, 1], c=t,
                           s=10*(i+1), label=f"pend: {i+1}")
        axs[0].set_xlabel("X-position")
        axs[0].set_ylabel("Y-position")
        axs[0].axis("square")

        force = jax.vmap(lamda_force, in_axes=(0, 0, None))(xout, pout, None)
        for i in range(N):
            axs[1].scatter(force[:, N+i, 0], force[:, N+i, 1], c=t,
                           s=10*(i+1), label=f"pend: {i+1}")
        axs[1].set_xlabel(r"F$_x$ (constraints)")
        axs[1].set_ylabel(r"F$_y$ (constraints)")
        axs[1].axis("square")

        title = f"{N}-Pendulum random state {ind} {ifdrag}"
        plt.suptitle(title, va="bottom")
        plt.savefig(_filename(title.replace(" ", "_")+".png"), dpi=300)

        if ind >= 10:
            break


fire.Fire(main)
