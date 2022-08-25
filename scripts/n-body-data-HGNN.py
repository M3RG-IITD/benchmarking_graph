################################################
################## IMPORT ######################
################################################

import sys
import fire
import os
from datetime import datetime
from functools import partial, wraps
from psystems.nsprings import chain

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
from src import hamiltonian
from src.graph import *
from src.md import *
from src.models import MSE, initialize_mlp
from src.nve import nve
from src.utils import *
from src.hamiltonian import *

from psystems.nbody import (get_fully_connected_senders_and_receivers, get_fully_edge_order, get_init_conf)

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

def ps(*args):
    for i in args:
        print(i.shape)

def main(N=4, dim=3, nconfig=1, saveat=100, ifdrag=0, dt=1e-3, stride = 100, runs=10000, train = False):

    tag = f"{N}-body-data"
    seed = 42
    out_dir = f"../results"
    rname = False
    rstring = "2" if train else "2_test"
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

    # init_confs = [chain(N)[:2] for i in range(nconfig)]

    # _, _, senders, receivers = chain(N)

    init_confs = get_init_conf(train)

    senders, receivers = get_fully_connected_senders_and_receivers(N)


    R, V = init_confs[0]

    print("Saving init configs...")
    savefile = OUT(src.io.savefile)
    savefile(f"initial-configs_{ifdrag}.pkl",
             init_confs, metadata={"N1": N, "N2": N})
    
    species = jnp.zeros(N, dtype=int)
    masses = jnp.ones(N)
    
    ################################################
    ################## SYSTEM ######################
    ################################################

    def drag(x, p, params):
        return -0.1 * (p*p).sum()
        
    def pot_energy_orig(x):
        dr = jnp.sqrt(jnp.square(x[senders, :] - x[receivers, :]).sum(axis=1))
        return vmap(partial(lnn.GRAVITATIONAL, Gc = 1))(dr).sum()/2

    kin_energy = partial(src.hamiltonian._T, mass=masses)

    def Hactual(x, p, params): return kin_energy(p) + pot_energy_orig(x)
    
    def external_force(x, p, params):
        F = 0*x
        F = jax.ops.index_update(F, (1, 1), -1.0)
        return F.reshape(-1, 1)
    
    if ifdrag == 0:
        print("Drag: 0.0")
        
        def drag(x, p, params):
            return 0.0
    elif ifdrag == 1:
        print("Drag: -0.1*p")
        
        def drag(x, p, params):
            return -0.1*p.reshape(-1, 1)
    
    zdot, lamda_force = get_zdot_lambda(
        N, dim, Hactual, drag=drag, constraints=None, external_force=None)

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

    t = jnp.linspace(0.0, runs*stride*dt, runs*stride)

    print("Data generation ...")
    ind = 0
    dataset_states = []
    for x, p in init_confs:
        _z_out = ode.odeint(zdot_func, get_z(x, p), t)
        z_out = _z_out[0::stride]
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

    print("plotting traj")

    ind = 0
    for states in dataset_states:
        z_out, _ = states
        xout, pout = zz(z_out)
        # xout = states.position
        # pout = states.velocity
        ind += 1
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        for i in range(N):
            axs[0].scatter(xout[:, i, 0], xout[:, i, 1], c=t[0::stride],
                           s=10*(i+1), label=f"pend: {i+1}")
        axs[0].set_xlabel("X-position")
        axs[0].set_ylabel("Y-position")
        axs[0].axis("square")

        force = jax.vmap(lamda_force, in_axes=(0, 0, None))(xout, pout, None)
        for i in range(N):
            axs[1].scatter(force[:, N+i, 0], force[:, N+i, 1], c=t[0::stride],
                           s=10*(i+1), label=f"pend: {i+1}")
        axs[1].set_xlabel(r"F$_x$ (constraints)")
        axs[1].set_ylabel(r"F$_y$ (constraints)")
        axs[1].axis("square")

        title = f"{N}-nbody random state {ind} {ifdrag}"
        plt.suptitle(title, va="bottom")
        plt.savefig(_filename(title.replace(" ", "_")+".png"), dpi=300)

        if ind > 3:
            break

fire.Fire(main)
