################################################
################## IMPORT ######################
################################################

import json
import sys
import os
from datetime import datetime
from functools import partial, wraps

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, value_and_grad, vmap
from jax.experimental import optimizers
from jax_md import space
#from shadow.plot import *
#from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from psystems.npendulum import PEF, get_init, hconstraints

MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8

import fire
import jraph
import src
from jax.config import config
from src import lnn
from src.graph import *
from src.lnn import acceleration, accelerationFull, accelerationTV
from src.md import *
from src.models import MSE, initialize_mlp
from src.nve import nve
from src.utils import *

import time

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
# jax.config.update('jax_platform_name', 'gpu')

def main(N=3, dim=2, saveat=100, nconfig=100, ifdrag=0, runs=100):
    tag = f"{N}-Pendulum-data"
    seed = 42
    out_dir = f"../noisy_data"
    rname = False
    rstring = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") if rname else "0_10000"
    filename_prefix = f"{out_dir}/{tag}/{rstring}/"

    def _filename(name):
        file = f"{filename_prefix}/{name}"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        filename = f"{filename_prefix}/{name}".replace("//", "/")
        print("===", filename, "===")
        return filename

    def displacement(a, b):
        return a - b

    def shift(R, dR, V):
        return R+dR, V

    def OUT(f):
        @wraps(f)
        def func(file, *args, **kwargs):
            return f(_filename(file), *args, **kwargs)
        return func

    loadmodel = OUT(src.models.loadmodel)
    savemodel = OUT(src.models.savemodel)

    loadfile = OUT(src.io.loadfile)
    savefile = OUT(src.io.savefile)
    save_ovito = OUT(src.io.save_ovito)

    ################################################
    ################## CONFIG ######################
    ################################################

    np.random.seed(seed)
    key = random.PRNGKey(seed)

    init_confs = [get_init(N, dim=dim) for i in range(nconfig)]

    print("Saving init configs...")
    savefile(f"initial-configs_{ifdrag}.pkl", init_confs)

    species = jnp.zeros(N, dtype=int)
    masses = jnp.ones(N)

    dt = 1.0e-5
    stride = 1000
    lr = 0.001

    ################################################
    ################## SYSTEM ######################
    ################################################

    pot_energy_orig = PEF
    kin_energy = partial(lnn._T, mass=masses)

    def Lactual(x, v, params):
        return kin_energy(v) - pot_energy_orig(x)

    def constraints(x, v, params):
        return jax.jacobian(lambda x: hconstraints(x.reshape(-1, dim)), 0)(x)

    def external_force(x, v, params):
        F = 0*R
        F = jax.ops.index_update(F, (1, 1), -1.0)
        return F.reshape(-1, 1)

    if ifdrag == 0:
        print("Drag: 0.0")

        def drag(x, v, params):
            return 0.0
    elif ifdrag == 1:
        print("Drag: -0.1*v")

        def drag(x, v, params):
            return -0.1*v.reshape(-1, 1)

    acceleration_fn_orig = lnn.accelerationFull(N, dim,
                                                lagrangian=Lactual,
                                                non_conservative_forces=drag,
                                                constraints=constraints,
                                                external_force=None)

    def force_fn_orig(R, V, params, mass=None):
        if mass is None:
            return acceleration_fn_orig(R, V, params)
        else:
            return acceleration_fn_orig(R, V, params)*mass.reshape(-1, 1)

    @jit
    def forward_sim(R, V):
        return predition(R,  V, None, force_fn_orig, shift, dt, masses, stride=stride, runs=runs)

    @jit
    def v_forward_sim(init_conf):
        return vmap(lambda x: forward_sim(x[0], x[1]))(init_conf)

    ################################################
    ############### DATA GENERATION ################
    ################################################

    print("Data generation ...")
    ind = 0
    dataset_states = []
    t = 0.0
    for R, V in init_confs:
        ind += 1
        print(f"{ind}/{len(init_confs)}", end='\r')
        start = time.time()
        model_states = forward_sim(R, V)
        end = time.time()
        t += end - start
        dataset_states += [model_states]
        
        if ind % saveat == 0:
            print(f"{ind} / {len(init_confs)}")
            print("Saving datafile...")
            savefile(f"model_states_{ifdrag}.pkl", dataset_states)

    print("Saving datafile...")
    savefile(f"model_states_{ifdrag}.pkl", dataset_states)

    def cal_energy(states):
        KE = vmap(kin_energy)(states.velocity)
        PE = vmap(pot_energy_orig)(states.position)
        L = vmap(Lactual, in_axes=(0, 0, None))(
            states.position, states.velocity, None)
        return jnp.array([PE, KE, L, KE+PE]).T

    print("plotting energy...")
    ind = 0
    for states in dataset_states:
        ind += 1
        Es = cal_energy(states)

        fig, axs = plt.subplots(1, 1, figsize=(20, 5))
        plt.plot(Es, label=["PE", "KE", "L", "TE"], lw=6, alpha=0.5)
        plt.legend(bbox_to_anchor=(1, 1))
        plt.ylabel("Energy")
        plt.xlabel("Time step")

        title = f"{N}-Pendulum random state {ind} {ifdrag}"
        plt.title(title)
        plt.savefig(_filename(title.replace(" ", "_")+".png"), dpi=300)
        if ind >= 10:
            break

    np.savetxt("../3-pendulum-simulation-time/simulation.txt", [t/nconfig], delimiter = "\n")

fire.Fire(main)




