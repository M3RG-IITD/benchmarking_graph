################################################
################## IMPORT ######################
################################################

import json
import sys
from datetime import datetime
from functools import partial, wraps

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, value_and_grad, vmap
from jax.experimental import optimizers
from jax_md import space
from shadow.plot import *
from sklearn.metrics import r2_score

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
from src.nve import NVEState, NVEStates, nve
from src.utils import *

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
# jax.config.update('jax_platform_name', 'gpu')


def main(N=2, dim=2, nconfig=10000, ifdrag=0, runs=10):

    tag = f"{N}-hybrid-data"
    seed = 42
    out_dir = f"../results"
    rname = False
    rstring = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") if rname else "0"
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

    def putpin(state):
        def f(x): return jnp.vstack([0*x[:1], x])
        return NVEState(f(state.position), f(state.velocity), f(state.force), f(state.mass))

    ################################################
    ################## CONFIG ######################
    ################################################

    np.random.seed(seed)
    key = random.PRNGKey(seed)

    # init_confs = [get_init(N, dim=dim) for i in range(nconfig)]

    R = jnp.array([
        [1.0, 0.0],
        [1.0, -1.0]
    ])
    V = 0*R

    init_confs = [(R, V)]

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

    parameters = [[dict(length=1.0, stiffness=1000.0)]]
    pot_energy_orig_spring = map_parameters(
        lnn.SPRING, displacement, species, parameters)

    pot_energy_orig_pen = PEF
    kin_energy = partial(lnn._T, mass=masses)

    def Lactual(x, v, params):
        return kin_energy(v) - pot_energy_orig_pen(x) - pot_energy_orig_spring(x)

    def constraints(x, v, params):
        def func(x):
            X = x.reshape(-1, dim)
            X = X[:1]
            return hconstraints(X),
        out = jax.jacobian(func)(x)
        return out[0]

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
    for R, V in init_confs:
        ind += 1
        print(f"{ind}/{len(init_confs)}", end='\r')
        model_states = forward_sim(R, V)
        dataset_states += [model_states]
        if ind % 1000 == 0:
            print(f"{ind} / {len(init_confs)}")
            print("Saving datafile...")
            savefile(f"model_states_{ifdrag}.pkl", dataset_states)

    print("Saving datafile...")
    savefile(f"model_states_{ifdrag}.pkl", dataset_states)

    def cal_energy(states):
        KE = vmap(kin_energy)(states.velocity)
        L = vmap(Lactual, in_axes=(0, 0, None))(
            states.position, states.velocity, None)
        PE = - (L-KE)
        return jnp.array([PE, KE, L, KE+PE]).T

    print("plotting energy...")
    ind = 0
    for states in dataset_states:
        ind += 1
        Es = cal_energy(states)

        fig, axs = panel(1, 1, figsize=(20, 5))
        plt.plot(Es, label=["PE", "KE", "L", "TE"], lw=6, alpha=0.5)
        plt.legend(bbox_to_anchor=(1, 1))
        plt.ylabel("Energy")
        plt.xlabel("Time step")

        title = f"{N}-hybrid random state {ind} {ifdrag}"
        plt.title(title)
        plt.savefig(
            _filename(title.replace(" ", "_")+".png"), dpi=300)

        save_ovito(f"actual_{ifdrag}.ovito", [putpin(
            state) for state in NVEStates(states)], lattice="")

        if ind > 10:
            break


fire.Fire(main)
