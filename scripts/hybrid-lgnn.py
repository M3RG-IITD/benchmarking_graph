################################################
################## IMPORT ######################
################################################

import json
import re
import sys
import unicodedata
from datetime import datetime
from functools import partial, wraps

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, value_and_grad, vmap
from jax.experimental import optimizers
from jax_md import space
from shadow.font import set_font_size
from shadow.plot import *
from sklearn.metrics import r2_score

from psystems.npendulum import (PEF, edge_order, get_init, hconstraints,
                                pendulum_connections)
from psystems.nsprings import (chain, edge_order, get_connections,
                               get_fully_connected_senders_and_receivers,
                               get_fully_edge_order)

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

set_font_size(size=30)

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
# jax.config.update('jax_platform_name', 'gpu')


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode(
            'ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def main(useN1=4, useN2=3, dim=2, trainm=0, time_step=1, ifdrag=0, runs=10, rname=1, semilog=1, maxtraj=1, grid=0, mpass1=1, mpass2=1, plotonly=0):

    PSYS = f"2-4-hybrid"
    TAG = f"lgnn"
    seed = 42
    out_dir = f"../results"

    def _filename(name, tag=TAG, trained=None):
        lt = name.split(".")
        if len(lt) > 1:
            a = lt[:-1]
            b = lt[-1]
            name = slugify(".".join(a)) + "." + b
        if tag == "data":
            part = f"_{ifdrag}."
        else:
            part = f"_{ifdrag}_{trainm}."
        if trained is not None:
            psys = f"{trained}"
        else:
            psys = PSYS
        name = ".".join(name.split(".")[:-1]) + \
            part + name.split(".")[-1]
        rstring = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") if rname else "0"
        filename_prefix = f"{out_dir}/{psys}-{tag}/{rstring}/"
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
        def func(file, *args, tag=TAG, trained=None, **kwargs):
            return f(_filename(file, tag=tag, trained=trained), *args, **kwargs)
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

    ##########################################
    ##############  INIT  ####################
    ##########################################

    def get_RV():
        θ1 = np.pi/50*(np.random.rand()-0.5)
        θ2 = -θ1
        sin1 = np.sin(θ1)
        cos1 = np.cos(θ1)
        sin2 = np.sin(θ2)
        cos2 = np.cos(θ2)
        l1 = 1
        l2 = 1

        R = jnp.array([
            [l1*sin1, -l1*cos1],
            [l1*sin1+l2*sin2, -l1*cos1-l2*cos2],
            [-0.2-0.2*np.random.rand(), -2.5],
            [0.2+0.2*np.random.rand(),  -2.5],
        ])

        print(R)

        t4 = np.pi/20*(np.random.rand()-0.5)
        rot = jnp.array([
            [np.cos(t4), -np.sin(t4)],
            [np.sin(t4), np.cos(t4)]
        ])

        R = rot.dot(R.T).T

        V = 0*R
        return R, V

    R, V = get_RV()

    print("Saving init configs...")
    savefile(f"initial-configs.pkl", [(R, V)])

    N = R.shape[0]

    species = jnp.zeros(N, dtype=int)
    masses = jnp.ones(N)

    dt = 1.0e-5
    stride = 1000

    ################################################
    ################## SYSTEM ######################
    ################################################

    #################
    ###  Spring  ####
    #################

    #################
    #         |
    #         |
    #         |
    #         0
    #       . | .
    #     .   |   .
    #   .     |     .
    # 2       |       3
    #   .     |     .
    #     .   |   .
    #       . | .
    #         1
    #################

    N1 = R.shape[0]

    if grid:
        print("It's a grid?")
        a = int(np.sqrt(N1))
        senders1, receivers1 = get_connections(a, a)
        eorder1 = edge_order(len(senders1))
    else:
        print("It's a random?")
        # senders1, receivers1 = get_fully_connected_senders_and_receivers(N)
        print("Creating Chain")
        # _, _, senders1, receivers1 = chain(N1)
        senders1, receivers1 = jnp.array(
            [0, 2, 1, 3]+[2, 1, 3, 0], dtype=int), jnp.array([2, 1, 3, 0]+[0, 2, 1, 3], dtype=int)
        eorder1 = edge_order(len(senders1))

    if trainm:
        print("kinetic energy: learnable")

        def L_energy_fn(params, graph):
            g, V, T = cal_graph(params, graph, mpass=mpass1, eorder=eorder1,
                                useT=True, useonlyedge=True)
            return T - V

    else:
        print("kinetic energy: 0.5mv^2")

        kin_energy = partial(lnn._T, mass=masses)

        def L_energy_fn(params, graph):
            g, V, T = cal_graph(params, graph, mpass=mpass1, eorder=eorder1,
                                useT=True, useonlyedge=True)
            return - V

    # state_graph1 = jraph.GraphsTuple(nodes={
    #     "position": R,
    #     "velocity": V,
    #     "type": species,
    # },
    #     edges={},
    #     senders=senders1,
    #     receivers=receivers1,
    #     n_node=jnp.array([N1]),
    #     n_edge=jnp.array([senders1.shape[0]]),
    #     globals={})

    def energy_fn(species):
        state_graph1 = jraph.GraphsTuple(nodes={
            "position": R,
            "velocity": V,
            "type": species
        },
            edges={},
            senders=senders1,
            receivers=receivers1,
            n_node=jnp.array([R.shape[0]]),
            n_edge=jnp.array([senders1.shape[0]]),
            globals={})

        def apply(R, V, params):
            state_graph1.nodes.update(position=R)
            state_graph1.nodes.update(velocity=V)
            return L_energy_fn(params, state_graph1)
        return apply

    apply_fn = energy_fn(species)

    def Lmodel_spring(x, v, params): return apply_fn(x, v, params["L"])

    params_spring = loadfile(f"lgnn_trained_model.dil",
                             trained=f"{useN1}-Spring")[0]

    #######################################################

    #################
    ### Pendulum ####
    #################

    N2 = 2
    senders2, receivers2 = pendulum_connections(N2)
    eorder2 = edge_order(len(senders2))

    if trainm:
        print("kinetic energy: learnable")

        def L_energy_fn2(params, graph):
            g, V, T = cal_graph(params, graph, eorder=eorder2,
                                useT=True)
            return T - V

    else:
        print("kinetic energy: 0.5mv^2")

        kin_energy = partial(lnn._T, mass=masses)

        def L_energy_fn2(params, graph):
            g, V, T = cal_graph(params, graph, eorder=eorder2,
                                useT=True)
            return - V

    # state_graph2 = jraph.GraphsTuple(nodes={
    #     "position": R[:N2],
    #     "velocity": V[:N2],
    #     "type": species[:N2],
    # },
    #     edges={},
    #     senders=senders2,
    #     receivers=receivers2,
    #     n_node=jnp.array([N2]),
    #     n_edge=jnp.array([senders2.shape[0]]),
    #     globals={})

    def energy_fn2(species):
        state_graph2 = jraph.GraphsTuple(nodes={
            "position": R,
            "velocity": V,
            "type": species
        },
            edges={},
            senders=senders2,
            receivers=receivers2,
            n_node=jnp.array([R.shape[0]]),
            n_edge=jnp.array([senders2.shape[0]]),
            globals={})

        def apply(R, V, params):
            state_graph2.nodes.update(position=R)
            state_graph2.nodes.update(velocity=V)
            return L_energy_fn2(params, state_graph2)
        return apply

    apply_fn2 = energy_fn2(species)

    def Lmodel_pen(x, v, params): return apply_fn2(x, v, params["L"])

    params_pen = loadfile(f"lgnn_trained_model.dil",
                          trained=f"{useN2}-Pendulum")[0]

    def constraints(x, v, params):
        def func(x):
            X = x.reshape(-1, dim)
            X = X[:2]
            return hconstraints(X),
        out = jax.jacobian(func)(x)
        return out[0]

    def nndrag(v, params):
        return - jnp.abs(models.forward_pass(params, v.reshape(-1), activation_fn=models.SquarePlus)) * v

    if ifdrag == 0:
        print("Drag: 0.0")

        def drag(x, v, params):
            return 0.0
    elif ifdrag == 1:
        print("Drag: nn")

        def drag(x, v, params):
            return vmap(nndrag, in_axes=(0, None))(v.reshape(-1), params["drag"]).reshape(-1, 1)

    def external_force(x, v, params):
        # F = 0*R
        # F = jax.ops.index_update(F, (1, 0), 10.0)
        # return F.reshape(-1, 1)
        return 0.0

    def Lmodel(x, v, params): return (kin_energy(v) +
                                      Lmodel_pen(x, v, params["pen"]) +
                                      Lmodel_spring(x, v, params["spring"]) +
                                      0.0)

    acceleration_fn = lnn.accelerationFull(N1, dim,
                                           lagrangian=Lmodel,
                                           non_conservative_forces=None,
                                           constraints=constraints,
                                           external_force=external_force)

    params = {
        "spring": params_spring,
        "pen": params_pen,
    }

    # def lag1(x, v, params):
    #     return kin_energy(v) + Lmodel_spring(x, v, params)

    # def lag2(x, v, params):
    #     return kin_energy(v) + Lmodel_pen(x[:2], v[:2], params)

    # acceleration_fn_spring = lnn.accelerationFull(N1, dim,
    #                                               lagrangian=lag1,
    #                                               non_conservative_forces=None,
    #                                               constraints=constraints,
    #                                               external_force=None)

    # acceleration_fn_pen = lnn.accelerationFull(N2, dim,
    #                                            lagrangian=lag2,
    #                                            non_conservative_forces=None,
    #                                            constraints=constraints,
    #                                            external_force=None)

    # def acceleration_fn(R, V, params):
    #     A1 = acceleration_fn_spring(R, V, params["spring"])
    #     A2 = acceleration_fn_pen(R[:2], V[:2], params["pen"])
    #     A1 = jax.ops.index_update(A1, jax.ops.index[:2], A1[:2] + A2)
    #     return A1

    def force_fn(R, V, params, mass=None):
        if mass is None:
            return acceleration_fn(R, V, params)
        else:
            return acceleration_fn(R, V, params)*mass.reshape(-1, 1)

    @jit
    def forward_sim(R, V):
        return predition(R,  V, params, force_fn, shift, dt, masses, stride=stride, runs=runs)

    def PEactual(x):
        PE = 0
        PE += 1*10*x[:, 1].sum()
        dr = jnp.square(x[senders1] - x[receivers1]).sum(axis=1)
        PE += vmap(partial(lnn.SPRING, stiffness=1.0, length=1.0))(dr).sum()
        return PE

    def Lactual(x, v, params):
        KE = kin_energy(v)
        PE = PEactual(x)
        L = KE - PE
        return L

    acceleration_fn_orig = lnn.accelerationFull(N1, dim,
                                                lagrangian=Lactual,
                                                non_conservative_forces=None,
                                                constraints=constraints,
                                                external_force=external_force)

    def force_fn_orig(R, V, params, mass=None):
        if mass is None:
            return acceleration_fn_orig(R, V, params)
        else:
            return acceleration_fn_orig(R, V, params)*mass.reshape(-1, 1)

    @jit
    def forward_sim_orig(R, V):
        return predition(R,  V, None, force_fn_orig, shift, dt, masses, stride=stride, runs=runs)

    ################################################
    ###############   FORWARD SIM   ################
    ################################################
    def norm(a):
        a2 = jnp.square(a)
        n = len(a2)
        a3 = a2.reshape(n, -1)
        return jnp.sqrt(a3.sum(axis=1))

    def RelErr(ya, yp):
        return norm(ya-yp) / (norm(ya) + norm(yp))

    Zerr = []
    Herr = []

    for ii in range(maxtraj):
        if plotonly:
            pred_traj, _ = loadfile(f"model_states_pred_ii_{ii}.pkl")
            actual_traj, _ = loadfile(f"model_states_actual_ii_{ii}.pkl")
        else:
            print("FORWARD SIM ...")
            R, V = get_RV()
            actual_traj = forward_sim_orig(R, V)
            pred_traj = forward_sim(R, V)
            print("Saving datafile...")
            savefile(f"model_states_pred_ii_{ii}.pkl", pred_traj)
            savefile(f"model_states_actual_ii_{ii}.pkl", actual_traj)
            save_ovito(f"pred_ii_{ii}.ovito", [
                putpin(state) for state in NVEStates(pred_traj)], lattice="")
            save_ovito(f"actual_ii_{ii}.ovito", [
                putpin(state) for state in NVEStates(actual_traj)], lattice="")

        def cal_energy(states):
            KE = vmap(kin_energy)(states.velocity)
            L = vmap(Lmodel, in_axes=(0, 0, None))(
                states.position, states.velocity, params)
            L = (L - L[0]) + 1
            PE = - (L - KE)
            return jnp.array([PE, KE, L, PE+KE]).T

        def cal_energy_actual(states):
            KE = vmap(kin_energy)(states.velocity)
            PE = vmap(PEactual)(states.position)
            L = KE - PE
            L = L - L[0] + 1
            PE = - (L - KE)
            return jnp.array([PE, KE, L, PE+KE]).T

        print("plotting energy...")
        Es = cal_energy(actual_traj)
        Es_actual = cal_energy_actual(actual_traj)

        fig, axs = panel(1, 1, figsize=(20, 5))
        colors = ["b", "r", "g", "m"]
        labels = ["PE", "KE", "L", "TE"]
        for it1, it2, c, lab in zip(Es_actual.T, Es.T, colors, labels):
            plt.plot(it1, color=c, label=lab, lw=6, alpha=0.5)
            plt.plot(it2, "--", color=c, label=lab+"_pred", lw=2)
        plt.legend(bbox_to_anchor=(1, 1))
        plt.ylabel("Energy")
        plt.xlabel("Time step")

        title = f"2-4-hybrid actual"
        plt.title(title)
        plt.savefig(
            _filename(title.replace(" ", "_")+f"_ii_{ii}.png"), dpi=300)

        print("plotting energy... (pred)")
        Esh = cal_energy(pred_traj)
        Es_actualh = cal_energy_actual(pred_traj)

        fig, axs = panel(1, 1, figsize=(20, 5))
        colors = ["b", "r", "g", "m"]
        labels = ["PE", "KE", "L", "TE"]
        for it1, it2, c, lab in zip(Es_actualh.T, Esh.T, colors, labels):
            plt.plot(it1, color=c, label=lab, lw=6, alpha=0.5)
            plt.plot(it2, "--", color=c, label=lab+"_pred", lw=2)
        plt.legend(bbox_to_anchor=(1, 1))
        plt.ylabel("Energy")
        plt.xlabel("Time step")

        title = f"2-4-hybrid pred"
        plt.title(title)
        plt.savefig(
            _filename(title.replace(" ", "_")+f"_ii_{ii}.png"), dpi=300)

        H = Es_actual[:, -1]
        Hhat = Es_actualh[:, -1]

        Herr += [RelErr(H, Hhat)]
        Zerr += [RelErr(actual_traj.position,
                        pred_traj.position)]

    def make_plots(y, x=None, yl="Err", xl="Time", ax=None, time_step=1):

        print(f"Plotting err for {yl}")
        fig, axs = panel(1, 2)

        mean_ = jnp.log(jnp.array(y)).mean(axis=0)
        std_ = jnp.log(jnp.array(y)).std(axis=0)

        up_b = jnp.exp(mean_ + 2*std_)
        low_b = jnp.exp(mean_ - 2*std_)
        y = jnp.exp(mean_)

        if ax == None:
            pass
        else:
            plt.sca(ax)

        x = np.array(range(len(mean_)))*time_step
        if semilog:
            plt.semilogy(x, y)
        else:
            plt.plot(x, y)
        plt.fill_between(x, low_b, up_b, alpha=0.5)
        plt.ylabel(yl)
        plt.xlabel(xl)
        plt.savefig(_filename(f"RelError_{yl}_{xl}.png"))

    fig, axs = panel(1, 2, label=["", ""], brackets=False)

    make_plots(Zerr, yl="Rollout error", time_step=time_step, ax=axs[0])
    make_plots(
        Herr, yl="Energy violation", time_step=time_step, ax=axs[1])


fire.Fire(main)
