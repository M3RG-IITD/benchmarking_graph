################################################
################## IMPORT ######################
################################################

import json
import sys
from datetime import datetime
from functools import partial, wraps
from statistics import mode

import fire
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, value_and_grad, vmap
from jax.experimental import optimizers
from jax_md import space
from pyexpat import model
from shadow.plot import *
from sklearn.metrics import r2_score
from sympy import fu

from psystems.nsprings import (chain, edge_order, get_connections,
                               get_fully_connected_senders_and_receivers,
                               get_fully_edge_order, get_init)

MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8

import jraph
import src
from jax.config import config
from src import lnn
from src.graph import *
from src.lnn import acceleration, accelerationFull, accelerationTV
from src.md import *
from src.models import MSE, initialize_mlp
from src.nve import NVEStates, nve
from src.utils import *

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
# jax.config.update('jax_platform_name', 'gpu')


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def pprint(*args, namespace=globals()):
    for arg in args:
        print(f"{namestr(arg, namespace)[0]}: {arg}")


def main(N=3, dt=1.0e-3, useN=None, withdata=None, datapoints=100, mpass=1, grid=False, stride=100, ifdrag=0, seed=42, rname=0, saveovito=0, trainm=0, runs=10, semilog=1, maxtraj=1, plotthings=False, redo=0):

    if useN is None:
        useN = N

    print("Configs: ")
    pprint(dt, stride, ifdrag,
           namespace=locals())

    PSYS = f"{N}-Spring"
    TAG = f"lgnn"
    out_dir = f"../results"

    randfilename = datetime.now().strftime(
        "%m-%d-%Y_%H-%M-%S") + f"_{datapoints}"

    def _filename(name, tag=TAG, trained=None):
        if tag == "data":
            part = f"_{ifdrag}."
        else:
            part = f"_{ifdrag}_{trainm}."
        if trained is not None:
            psys = f"{trained}-{PSYS.split('-')[1]}"
        else:
            psys = PSYS
        name = ".".join(name.split(".")[:-1]) + \
            part + name.split(".")[-1]
        rstring = randfilename if (rname and (tag != "data")) else (
            "0" if (tag == "data") or (withdata == None) else f"0_{withdata}")
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

    def _fileexist(f):
        if redo:
            return False
        else:
            return os.path.isfile(f)

    loadmodel = OUT(src.models.loadmodel)
    savemodel = OUT(src.models.savemodel)
    loadfile = OUT(src.io.loadfile)
    savefile = OUT(src.io.savefile)
    save_ovito = OUT(src.io.save_ovito)
    fileexist = OUT(_fileexist)

    ################################################
    ################## CONFIG ######################
    ################################################
    np.random.seed(seed)
    key = random.PRNGKey(seed)

    dataset_states = loadfile(f"model_states.pkl", tag="data")[0]
    model_states = dataset_states[0]

    if grid:
        a = int(np.sqrt(N))
        senders, receivers = get_connections(a, a)
        eorder = edge_order(len(senders))
    else:
        # senders, receivers = get_fully_connected_senders_and_receivers(N)
        # eorder = get_fully_edge_order(N)
        print("Creating Chain")
        _, _, senders, receivers = chain(N)
        eorder = edge_order(len(senders))

    R = model_states.position[0]
    V = model_states.velocity[0]

    print(
        f"Total number of training data points: {len(dataset_states)}x{model_states.position.shape[0]}")

    N, dim = model_states.position.shape[-2:]
    species = jnp.zeros(N, dtype=int)
    masses = jnp.ones(N)

    ################################################
    ################## SYSTEM ######################
    ################################################

    # parameters = [[dict(length=1.0)]]
    # pot_energy_orig = map_parameters(
    #     lnn.SPRING, displacement, species, parameters)

    def pot_energy_orig(x):
        dr = jnp.square(x[senders] - x[receivers]).sum(axis=1)
        return vmap(partial(lnn.SPRING, stiffness=1.0, length=1.0))(dr).sum()

    kin_energy = partial(lnn._T, mass=masses)

    def Lactual(x, v, params):
        return kin_energy(v) - pot_energy_orig(x)

    # def constraints(x, v, params):
    #     return jax.jacobian(lambda x: hconstraints(x.reshape(-1, dim)), 0)(x)

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
                                                constraints=None,
                                                external_force=None)

    def force_fn_orig(R, V, params, mass=None):
        if mass is None:
            return acceleration_fn_orig(R, V, params)
        else:
            return acceleration_fn_orig(R, V, params)*mass.reshape(-1, 1)

    def get_forward_sim(params=None, force_fn=None, runs=10):
        @jit
        def fn(R, V):
            return predition(R,  V, params, force_fn, shift, dt, masses, stride=stride, runs=runs)
        return fn

    sim_orig = get_forward_sim(
        params=None, force_fn=force_fn_orig, runs=maxtraj*runs+1)

    def simGT():
        print("Simulating ground truth ...")
        _traj = sim_orig(R, V)
        metadata = {"key": f"maxtraj={maxtraj}, runs={runs}"}
        savefile("gt_trajectories.pkl",
                 _traj, metadata=metadata)
        return _traj

    # if fileexist("gt_trajectories.pkl"):
    #     print("Loading from saved.")
    #     full_traj, metadata = loadfile("gt_trajectories.pkl")
    #     full_traj = NVEStates(full_traj)
    #     if metadata["key"] != f"maxtraj={maxtraj}, runs={runs}":
    #         print("Metadata doesnot match.")
    #         full_traj = NVEStates(simGT())
    # else:
    #     full_traj = NVEStates(simGT())

    ################################################
    ################### ML Model ###################
    ################################################

    # def L_energy_fn(params, graph):
    #     g, V, T = cal_graph(params, graph, eorder=eorder, useT=True)
    #     return T - V

    if trainm:
        print("kinetic energy: learnable")

        def L_energy_fn(params, graph):
            g, V, T = cal_graph(params, graph, mpass=mpass, eorder=eorder,
                                useT=True, useonlyedge=True)
            return T - V

    else:
        print("kinetic energy: 0.5mv^2")

        kin_energy = partial(lnn._T, mass=masses)

        def L_energy_fn(params, graph):
            g, V, T = cal_graph(params, graph, mpass=mpass, eorder=eorder,
                                useT=True, useonlyedge=True)
            return kin_energy(graph.nodes["velocity"]) - V

    state_graph = jraph.GraphsTuple(nodes={
        "position": R,
        "velocity": V,
        "type": species,
    },
        edges={},
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([N]),
        n_edge=jnp.array([senders.shape[0]]),
        globals={})

    def energy_fn(species):
        # senders, receivers = [np.array(i)
        #                       for i in get_fully_connected_senders_and_receivers(N)]
        state_graph = jraph.GraphsTuple(nodes={
            "position": R,
            "velocity": V,
            "type": species
        },
            edges={},
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([R.shape[0]]),
            n_edge=jnp.array([senders.shape[0]]),
            globals={})

        def apply(R, V, params):
            state_graph.nodes.update(position=R)
            state_graph.nodes.update(velocity=V)
            return L_energy_fn(params, state_graph)
        return apply

    apply_fn = energy_fn(species)
    v_apply_fn = vmap(apply_fn, in_axes=(None, 0))

    def Lmodel(x, v, params): return apply_fn(x, v, params["L"])

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

    acceleration_fn_model = accelerationFull(N, dim,
                                             lagrangian=Lmodel,
                                             constraints=None,
                                             non_conservative_forces=drag)

    def force_fn_model(R, V, params, mass=None):
        if mass is None:
            return acceleration_fn_model(R, V, params)
        else:
            return acceleration_fn_model(R, V, params)*mass.reshape(-1, 1)

    params = loadfile(f"lgnn_trained_model.dil", trained=useN)[0]

    sim_model = get_forward_sim(
        params=params, force_fn=force_fn_model, runs=runs)

    ################################################
    ############## forward simulation ##############
    ################################################

    def norm(a):
        a2 = jnp.square(a)
        n = len(a2)
        a3 = a2.reshape(n, -1)
        return jnp.sqrt(a3.sum(axis=1))

    def RelErr(ya, yp):
        return norm(ya-yp) / (norm(ya) + norm(yp))

    def Err(ya, yp):
        return ya-yp

    def AbsErr(*args):
        return jnp.abs(Err(*args))

    def cal_energy_fn(lag=None, params=None):
        @jit
        def fn(states):
            KE = vmap(kin_energy)(states.velocity)
            L = vmap(lag, in_axes=(0, 0, None)
                     )(states.position, states.velocity, params)
            PE = -(L - KE)
            return jnp.array([PE, KE, L, KE+PE]).T
        return fn

    Es_fn = cal_energy_fn(lag=Lactual, params=None)
    Es_pred_fn = cal_energy_fn(lag=Lmodel, params=params)

    def net_force_fn(force=None, params=None):
        @jit
        def fn(states):
            return vmap(force, in_axes=(0, 0, None))(states.position, states.velocity, params)
        return fn

    net_force_orig_fn = net_force_fn(force=force_fn_orig)
    net_force_model_fn = net_force_fn(
        force=force_fn_model, params=params)

    nexp = {
        "z_pred": [],
        "z_actual": [],
        "Zerr": [],
        "Herr": [],
        "E": [],
    }

    trajectories = []

    sim_orig2 = get_forward_sim(
        params=None, force_fn=force_fn_orig, runs=runs)

    skip = 0

    for ind in range(len(dataset_states)):
        if ind > maxtraj+skip:
            break

        _ind = ind*runs

        print(f"Simulating trajectory {ind}/{len(dataset_states)} ...")

        # R = full_traj[_ind].position
        # V = full_traj[_ind].velocity
        # start_ = _ind+1
        # stop_ = start_+runs

        R = dataset_states[ind].position[0]
        V = dataset_states[ind].velocity[0]

        try:
            actual_traj = sim_orig2(R, V)  # full_traj[start_:stop_]
            pred_traj = sim_model(R, V)

            if saveovito:
                save_ovito(f"pred_{ind}.ovito", [
                    state for state in NVEStates(pred_traj)], lattice="")
                save_ovito(f"actual_{ind}.ovito", [
                    state for state in NVEStates(actual_traj)], lattice="")

            trajectories += [(actual_traj, pred_traj)]
            savefile("trajectories.pkl", trajectories)

            if plotthings:
                for key, traj in {"actual": actual_traj, "pred": pred_traj}.items():

                    print(f"plotting energy ({key})...")

                    Es = Es_fn(traj)
                    Es_pred = Es_pred_fn(traj)

                    Es_pred = Es_pred - Es_pred[0] + Es[0]

                    fig, axs = panel(1, 2, figsize=(20, 5))
                    axs[0].plot(Es, label=["PE", "KE", "L", "TE"],
                                lw=6, alpha=0.5)
                    axs[1].plot(Es_pred, "--", label=["PE", "KE", "L", "TE"])
                    plt.legend(bbox_to_anchor=(1, 1), loc=2)
                    axs[0].set_facecolor("w")

                    xlabel("Time step", ax=axs[0])
                    xlabel("Time step", ax=axs[1])
                    ylabel("Energy", ax=axs[0])
                    ylabel("Energy", ax=axs[1])

                    title = f"LGNN {N}-Spring Exp {ind}"
                    plt.title(title)
                    plt.savefig(_filename(title.replace(
                        " ", "-")+f"_{key}_traj.png"))

                    net_force_orig = net_force_orig_fn(traj)
                    net_force_model = net_force_model_fn(traj)

                    fig, axs = panel(1+R.shape[0], 1, figsize=(20,
                                                               R.shape[0]*5), hshift=0.1, vs=0.35)
                    for i, ax in zip(range(R.shape[0]+1), axs):
                        if i == 0:
                            ax.text(0.6, 0.8, "Averaged over all particles",
                                    transform=ax.transAxes, color="k")
                            ax.plot(net_force_orig.sum(axis=1), lw=6, label=[
                                    r"$F_x$", r"$F_y$", r"$F_z$"][:R.shape[1]], alpha=0.5)
                            ax.plot(net_force_model.sum(
                                axis=1), "--", color="k")
                            ax.plot([], "--", c="k", label="Predicted")
                        else:
                            ax.text(0.6, 0.8, f"For particle {i}",
                                    transform=ax.transAxes, color="k")
                            ax.plot(net_force_orig[:, i-1, :], lw=6, label=[r"$F_x$",
                                    r"$F_y$", r"$F_z$"][:R.shape[1]], alpha=0.5)
                            ax.plot(
                                net_force_model[:, i-1, :], "--", color="k")
                            ax.plot([], "--", c="k", label="Predicted")

                        ax.legend(loc=2, bbox_to_anchor=(1, 1),
                                  labelcolor="markerfacecolor")
                        ax.set_ylabel("Net force")
                        ax.set_xlabel("Time step")
                        ax.set_title(f"{N}-Spring Exp {ind}")
                    plt.savefig(_filename(f"net_force_Exp_{ind}_{key}.png"))

            Es = Es_fn(actual_traj)
            H = Es[:, -1]
            L = Es[:, 2]

            Eshat = Es_fn(pred_traj)
            KEhat = Eshat[:, 1]
            Lhat = Eshat[:, 2]

            k = L[5]/Lhat[5]
            print(f"scalling factor: {k}")
            Lhat = Lhat*k
            Hhat = 2*KEhat - Lhat

            nexp["Herr"] += [RelErr(H, Hhat)]
            nexp["E"] += [Es, Eshat]

            nexp["z_pred"] += [pred_traj.position]
            nexp["z_actual"] += [actual_traj.position]
            nexp["Zerr"] += [RelErr(actual_traj.position,
                                    pred_traj.position)]

            fig, axs = panel(1, 2, figsize=(20, 5))
            axs[0].plot(Es, label=["PE", "KE", "L", "TE"], lw=6, alpha=0.5)
            axs[1].plot(Eshat, "--", label=["PE", "KE", "L", "TE"])
            plt.legend(bbox_to_anchor=(1, 1), loc=2)
            axs[0].set_facecolor("w")

            xlabel("Time step", ax=axs[0])
            xlabel("Time step", ax=axs[1])
            ylabel("Energy", ax=axs[0])
            ylabel("Energy", ax=axs[1])

            title = f"LGNN {N}-Spring Exp {ind} pred traj"
            axs[1].set_title(title)
            title = f"LGNN {N}-Spring Exp {ind} actual traj"
            axs[0].set_title(title)

            plt.savefig(
                _filename(f"LGNN {N}-Spring Exp {ind}".replace(" ", "-")+f"_actualH.png"))
        except:
            if skip < 20:
                skip += 1

    savefile(f"error_parameter.pkl", nexp)

    def make_plots(nexp, key, yl="Err"):
        print(f"Plotting err for {key}")
        fig, axs = panel(1, 1)
        for i in range(len(nexp[key])):
            if semilog:
                plt.semilogy(nexp[key][i].flatten())
            else:
                plt.plot(nexp[key][i].flatten())

        plt.ylabel(yl)
        plt.xlabel("Time")
        plt.savefig(_filename(f"RelError_{key}.png"))

        fig, axs = panel(1, 1)
        mean_ = jnp.log(jnp.array(nexp[key])).mean(axis=0)
        std_ = jnp.log(jnp.array(nexp[key])).std(axis=0)

        up_b = jnp.exp(mean_ + 2*std_)
        low_b = jnp.exp(mean_ - 2*std_)
        y = jnp.exp(mean_)

        x = range(len(mean_))
        if semilog:
            plt.semilogy(x, y)
        else:
            plt.plot(x, y)
        plt.fill_between(x, low_b, up_b, alpha=0.5)
        plt.ylabel(yl)
        plt.xlabel("Time")
        plt.savefig(_filename(f"RelError_std_{key}.png"))

    make_plots(nexp, "Zerr",
               yl=r"$\frac{||\hat{z}-z||_2}{||\hat{z}||_2+||z||_2}$")
    make_plots(nexp, "Herr",
               yl=r"$\frac{||H(\hat{z})-H(z)||_2}{||H(\hat{z})||_2+||H(z)||_2}$")


fire.Fire(main)
