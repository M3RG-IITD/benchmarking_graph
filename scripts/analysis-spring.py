################################################
################## IMPORT ######################
################################################

import json
import sys
import os
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
# from sklearn.metrics import r2_score
# from sympy import fu

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
from src.md import predition2
from src.models import MSE, initialize_mlp
from src.nve import NVEStates, nve
from src.utils import *
from src.hamiltonian import *

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
# jax.config.update('jax_platform_name', 'gpu')

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def pprint(*args, namespace=globals()):
    for arg in args:
        print(f"{namestr(arg, namespace)[0]}: {arg}")

def Full_Graph_analysis(plot_type, N=3, dt=1.0e-3, useN=5, withdata=None, datapoints=100, mpass=1, grid=False, stride=100, ifdrag=0, seed=42, rname=0, saveovito=1, trainm=1, runs=100, semilog=1, maxtraj=10, plotthings=False, redo=0):

    if useN is None:
        useN = N

    print("Configs: ")
    pprint(dt, stride, ifdrag,
           namespace=locals())

    PSYS = f"{N}-Spring"
    TAG = f"full-graph"
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
        # rstring = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") if rname else "0"
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

    # apply_fn = energy_fn(species)
    # v_apply_fn = vmap(apply_fn, in_axes=(None, 0))
    def L_change_fn(params, graph):
        g, change = cal_graph_modified(params, graph, mpass=mpass, eorder=eorder,
                                useT=True)
        return change

    def change_fn(species):
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
            return L_change_fn(params, state_graph)
        return apply

    apply_fn = change_fn(species)
    v_apply_fn = vmap(apply_fn, in_axes=(None, 0))

    def Lmodel(x, v, params): return apply_fn(x, v, params["L"])

    def change_R_V(N, dim):

        def fn(Rs, Vs, params):
            return Lmodel(Rs, Vs, params)
        return fn
    
    change_R_V_ = change_R_V(N, dim)

    v_change_R_V_ = vmap(change_R_V_, in_axes=(0, 0, None))

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

    params = loadfile(f"trained_model.dil", trained=useN)[0]

    # sim_model = get_forward_sim(
    #     params=params, force_fn=force_fn_model, runs=runs)

    def get_forward_sim_full_graph_network(params = None, run = runs):
        @jit
        def fn(R, V):
            return predition2(R,  V, params, change_R_V_, dt, masses, stride=stride, runs=run)
        return fn

    sim_model = get_forward_sim_full_graph_network(params=params, run=runs)

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
    # Es_pred_fn = cal_energy_fn(lag=Lmodel, params=params)

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

    for ind in range(maxtraj):
        # if ind > maxtraj+skip:
        #     break

        _ind = ind*runs

        print(f"Simulating trajectory {ind}/{maxtraj} ...")

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
                save_ovito(f"pred_{ind}.data", [
                    state for state in NVEStates(pred_traj)], lattice="")
                save_ovito(f"actual_{ind}.data", [
                    state for state in NVEStates(actual_traj)], lattice="")

            trajectories += [(actual_traj, pred_traj)]
            savefile("trajectories.pkl", trajectories)

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

            # fig, axs = panel(1, 2, figsize=(20, 5))
            # axs[0].plot(Es, label=["PE", "KE", "L", "TE"], lw=6, alpha=0.5)
            # axs[1].plot(Eshat, "--", label=["PE", "KE", "L", "TE"])
            # plt.legend(bbox_to_anchor=(1, 1), loc=2)
            # axs[0].set_facecolor("w")

            # xlabel("Time step", ax=axs[0])
            # xlabel("Time step", ax=axs[1])
            # ylabel("Energy", ax=axs[0])
            # ylabel("Energy", ax=axs[1])

            # title = f"Full Graph {N}-Spring Exp {ind} pred traj"
            # axs[1].set_title(title)
            # title = f"Full Graph {N}-Spring Exp {ind} actual traj"
            # axs[0].set_title(title)

            # plt.savefig(
            #     _filename(f"Full Graph {N}-Spring Exp {ind}".replace(" ", "-")+f"_actualH.png"))
        except:
            if skip < 20:
                skip += 1

    savefile(f"error_parameter.pkl", nexp)

    def make_plots(nexp, key, yl="Err"):
        # print(f"Plotting err for {key}")
        # fig, axs = panel(1, 1)

        # for i in range(len(nexp[key])):
        #     if semilog:
        #         plt.semilogy(nexp[key][i].flatten())
        #     else:
        #         plt.plot(nexp[key][i].flatten())

        # plt.ylabel(yl)
        # plt.xlabel("Time")
        # plt.savefig(_filename(f"RelError_{key}.png"))

        # fig, axs = panel(1, 1)
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
        # plt.ylabel(yl)
        # plt.xlabel("Time")
        # plt.savefig(_filename(f"RelError_std_{key}.png"))

    if plot_type == 0:
        make_plots(nexp, "Zerr", yl=r"$\frac{||\hat{z}-z||_2}{||\hat{z}||_2+||z||_2}$")

    if plot_type == 1:
        make_plots(nexp, "Herr", yl=r"$\frac{||H(\hat{z})-H(z)||_2}{||H(\hat{z})||_2+||H(z)||_2}$")

class Datastate:
    def __init__(self, model_states):
        self.position = model_states.position[:-1]
        self.velocity = model_states.velocity[:-1]
        self.force = model_states.force[:-1]
        self.mass = model_states.mass[:-1]
        self.index = 0
        self.change_position = model_states.position[1:]-model_states.position[:-1]
        self.change_velocity = model_states.velocity[1:]-model_states.velocity[:-1]

def Neural_ODE_analysis(plot_type, N=3, dt=1.0e-3, useN=5, withdata=None, datapoints=100, mpass=1, grid=False, stride=100, ifdrag=0, seed=42, rname=0, saveovito=1, trainm=1, runs=100, semilog=1, maxtraj=10, plotthings=False, redo=0):

    if useN is None:
        useN = N

    print("Configs: ")
    pprint(dt, stride, ifdrag,
           namespace=locals())

    PSYS = f"{N}-Spring"
    TAG = f"Neural-ODE"
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
        # rstring = randfilename if (rname and (tag != "data")) else (
        #     "1" if (tag == "data") or (withdata == None) else f"0_{withdata}")
        rstring  = "0" if (tag != "data" ) else "1"
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

    # def energy_fn(species):
    #     # senders, receivers = [np.array(i)
    #     #                       for i in get_fully_connected_senders_and_receivers(N)]
    #     state_graph = jraph.GraphsTuple(nodes={
    #         "position": R,
    #         "velocity": V,
    #         "type": species
    #     },
    #         edges={},
    #         senders=senders,
    #         receivers=receivers,
    #         n_node=jnp.array([R.shape[0]]),
    #         n_edge=jnp.array([senders.shape[0]]),
    #         globals={})

    #     def apply(R, V, params):
    #         state_graph.nodes.update(position=R)
    #         state_graph.nodes.update(velocity=V)
    #         return L_energy_fn(params, state_graph)
    #     return apply

    

    def L_change_fn(params, graph):
        g, change = cal_graph_modified(params, graph, eorder=eorder,
                                useT=True)
        return change

    def change_fn(species):
        #senders, receivers = [np.array(i)
        #                      for i in pendulum_connections(R.shape[0])]
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
            return L_change_fn(params, state_graph)
        return apply

    #apply_fn = energy_fn(species)
    apply_fn = change_fn(species)
    v_apply_fn = vmap(apply_fn, in_axes=(None, 0))

    def Lmodel(x, v, params): return apply_fn(x, v, params["L"])

    def change_Acc(N, dim):
        def fn(Rs, Vs, params):
            return Lmodel(Rs, Vs, params)
        return fn
    
    change_Acc_ = change_Acc(N, dim)

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

    params = loadfile(f"trained_model.dil", trained=useN)[0]

    # sim_model = get_forward_sim(
    #     params=params, force_fn=force_fn_model, runs=runs)

    def get_forward_sim_neural_ode(params = None, run = runs):
        @jit
        def fn(R, V):
            #return predition(R,  V, params, change_Acc_, shift, dt, masses, stride=stride, runs=runs)
            return predition3(R,  V, params, change_Acc_, dt, masses, stride=stride, runs=run)
        return fn

    sim_model = get_forward_sim_neural_ode(params=params, run=runs)

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

    for ind in range(maxtraj):
        # if ind > maxtraj+skip:
        #     break

        _ind = ind*runs

        print(f"Simulating trajectory {ind}/{maxtraj} ...")

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
                save_ovito(f"pred_{ind}.data", [
                    state for state in NVEStates(pred_traj)], lattice="")
                save_ovito(f"actual_{ind}.data", [
                    state for state in NVEStates(actual_traj)], lattice="")

            trajectories += [(actual_traj, pred_traj)]
            savefile("trajectories.pkl", trajectories)

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

            # fig, axs = panel(1, 2, figsize=(20, 5))
            # axs[0].plot(Es, label=["PE", "KE", "L", "TE"], lw=6, alpha=0.5)
            # axs[1].plot(Eshat, "--", label=["PE", "KE", "L", "TE"])
            # plt.legend(bbox_to_anchor=(1, 1), loc=2)
            # axs[0].set_facecolor("w")

            # xlabel("Time step", ax=axs[0])
            # xlabel("Time step", ax=axs[1])
            # ylabel("Energy", ax=axs[0])
            # ylabel("Energy", ax=axs[1])

            # title = f"Neural ODE {N}-Spring Exp {ind} pred traj"
            # axs[1].set_title(title)
            # title = f"Neural ODE {N}-Spring Exp {ind} actual traj"
            # axs[0].set_title(title)

            # plt.savefig(
            #     _filename(f"Neural ODE {N}-Spring Exp {ind}".replace(" ", "-")+f"_actualH.png"))
        except:
            if skip < 20:
                skip += 1

    savefile(f"error_parameter.pkl", nexp)

    def make_plots(nexp, key, yl="Err"):
        # print(f"Plotting err for {key}")
        # fig, axs = panel(1, 1)
        # for i in range(len(nexp[key])):
        #     if semilog:
        #         plt.semilogy(nexp[key][i].flatten())
        #     else:
        #         plt.plot(nexp[key][i].flatten())

        # plt.ylabel(yl)
        # plt.xlabel("Time")
        # plt.savefig(_filename(f"RelError_{key}.png"))

        # fig, axs = panel(1, 1)
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
        # plt.ylabel(yl)
        # plt.xlabel("Time")
        # plt.savefig(_filename(f"RelError_std_{key}.png"))

    if plot_type == 0:
        make_plots(nexp, "Zerr", yl=r"$\frac{||\hat{z}-z||_2}{||\hat{z}||_2+||z||_2}$")

    if plot_type == 1:
        make_plots(nexp, "Herr", yl=r"$\frac{||H(\hat{z})-H(z)||_2}{||H(\hat{z})||_2+||H(z)||_2}$")

def HGNN_analysis(plot_type, N=3, dim=2, dt=1.0e-3, stride=100, useN=5, withdata=None, datapoints=100, grid=False, ifdrag=0, seed=42, rname=0, saveovito=1, trainm=1, runs=100, semilog=1, maxtraj=10, plotthings=False, redo=0):
    print("Configs: ")
    pprint(dt, ifdrag, namespace=locals())

    PSYS = f"{N}-Spring"
    TAG = f"hgnn"
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
        # rstring = randfilename if (rname and (tag != "data")) else (
        #     "0" if (tag == "data") or (withdata == None) else f"0_{withdata}")
        rstring  = "0" if (tag != "data" ) else "2"
        filename_prefix = f"{out_dir}/{psys}-{tag}/{rstring}/"
        file = f"{filename_prefix}/{name}"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        filename = f"{filename_prefix}/{name}".replace("//", "/")
        print("===", filename, "===")
        return filename

    def OUT(f):
        @wraps(f)
        def func(file, *args, tag=TAG, trained=None, **kwargs):
            return f(_filename(file, tag=tag, trained=trained),
                     *args, **kwargs)
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

    dataset_states = loadfile(f"model_states.pkl", tag="data")[0]
    z_out, zdot_out = dataset_states[0]
    xout, pout = jnp.split(z_out, 2, axis=1)

    R = xout[0]
    V = pout[0]

    print(
        f"Total number of training data points: {len(dataset_states)}x{z_out.shape}")

    N, dim = xout.shape[-2:]
    species = jnp.zeros(N, dtype=int)
    masses = jnp.ones(N)

    ################################################
    ################## SYSTEM ######################
    ################################################
    
    def pot_energy_orig(x):
        dr = jnp.square(x[senders, :] - x[receivers, :]).sum(axis=1)
        return jax.vmap(partial(src.hamiltonian.SPRING, stiffness=1.0, length=1.0))(dr).sum()
    
    
    kin_energy = partial(src.hamiltonian._T, mass=masses)
    
    def Hactual(x, p, params):
        return kin_energy(p) + pot_energy_orig(x)
    
    # def phi(x):
    #     X = jnp.vstack([x[:1, :]*0, x])
    #     return jnp.square(X[:-1, :] - X[1:, :]).sum(axis=1) - 1.0
    
    # constraints = get_constraints(N, dim, phi)

    def external_force(x, v, params):
        F = 0*R
        F = jax.ops.index_update(F, (1, 1), -1.0)
        return F.reshape(-1, 1)

    if ifdrag == 0:
        print("Drag: 0.0")

        def drag(x, p, params):
            return 0.0
    elif ifdrag == 1:
        print("Drag: -0.1*v")

        def drag(x, p, params):
            return -0.1*p.reshape(-1, 1)

    zdot, lamda_force = get_zdot_lambda(
        N, dim, hamiltonian=Hactual, drag=drag, constraints=None)

    def zdot_func(z, t, params):
        x, p = jnp.split(z, 2)
        return zdot(x, p, params)

    def z0(x, p):
        return jnp.vstack([x, p])

    def get_forward_sim(params=None, zdot_func=None, runs=10):
        def fn(R, V):
            t = jnp.linspace(0.0, runs*stride*dt, runs*stride)
            _z_out = ode.odeint(zdot_func, z0(R, V), t, params)
            return _z_out[0::stride]
        return fn

    sim_orig = get_forward_sim(
        params=None, zdot_func=zdot_func, runs=maxtraj*runs)
    # z_out = sim_orig(R, V)

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

    def H_energy_fn(params, graph):
        g, g_PE, g_KE = cal_graph(params, graph, eorder=eorder,
                                  useT=True)
        return g_PE + g_KE

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
        #                       for i in Spring_connections(R.shape[0])]
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
            # jax.tree_util.tree_map(lambda a: print(a.shape), state_graph.nodes)
            return H_energy_fn(params, state_graph)
        return apply

    apply_fn = energy_fn(species)
    v_apply_fn = vmap(apply_fn, in_axes=(None, 0))

    def Hmodel(x, v, params):
        return apply_fn(x, v, params["H"])

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

    zdot_model, lamda_force_model = get_zdot_lambda(
        N, dim, hamiltonian=Hmodel, drag=drag, constraints=None)

    def zdot_model_func(z, t, params):
        x, p = jnp.split(z, 2)
        return zdot_model(x, p, params)

    params = loadfile(f"trained_model.dil", trained=useN)[0]

    sim_model = get_forward_sim(
        params=params, zdot_func=zdot_model_func, runs=runs)

    # z_model_out = sim_model(R, V)

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

    def caH_energy_fn(lag=None, params=None):
        def fn(states):
            KE = vmap(kin_energy)(states.velocity)
            H = vmap(lag, in_axes=(0, 0, None)
                     )(states.position, states.velocity, params)
            PE = (H - KE)
            # return jnp.array([H]).T
            return jnp.array([PE, KE, H, KE+PE]).T
        return fn

    Es_fn = caH_energy_fn(lag=Hactual, params=None)
    Es_pred_fn = caH_energy_fn(lag=Hmodel, params=params)
    # Es_pred_fn(pred_traj)

    def net_force_fn(force=None, params=None):
        def fn(states):
            zdot_out = vmap(force, in_axes=(0, 0, None))(
                states.position, states.velocity, params)
            _, force_out = jnp.split(zdot_out, 2, axis=1)
            return force_out
        return fn

    net_force_orig_fn = net_force_fn(force=zdot)
    net_force_model_fn = net_force_fn(force=zdot_model, params=params)

    nexp = {
        "z_pred": [],
        "z_actual": [],
        "Zerr": [],
        "Herr": [],
        "E": [],
    }

    trajectories = []

    sim_orig2 = get_forward_sim(params=None, zdot_func=zdot_func, runs=runs)

    for ind in range(maxtraj):
        # if ind > maxtraj:
        #     break
        print(f"Simulating trajectory {ind}/{maxtraj} ...")
        # R = full_traj[_ind].position
        # V = full_traj[_ind].velocity
        # start_ = _ind+1
        # stop_ = start_+runs

        z_out, _ = dataset_states[ind]
        xout, pout = jnp.split(z_out, 2, axis=1)

        R = xout[0]
        V = pout[0]

        z_actual_out = sim_orig2(R, V)  # full_traj[start_:stop_]
        x_act_out, p_act_out = jnp.split(z_actual_out, 2, axis=1)
        zdot_act_out = jax.vmap(zdot, in_axes=(0, 0, None))(
            x_act_out, p_act_out, None)
        _, force_act_out = jnp.split(zdot_act_out, 2, axis=1)
        my_state = States()
        my_state.position = x_act_out
        my_state.velocity = p_act_out
        my_state.force = force_act_out
        my_state.mass = jnp.ones(x_act_out.shape[0])
        actual_traj = my_state

        z_pred_out = sim_model(R, V)
        x_pred_out, p_pred_out = jnp.split(z_pred_out, 2, axis=1)
        zdot_pred_out = jax.vmap(zdot_model, in_axes=(
            0, 0, None))(x_pred_out, p_pred_out, params)
        _, force_pred_out = jnp.split(zdot_pred_out, 2, axis=1)
        my_state_pred = States()
        my_state_pred.position = x_pred_out
        my_state_pred.velocity = p_pred_out
        my_state_pred.force = force_pred_out
        my_state_pred.mass = jnp.ones(x_pred_out.shape[0])
        pred_traj = my_state_pred

        if saveovito:
            save_ovito(f"pred_{ind}.data", [
                state for state in NVEStates(pred_traj)], lattice="")
            save_ovito(f"actual_{ind}.data", [
                state for state in NVEStates(actual_traj)], lattice="")
        
        trajectories += [(actual_traj, pred_traj)]
        
        Es = Es_fn(actual_traj)
        Eshat = Es_fn(pred_traj)
        H = Es[:, -1]
        Hhat = Eshat[:, -1]
        
        nexp["Herr"] += [RelErr(H, Hhat)+1e-30]
        nexp["E"] += [Es, Eshat]
        
        nexp["z_pred"] += [pred_traj.position]
        nexp["z_actual"] += [actual_traj.position]
        nexp["Zerr"] += [RelErr(actual_traj.position,
                                pred_traj.position)+1e-30]
        
        savefile("trajectories.pkl", trajectories)
        savefile(f"error_parameter.pkl", nexp)

    def make_plots(nexp, key, yl="Err", xl="Time", key2=None):
        # print(f"Plotting err for {key}")
        # fig, axs = panel(1, 1)
        # filepart = f"{key}"
        # for i in range(len(nexp[key])):
        #     y = nexp[key][i].flatten()[2:]
        #     if key2 is None:
        #         x = range(len(y))
        #     else:
        #         x = nexp[key2][i].flatten()
        #         filepart = f"{filepart}_{key2}"
        #     if semilog:
        #         plt.semilogy(x, y)
        #     else:
        #         plt.plot(x, y)

        # plt.ylabel(yl)
        # plt.xlabel(xl)

        # plt.savefig(_filename(f"RelError_{filepart}.png"))  

        # fig, axs = panel(1, 1)

        mean_ = jnp.log(jnp.array(nexp[key])).mean(axis=0)[2:]
        std_ = jnp.log(jnp.array(nexp[key])).std(axis=0)[2:]

        up_b = jnp.exp(mean_ + 2*std_)
        low_b = jnp.exp(mean_ - 2*std_)
        y = jnp.exp(mean_)

        x = range(len(mean_))
        if semilog:
            plt.semilogy(x, y)
        else:
            plt.plot(x, y)
        plt.fill_between(x, low_b, up_b, alpha=0.5)
        # plt.ylabel(yl)
        # plt.xlabel("Time")
        # plt.savefig(_filename(f"RelError_std_{key}.png"))  # , dpi=500)

    if plot_type == 0:
        make_plots(nexp, "Zerr", yl=r"$\frac{||\hat{z}-z||_2}{||\hat{z}||_2+||z||_2}$")

    if plot_type == 1:
        make_plots(nexp, "Herr", yl=r"$\frac{||H(\hat{z})-H(z)||_2}{||H(\hat{z})||_2+||H(z)||_2}$")

def LGNN_analysis(plot_type, N=3, dt=1.0e-3, useN=5, withdata=None, datapoints=100, mpass=1, grid=False, stride=100, ifdrag=0, seed=42, rname=0, saveovito=1, trainm=1, runs=100, semilog=1, maxtraj=10, plotthings=False, redo=0):

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

    params = loadfile(f"trained_model.dil", trained=useN)[0]

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

    for ind in range(maxtraj):
        # if ind > maxtraj+skip:
        #     break

        _ind = ind*runs

        print(f"Simulating trajectory {ind}/{maxtraj} ...")

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
                save_ovito(f"pred_{ind}.data", [
                    state for state in NVEStates(pred_traj)], lattice="")
                save_ovito(f"actual_{ind}.data", [
                    state for state in NVEStates(actual_traj)], lattice="")

            trajectories += [(actual_traj, pred_traj)]
            savefile("trajectories.pkl", trajectories)

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

            # fig, axs = panel(1, 2, figsize=(20, 5))
            # axs[0].plot(Es, label=["PE", "KE", "L", "TE"], lw=6, alpha=0.5)
            # axs[1].plot(Eshat, "--", label=["PE", "KE", "L", "TE"])
            # plt.legend(bbox_to_anchor=(1, 1), loc=2)
            # axs[0].set_facecolor("w")

            # xlabel("Time step", ax=axs[0])
            # xlabel("Time step", ax=axs[1])
            # ylabel("Energy", ax=axs[0])
            # ylabel("Energy", ax=axs[1])

            # title = f"LGNN {N}-Spring Exp {ind} pred traj"
            # axs[1].set_title(title)
            # title = f"LGNN {N}-Spring Exp {ind} actual traj"
            # axs[0].set_title(title)

            # plt.savefig(
            #     _filename(f"LGNN {N}-Spring Exp {ind}".replace(" ", "-")+f"_actualH.png"))
        except:
            if skip < 20:
                skip += 1

    savefile(f"error_parameter.pkl", nexp)

    def make_plots(nexp, key, yl="Err"):
        # print(f"Plotting err for {key}")
        # fig, axs = panel(1, 1)
        # for i in range(len(nexp[key])):
        #     if semilog:
        #         plt.semilogy(nexp[key][i].flatten())
        #     else:
        #         plt.plot(nexp[key][i].flatten())

        # plt.ylabel(yl)
        # plt.xlabel("Time")
        # plt.savefig(_filename(f"RelError_{key}.png"))

        # fig, axs = panel(1, 1)
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
        # plt.ylabel(yl)
        # plt.xlabel("Time")
        # plt.savefig(_filename(f"RelError_std_{key}.png"))

    if plot_type == 0:
        make_plots(nexp, "Zerr", yl=r"$\frac{||\hat{z}-z||_2}{||\hat{z}||_2+||z||_2}$")

    if plot_type == 1:
        make_plots(nexp, "Herr", yl=r"$\frac{||H(\hat{z})-H(z)||_2}{||H(\hat{z})||_2+||H(z)||_2}$")

def main():
    plt.clf()
    fig, axs = plt.subplots(1, 1)

    Full_Graph_analysis(0)
    LGNN_analysis(0)
    Neural_ODE_analysis(0)
    HGNN_analysis(0)

    plt.ylabel("Zerr")
    plt.xlabel("Time Step")
    plt.legend(["DDGNN", "LGNN", "GNODE", "HGNN"])
    plt.savefig("../results/Analysis-Spring/Zerr.png")
    plt.close()

    plt.clf()
    fig, axs = plt.subplots(1, 1)

    Full_Graph_analysis(1)
    LGNN_analysis(1)
    Neural_ODE_analysis(1)
    HGNN_analysis(1)

    plt.ylabel("Herr")
    plt.xlabel("Time Step")
    plt.legend(["DDGNN", "LGNN", "GNODE", "HGNN"])
    plt.savefig("../results/Analysis-Spring/Herr.png")
    plt.close()

fire.Fire(main)



