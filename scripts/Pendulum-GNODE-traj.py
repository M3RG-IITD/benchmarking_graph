################################################
################## IMPORT ######################
################################################

import json
import sys
import os
from datetime import datetime
from functools import partial, wraps
from statistics import mode
import pickle

import fire
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, value_and_grad, vmap
from jax.experimental import optimizers
from jax_md import space
import matplotlib.pyplot as plt
# from shadow.plot import *
# from sklearn.metrics import r2_score
# from torch import batch_norm_gather_stats_with_counts

from psystems.npendulum import (PEF, edge_order, get_init, hconstraints,
                                pendulum_connections)

MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8

import jraph
import src
from jax.config import config
from src import fgn, lnn
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


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def pprint(*args, namespace=globals()):
    for arg in args:
        print(f"{namestr(arg, namespace)[0]}: {arg}")


def wrap_main(f):
    def fn(*args, **kwargs):
        config = (args, kwargs)
        print("Configs: ")
        print(f"Args: ")
        for i in args:
            print(i)
        print(f"KwArgs: ")
        for k, v in kwargs.items():
            print(k, ":", v)
        return f(*args, **kwargs, config=config)

    return fn


def Main(N=2, epochs=10000, seed=42, rname=False, error_fn="L2error", mpass=1, saveat=10,
         dt=1.0e-5, ifdrag=0, trainm=1, stride=1000, lr=0.001, datapoints=None, batch_size=1000):

    return wrap_main(main)(N=N, epochs=epochs, seed=seed, rname=rname, error_fn=error_fn, mpass=mpass,
                           dt=dt, ifdrag=ifdrag, trainm=trainm, stride=stride, lr=lr, datapoints=datapoints,
                           batch_size=batch_size, saveat=saveat)


def main(N=2, epochs=10000, seed=42, rname=False,  error_fn="L2error", mpass=1, saveat=10,
         dt=1.0e-5, ifdrag=0, trainm=1, stride=1000, lr=0.001,  withdata=None, datapoints=None, batch_size=1000, config=None, traj = 5):

    # print("Configs: ")
    # pprint(N, epochs, seed, rname,
    #        dt, stride, lr, ifdrag, batch_size,
    #        namespace=locals())

    randfilename = datetime.now().strftime(
        "%m-%d-%Y_%H-%M-%S") + f"_{datapoints}"

    PSYS = f"{N}-Pendulum"
    TAG = f"gnode-traj"
    out_dir = f"../results"

    def _filename(name, tag=TAG):
        rstring = randfilename if (rname and (tag != "data")) else (
            "0" if (tag == "data") or (withdata == None) else f"{withdata}")
        filename_prefix = f"{out_dir}/{PSYS}-{tag}/{rstring}/"
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
        def func(file, *args, tag=TAG, **kwargs):
            return f(_filename(file, tag=tag), *args, **kwargs)
        return func

    loadmodel = OUT(src.models.loadmodel)
    savemodel = OUT(src.models.savemodel)

    loadfile = OUT(src.io.loadfile)
    savefile = OUT(src.io.savefile)
    save_ovito = OUT(src.io.save_ovito)

    savefile(f"config_{ifdrag}_{trainm}.pkl", config)

    ################################################
    ################## CONFIG ######################
    ################################################
    np.random.seed(seed)
    key = random.PRNGKey(seed)

    try:
        dataset_states = loadfile(f"model_states_{ifdrag}.pkl", tag="data")[0]
    except:
        raise Exception("Generate dataset first. Use *-data.py file.")

    if datapoints is not None:
        dataset_states = dataset_states[:datapoints]

    model_states = dataset_states[0]

    print(
        f"Total number of data points: {len(dataset_states)}x{model_states.position.shape[0]}")

    N, dim = model_states.position.shape[-2:]
    species = jnp.zeros(N, dtype=int)
    masses = jnp.ones(N)

    Rs, Vs, Fs = States().fromlist(dataset_states).get_array()

    #print(Rs.shape)
    Rs = Rs.reshape(-1,1, N, dim)
    Vs = Vs.reshape(-1,1, N, dim)
    Fs = Fs.reshape(-1,1, N, dim)
    #print(Rs.shape)
    #print("test")
    Rsd = []
    Vsd = []
    Fsd = []

    for i in range(traj):
        Rsd += [Rs[i::traj]]
        Vsd += [Vs[i::traj]]
        Fsd += [Fs[i::traj]]


    #print(Rsd[0].shape)
    #print("concatenate")
    Rsg = jnp.concatenate(Rsd,axis = 1)
    Vsg = jnp.concatenate(Vsd,axis = 1)
    Fsg = jnp.concatenate(Fsd,axis = 1)
    #print(Rsg.shape)


    # Rsd = jnp.array(Rsd)
    # Vsd = jnp.array(Vsd)
    # Fsd = jnp.array(Fsd)


    mask = np.random.choice(len(Rsg), len(Rsg), replace=False)
    allRs = Rsg[mask]
    allVs = Vsg[mask]
    allFs = Fsg[mask]

    Ntr = int(0.75*len(Rsg))
    Nts = len(Rsg) - Ntr

    Rsg = allRs[:Ntr]
    Vsg = allVs[:Ntr]
    Fsg = allFs[:Ntr]

    Rsgt = allRs[Ntr:]
    Vsgt = allVs[Ntr:]
    Fsgt = allFs[Ntr:]

    ################################################
    ################## SYSTEM ######################
    ################################################

    # pot_energy_orig = PEF
    # kin_energy = partial(lnn._T, mass=masses)

    # def Lactual(x, v, params):
    #     return kin_energy(v) - pot_energy_orig(x)

    def constraints(x, v, params):
        return jax.jacobian(lambda x: hconstraints(x.reshape(-1, dim)), 0)(x)

    # def external_force(x, v, params):
    #     F = 0*R
    #     F = jax.ops.index_update(F, (1, 1), -1.0)
    #     return F.reshape(-1, 1)

    # def drag(x, v, params):
    #     return -0.1*v.reshape(-1, 1)

    # acceleration_fn_orig = lnn.accelerationFull(N, dim,
    #                                             lagrangian=Lactual,
    #                                             non_conservative_forces=None,
    #                                             constraints=constraints,
    #                                             external_force=None)

    # def force_fn_orig(R, V, params, mass=None):
    #     if mass is None:
    #         return acceleration_fn_orig(R, V, params)
    #     else:
    #         return acceleration_fn_orig(R, V, params)*mass.reshape(-1, 1)

    # @jit
    # def forward_sim(R, V):
    #     return predition(R,  V, None, force_fn_orig, shift, dt, masses, stride=stride, runs=10)

    ################################################
    ################### ML Model ###################
    ################################################

    senders, receivers = pendulum_connections(N)
    eorder = edge_order(N)

    hidden_dim = [16, 16]
    edgesize = 1
    nodesize = 5
    ee = 8
    ne = 8
    Lparams = dict(
        ee_params=initialize_mlp([edgesize, ee], key),
        ne_params=initialize_mlp([nodesize, ne], key),
        e_params=initialize_mlp([ee+2*ne, *hidden_dim, ee], key),
        n_params=initialize_mlp([2*ee+ne, *hidden_dim, ne], key),
        g_params=initialize_mlp([ne, *hidden_dim, 1], key),
        acc_params=initialize_mlp([ne, *hidden_dim, dim], key),
    )

    # if trainm:
    #     print("kinetic energy: learnable")

    #     def L_energy_fn(params, graph):
    #         g, V, T = cal_graph(params, graph, eorder=eorder,
    #                             useT=True)
    #         return T - V

    # else:
    #     print("kinetic energy: 0.5mv^2")

    #     kin_energy = partial(lnn._T, mass=masses)

    #     def L_energy_fn(params, graph):
    #         g, V, T = cal_graph(params, graph, eorder=eorder,
    #                             useT=True)
    #         return kin_energy(graph.nodes["velocity"]) - V

    R, V = Rsg[0][0], Vsg[0][0]

    species = jnp.array(species).reshape(-1, 1)

    def dist(*args):
        disp = displacement(*args)
        return jnp.sqrt(jnp.square(disp).sum())

    dij = vmap(dist, in_axes=(0, 0))(R[senders], R[receivers])

    state_graph = jraph.GraphsTuple(nodes={
        "position": R,
        "velocity": V,
        "type": species,
    },
        edges={"dij": dij},
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([N]),
        n_edge=jnp.array([senders.shape[0]]),
        globals={})

    def acceleration_fn(params, graph):
        acc = fgn.cal_acceleration(params, graph, mpass=1)
        return acc

    def acc_fn(species):
        senders, receivers = [np.array(i)
                              for i in pendulum_connections(R.shape[0])]
        state_graph = jraph.GraphsTuple(nodes={
            "position": R,
            "velocity": V,
            "type": species
        },
            edges={"dij": dij},
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([R.shape[0]]),
            n_edge=jnp.array([senders.shape[0]]),
            globals={})

        def apply(R, V, params):
            state_graph.nodes.update(position=R)
            state_graph.nodes.update(velocity=V)
            state_graph.edges.update(dij=vmap(dist, in_axes=(0, 0))(R[senders], R[receivers])
                                     )
            return acceleration_fn(params, state_graph)
        return apply

    apply_fn = jit(acc_fn(species))
    v_apply_fn = vmap(apply_fn, in_axes=(None, 0))

    def acceleration_fn_model(x, v, params): return apply_fn(x, v, params["L"])

    params = {"L": Lparams}

    print(acceleration_fn_model(R, V, params))

    # def nndrag(v, params):
    #     return - jnp.abs(models.forward_pass(params, v.reshape(-1), activation_fn=models.SquarePlus)) * v

    # if ifdrag == 0:
    #     print("Drag: 0.0")

    #     def drag(x, v, params):
    #         return 0.0
    # elif ifdrag == 1:
    #     print("Drag: -0.1*v")

    #     def drag(x, v, params):
    #         return vmap(nndrag, in_axes=(0, None))(v.reshape(-1), params["drag"]).reshape(-1, 1)

    # params["drag"] = initialize_mlp([1, 5, 5, 1], key)

    # acceleration_fn_model = accelerationFull(N, dim,
    #                                          lagrangian=Lmodel,
    #                                          constraints=constraints,
    #                                          non_conservative_forces=drag)

    v_acceleration_fn_model = vmap(acceleration_fn_model, in_axes=(0, 0, None))

    ################################################
    ########### trajectory #########################

    def force_fn_model(R, V, params, mass=None):
        if mass is None:
            return acceleration_fn_model(R, V, params)
        else:
            return acceleration_fn_model(R, V, params)*mass.reshape(-1, 1)

    def get_forward_sim(params=None, force_fn=None, runs=10):
        @jit
        def fn(R, V):
            return predition(R,  V, params, force_fn, shift, dt, masses, stride=stride, runs=runs)
        return fn

    def sim_model(x, v, params):
        fn = get_forward_sim(
            params=params, force_fn=force_fn_model, runs=traj)
        temp = fn(x,v)
        # print("inside sim_model")
        # print(type(temp))
        # temp1 = jnp.array(temp.position)
        # print(type(temp1))
        # print(temp1.shape)
        # #temp1 = States().fromlist(temp).get_array()
        # print("exited")
        return jnp.array(temp.position), jnp.array(temp.velocity)

    v_sim_model = vmap(sim_model, in_axes=(0, 0, None), out_axes=(0,0))


    ################################################
    ################## ML Training #################
    ################################################

    # LOSS = getattr(src.models, error_fn)

    @jit
    def loss_fn(params, Rs, Vs, Fs):
        # print(Rs.shape)
        # print(Rs[::,0].shape)
        pred_Rs, pred_Vs = v_sim_model(Rs[::,0], Vs[::,0],params)
        # print("inside loss")
        # print(pred_Rs[0][0])
        # print("----")
        # print(Rs[0][0])
        # print("exited")
        # print(type(pred_Rs))

        # #pred_Rs, pred_Vs, pred_Fs = States().fromlist(pred_traj).get_array()
        # print(pred_Rs.shape)


        # print(Rs.shape)
        #pred = v_acceleration_fn_model(Rs, Vs, params)
        return MSE(pred_Rs, Rs) + MSE(pred_Vs, Vs)

    @jit
    def gloss(*args):
        return value_and_grad(loss_fn)(*args)

    opt_init, opt_update_, get_params = optimizers.adam(lr)

    @jit
    def update(i, opt_state, params, loss__, *data):
        """ Compute the gradient for a batch and update the parameters """
        value, grads_ = gloss(params, *data)
        opt_state = opt_update(i, grads_, opt_state)
        return opt_state, get_params(opt_state), value

    @ jit
    def opt_update(i, grads_, opt_state):
        grads_ = jax.tree_map(jnp.nan_to_num, grads_)
        # grads_ = jax.tree_map(partial(jnp.clip, a_min=-1000.0, a_max=1000.0), grads_)
        return opt_update_(i, grads_, opt_state)

    @ jit
    def step(i, ps, *args):
        return update(i, *ps, *args)

    def batching(*args, size=None):
        L = len(args[0])
        if size != None:
            nbatches1 = int((L - 0.5) // size) + 1
            nbatches2 = max(1, nbatches1 - 1)
            size1 = int(L/nbatches1)
            size2 = int(L/nbatches2)
            if size1*nbatches1 > size2*nbatches2:
                size = size1
                nbatches = nbatches1
            else:
                size = size2
                nbatches = nbatches2
        else:
            nbatches = 1
            size = L

        newargs = []
        for arg in args:
            newargs += [jnp.array([arg[i*size:(i+1)*size]
                                   for i in range(nbatches)])]
        return newargs

    bRs, bVs, bFs = batching(Rsg, Vsg, Fsg,
                             size=min(len(Rsg), batch_size))

    print(f"training ...")

    params = pickle.load(open("../results/2-Pendulum-gnode/0/trained_model_0_1_low.dil", "rb"))[0]

    opt_state = opt_init(params)
    epoch = 0
    optimizer_step = -1
    larray = []
    ltarray = []
    last_loss = 1000

    larray += [loss_fn(params, Rsg, Vsg, Fsg)]
    ltarray += [loss_fn(params, Rsgt, Vsgt, Fsgt)]

    def print_loss():
        print(
            f"Epoch: {epoch}/{epochs}:  train={larray[-1]}, test={ltarray[-1]}")

    # print_loss()
    start = time.time()
    train_time_arr = []

    for epoch in range(epochs):
        for data in zip(bRs, bVs, bFs):
            optimizer_step += 1
            opt_state, params, l_ = step(
                optimizer_step, (opt_state, params, 0), *data)
            # break
        # break
        # opt_state, params, l = step(
        #     optimizer_step, (opt_state, params, 0), Rs, Vs, Fs)

        if epoch % 1 == 0:
            larray += [loss_fn(params, Rsg, Vsg, Fsg)]
            ltarray += [loss_fn(params, Rsgt, Vsgt, Fsgt)]
            print_loss()

        now = time.time()
        train_time_arr.append((now - start))

        if epoch % saveat == 0:
            metadata = {
                "savedat": epoch,
                "mpass": mpass,
                "ifdrag": ifdrag,
                "trainm": trainm,
            }
            savefile(f"trained_model_{ifdrag}_{trainm}.dil",
                     params, metadata=metadata)
            savefile(f"loss_array_{ifdrag}_{trainm}.dil",
                     (larray, ltarray), metadata=metadata)
            if last_loss > larray[-1]:
                last_loss = larray[-1]
                savefile(f"trained_model_{ifdrag}_{trainm}_low.dil",
                         params, metadata=metadata)

        

            plt.clf()
            fig, axs = plt.subplots(1, 1)
            plt.semilogy(larray, label="Training")
            plt.semilogy(ltarray, label="Test")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(_filename(f"training_loss_{ifdrag}_{trainm}.png"))

            np.savetxt("../training-time/gnode-traj-fgn.txt", train_time_arr, delimiter = "\n")
            np.savetxt("../training-loss/gnode-traj-fgn.txt", larray, delimiter = "\n")
            np.savetxt("../training-loss/gnode-traj-fgn.txt", ltarray, delimiter = "\n")
    #"""
    plt.clf()
    fig, axs = plt.subplots(1, 1)
    plt.semilogy(larray, label="Training")
    plt.semilogy(ltarray, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(_filename(f"training_loss_{ifdrag}_{trainm}.png"))

    metadata = {
        "savedat": epoch,
        "mpass": mpass,
        "ifdrag": ifdrag,
        "trainm": trainm,
    }
    params = get_params(opt_state)
    savefile(f"trained_model_{ifdrag}_{trainm}.dil",
             params, metadata=metadata)
    savefile(f"loss_array_{ifdrag}_{trainm}.dil",
             (larray, ltarray), metadata=metadata)

    np.savetxt("../training-time/gnode-traj-fgn.txt", train_time_arr, delimiter = "\n")
    np.savetxt("../training-loss/gnode-traj-fgn.txt", larray, delimiter = "\n")
    np.savetxt("../training-loss/gnode-traj-fgn.txt", ltarray, delimiter = "\n")

    #"""
fire.Fire(Main)
