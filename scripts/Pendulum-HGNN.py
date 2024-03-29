################################################
################## IMPORT ######################
################################################

from posixpath import split
import sys
import os
from datetime import datetime
from functools import partial, wraps

import fire
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental import ode
from shadow.plot import *
# from shadow.plot import panel
#import matplotlib.pyplot as plt

from psystems.npendulum import (PEF, edge_order, get_init, hconstraints,
                                pendulum_connections)


MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8

import jraph
import src
from jax.config import config
from src import lnn
from src.graph import *
from src.md import *
from src.models import MSE, initialize_mlp
from src.nve import nve
from src.utils import *
from src.hamiltonian import *

import time

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def pprint(*args, namespace=globals()):
    for arg in args:
        print(f"{namestr(arg, namespace)[0]}: {arg}")

def main(N = 3, epochs = 10000, seed = 42, rname = False,dt = 1.0e-5, ifdrag = 0, trainm = 1, stride=1000, lr = 0.001, withdata = None, datapoints = None, batch_size = 100, ifDataEfficiency = 0, if_lr_search = 0, if_act_search = 0, mpass = 1, if_mpass_search = 0, if_hidden_search = 0, hidden = 5, if_nhidden_search=0, nhidden=2, if_noisy_data = 1):    
    
    if (ifDataEfficiency == 1):
        data_points = int(sys.argv[1])
        batch_size = int(data_points/100)

    print("Configs: ")
    pprint(N, epochs, seed, rname, dt, lr, ifdrag, batch_size, namespace=locals())

    randfilename = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + f"_{datapoints}"

    PSYS = f"{N}-Pendulum"
    TAG = f"hgnn"

    if (ifDataEfficiency == 1):
        out_dir = f"../data-efficiency"
    elif (if_lr_search == 1):
        out_dir = f"../lr_search"
    elif (if_act_search == 1):
        out_dir = f"../act_search"
    elif (if_mpass_search == 1):
        out_dir = f"../mpass_search"
    elif (if_hidden_search == 1):
        out_dir = f"../mlp_hidden_search"
    elif (if_nhidden_search == 1):
        out_dir = f"../mlp_nhidden_search"
    elif (if_noisy_data == 1):
        out_dir = f"../noisy_data"
    else:
        out_dir = f"../results"

    def _filename(name, tag=TAG):
        # rstring = randfilename if (rname and (tag != "data")) else (
        #     "0" if (tag == "data") or (withdata == None) else f"0_{withdata}")
        rstring = "2" if (tag == "data") else "0"

        if (ifDataEfficiency == 1):
            rstring = "2_" + str(data_points)
        elif (if_lr_search == 1):
            rstring = "2_" + str(lr)
        elif (if_mpass_search == 1):
            rstring = "2_" + str(mpass)
        elif (if_act_search == 1):
            rstring = "2_softplus"
        elif (if_hidden_search == 1):
            rstring = "2_" + str(hidden)
        elif (if_nhidden_search == 1):
            rstring = "2_" + str(nhidden)

        if (tag == "data"):
            filename_prefix = f"../results/{PSYS}-{tag}/{2}/"
        else:
            filename_prefix = f"{out_dir}/{PSYS}-{tag}/{rstring}/"

        file = f"{filename_prefix}/{name}"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        filename = f"{filename_prefix}/{name}".replace("//", "/")
        print("===", filename, "===")
        return filename


    def OUT(f):
        @wraps(f)
        def func(file, *args, tag=TAG, **kwargs):
            return f(_filename(file, tag=tag), *args, **kwargs)
        return func


    loadfile = OUT(src.io.loadfile)
    savefile = OUT(src.io.savefile)


    ################################################
    ################## CONFIG ######################
    ################################################

    np.random.seed(seed)
    key = random.PRNGKey(seed)


    try:
        dataset_states = loadfile(f"model_states_{ifdrag}.pkl", tag="data")[0]
    except:
        raise Exception("Generate dataset first.")

    if datapoints is not None:
        dataset_states = dataset_states[:datapoints]


    model_states = dataset_states[0]
    z_out, zdot_out = model_states
    

    print(f"Total number of data points: {len(dataset_states)}x{z_out.shape[0]}")
    
    N2, dim = z_out.shape[-2:]
    N = N2//2
    species = jnp.zeros(N, dtype=int)
    masses = jnp.ones(N)

    array = jnp.array([jnp.array(i) for i in dataset_states])

    Zs = array[:, 0, :, :, :]
    Zs_dot = array[:, 1, :, :, :]

    Zs = Zs.reshape(-1, N2, dim)
    Zs_dot = Zs_dot.reshape(-1, N2, dim)

    if (if_noisy_data == 1):
        Zs = np.array(Zs)
        Zs_dot = np.array(Zs_dot)

        np.random.seed(100)
        for i in range(len(Zs)):
            Zs[i] += np.random.normal(0,1,1)
            Zs_dot[i] += np.random.normal(0,1,1)

        Zs = jnp.array(Zs)
        Zs_dot = jnp.array(Zs_dot)

    mask = np.random.choice(len(Zs), len(Zs), replace=False)
    allZs = Zs[mask]
    allZs_dot = Zs_dot[mask]

    Ntr = int(0.75*len(Zs))
    Nts = len(Zs) - Ntr

    Zs = allZs[:Ntr]
    Zs_dot = allZs_dot[:Ntr]

    Zst = allZs[Ntr:]
    Zst_dot = allZs_dot[Ntr:]

    ################################################
    ################## SYSTEM ######################
    ################################################


    def phi(x):
        X = jnp.vstack([x[:1, :]*0, x])
        return jnp.square(X[:-1, :] - X[1:, :]).sum(axis=1) - 1.0


    constraints = get_constraints(N, dim, phi)


    ################################################
    ################### ML Model ###################
    ################################################

    senders, receivers = pendulum_connections(N)
    eorder = edge_order(N)

    Ef = 1  # eij dim
    Nf = dim
    Oh = 1

    Eei = 5
    Nei = 5

    hidden = hidden
    nhidden = nhidden


    def get_layers(in_, out_):
        return [in_] + [hidden]*nhidden + [out_]


    def mlp(in_, out_, key, **kwargs):
        return initialize_mlp(get_layers(in_, out_), key, **kwargs)


    # # fne_params = mlp(Oh, Nei, key)
    fneke_params = initialize_mlp([Oh, Nei], key)
    fne_params = initialize_mlp([Oh, Nei], key)

    fb_params = mlp(Ef, Eei, key)
    fv_params = mlp(Nei+Eei, Nei, key)
    fe_params = mlp(Nei, Eei, key)

    ff1_params = mlp(Eei, 1, key)
    ff2_params = mlp(Nei, 1, key)
    ff3_params = mlp(dim+Nei, 1, key)
    ke_params = initialize_mlp([1+Nei, 10, 10, 1], key, affine=[True])

    Hparams = dict(fb=fb_params,
                fv=fv_params,
                fe=fe_params,
                ff1=ff1_params,
                ff2=ff2_params,
                ff3=ff3_params,
                fne=fne_params,
                fneke=fneke_params,
                ke=ke_params)

    def H_energy_fn(params, graph):
        if (if_act_search == 1):
            g, V, T = cal_graph(params, graph, eorder=eorder, useT=True, act_fn=models.SoftPlus)
        else:
            g, V, T = cal_graph(params, graph, eorder=eorder, useT=True, mpass=mpass)
        return T + V

    R, V = jnp.split(Zs[0], 2, axis=0)

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
        senders, receivers = [np.array(i)
                            for i in pendulum_connections(R.shape[0])]
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
            return H_energy_fn(params, state_graph)
        return apply

    apply_fn = energy_fn(species)
    v_apply_fn = vmap(apply_fn, in_axes=(None, 0))

    def Hmodel(x, v, params):
        return apply_fn(x, v, params["H"])

    params = {"H": Hparams}

    def nndrag(v, params):
        return - jnp.abs(models.forward_pass(params, v.reshape(-1), activation_fn=models.SquarePlus)) * v

    if ifdrag == 0:
        print("Drag: 0.0")

        def drag(x, p, params):
            return 0.0
    elif ifdrag == 1:
        print("Drag: -0.1*v")

        def drag(x, p, params):
            return vmap(nndrag, in_axes=(0, None))(p.reshape(-1), params["drag"]).reshape(-1, 1)

    params["drag"] = initialize_mlp([1, 5, 5, 1], key)

    zdot_model, lamda_force_model = get_zdot_lambda(
        N, dim, hamiltonian=Hmodel, drag=None, constraints=None,external_force=None)


    v_zdot_model = vmap(zdot_model, in_axes=(0, 0, None))

    ################################################
    ################## ML Training #################
    ################################################

    @jit
    def loss_fn(params, Rs, Vs, Zs_dot):
        pred = v_zdot_model(Rs, Vs, params)
        return MSE(pred, Zs_dot)

    def gloss(*args):
        return value_and_grad(loss_fn)(*args)

    def update(i, opt_state, params, loss__, *data):
        """ Compute the gradient for a batch and update the parameters """
        value, grads_ = gloss(params, *data)
        opt_state = opt_update(i, grads_, opt_state)
        return opt_state, get_params(opt_state), value

    @ jit
    def step(i, ps, *args):
        return update(i, *ps, *args)

    opt_init, opt_update_, get_params = optimizers.adam(lr)

    @ jit
    def opt_update(i, grads_, opt_state):
        grads_ = jax.tree_map(jnp.nan_to_num, grads_)
        # grads_ = jax.tree_map(partial(jnp.clip, a_min=-1000.0, a_max=1000.0), grads_)
        return opt_update_(i, grads_, opt_state)

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


    Rs, Vs = jnp.split(Zs, 2, axis=1)
    Rst, Vst = jnp.split(Zst, 2, axis=1)

    bRs, bVs, bZs_dot = batching(Rs, Vs, Zs_dot,
                                size=min(len(Rs), batch_size))

    print(f"training ...")

    opt_state = opt_init(params)
    epoch = 0
    optimizer_step = -1
    larray = []
    ltarray = []

    start = time.time()
    train_time_arr = []

    last_loss = 1000

    for epoch in range(epochs):
        l = 0.0
        for data in zip(bRs, bVs, bZs_dot):
            optimizer_step += 1
            opt_state, params, l_ = step(
                optimizer_step, (opt_state, params, 0), *data)
            l += l_
        l = l/len(bRs)
        if epoch % 1 == 0:
            # opt_state, params, l = step(
            #     optimizer_step, (opt_state, params, 0), Rs, Vs, Zs_dot)
            larray += [l]
            ltarray += [loss_fn(params, Rst, Vst, Zst_dot)]
            print(
                f"Epoch: {epoch}/{epochs} Loss (MSE):  train={larray[-1]}, test={ltarray[-1]}")
        if epoch % 10 == 0:
            # print(
            #     f"Epoch: {epoch}/{epochs} Loss (MSE):  train={larray[-1]}, test={ltarray[-1]}")
            savefile(f"trained_model_{ifdrag}_{trainm}.dil",
                    params, metadata={"savedat": epoch})
            savefile(f"loss_array_{ifdrag}_{trainm}.dil",
                    (larray, ltarray), metadata={"savedat": epoch})

            if last_loss > larray[-1]:
                last_loss = larray[-1]
                savefile(f"trained_model_low_{ifdrag}_{trainm}.dil",
                         params, metadata={"savedat": epoch})

        now = time.time()
        train_time_arr.append((now - start))

    fig, axs = panel(1, 1)
    plt.semilogy(larray, label="Training")
    plt.semilogy(ltarray, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(_filename(f"training_loss_{ifdrag}_{trainm}.png"))

    params = get_params(opt_state)
    savefile(f"trained_model_{ifdrag}_{trainm}.dil",
            params, metadata={"savedat": epoch})
    savefile(f"loss_array_{ifdrag}_{trainm}.dil",
            (larray, ltarray), metadata={"savedat": epoch})

    if (ifDataEfficiency == 0 and if_lr_search == 0 and if_act_search == 0 and if_mpass_search == 0 and if_hidden_search == 0 and if_nhidden_search == 0):
        np.savetxt(f"../{N}-pendulum-training-time/hgnn.txt", train_time_arr, delimiter = "\n")
        np.savetxt(f"../{N}-pendulum-training-loss/hgnn-train.txt", larray, delimiter = "\n")
        np.savetxt(f"../{N}-pendulum-training-loss/hgnn-test.txt", ltarray, delimiter = "\n")

    if (if_lr_search == 1):
        np.savetxt(f"../lr_search/{N}-pendulum-training-time/hgnn_{lr}.txt", train_time_arr, delimiter = "\n")
        np.savetxt(f"../lr_search/{N}-pendulum-training-loss/hgnn-train_{lr}.txt", larray, delimiter = "\n")
        np.savetxt(f"../lr_search/{N}-pendulum-training-loss/hgnn-test_{lr}.txt", ltarray, delimiter = "\n")

    if (if_act_search == 1):
        np.savetxt(f"../act_search/{N}-pendulum-training-time/hgnn_softplus.txt", train_time_arr, delimiter = "\n")
        np.savetxt(f"../act_search/{N}-pendulum-training-loss/hgnn-train_softplus.txt", larray, delimiter = "\n")
        np.savetxt(f"../act_search/{N}-pendulum-training-loss/hgnn-test_softplus.txt", ltarray, delimiter = "\n")

    if (if_mpass_search == 1):
        np.savetxt(f"../mpass_search/{N}-pendulum-training-time/hgnn_{mpass}.txt", train_time_arr, delimiter = "\n")
        np.savetxt(f"../mpass_search/{N}-pendulum-training-loss/hgnn-train_{mpass}.txt", larray, delimiter = "\n")
        np.savetxt(f"../mpass_search/{N}-pendulum-training-loss/hgnn-test_{mpass}.txt", ltarray, delimiter = "\n")

    if (if_hidden_search == 1):
        np.savetxt(f"../mlp_hidden_search/{N}-pendulum-training-time/hgnn_{hidden}.txt", train_time_arr, delimiter = "\n")
        np.savetxt(f"../mlp_hidden_search/{N}-pendulum-training-loss/hgnn-train_{hidden}.txt", larray, delimiter = "\n")
        np.savetxt(f"../mlp_hidden_search/{N}-pendulum-training-loss/hgnn-test_{hidden}.txt", ltarray, delimiter = "\n")

    if (if_nhidden_search == 1):
        np.savetxt(f"../mlp_nhidden_search/{N}-pendulum-training-time/hgnn_{nhidden}.txt", train_time_arr, delimiter = "\n")
        np.savetxt(f"../mlp_nhidden_search/{N}-pendulum-training-loss/hgnn-train_{nhidden}.txt", larray, delimiter = "\n")
        np.savetxt(f"../mlp_nhidden_search/{N}-pendulum-training-loss/hgnn-test_{nhidden}.txt", ltarray, delimiter = "\n")

main()
# main(lr=0.3)
# main(nhidden=4)
# main(nhidden=8)
# main(nhidden=16)











