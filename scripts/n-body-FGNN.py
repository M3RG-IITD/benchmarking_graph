################################################
################## IMPORT ######################
################################################

import json
import sys
from datetime import datetime
from functools import partial, wraps

import fire
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, value_and_grad, vmap
from jax.experimental import optimizers
from jax_md import space
from shadow.plot import *
#from sklearn.metrics import r2_score
import time

from psystems.nbody import (get_fully_connected_senders_and_receivers,get_fully_edge_order, get_init_conf)

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

# config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# jax.config.update('jax_platform_name', 'gpu')

class Datastate:
    def __init__(self, model_states):
        self.position = model_states.position[:-1]
        self.velocity = model_states.velocity[:-1]
        self.force = model_states.force[:-1]
        self.mass = model_states.mass[:-1]
        self.index = 0
        self.change_position = model_states.position[1:]-model_states.position[:-1]
        self.change_velocity = model_states.velocity[1:]-model_states.velocity[:-1]


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

def Main(N=4, epochs=10000, seed=42, rname=False, saveat=10, error_fn="L2error",
         dt=1.0e-3, ifdrag=0, stride=100, trainm=1, grid=False, mpass=1, lr=0.001,
         withdata=None, datapoints=None, batch_size=100, if_noisy_data=0):

    return wrap_main(main)(N=N, epochs=epochs, seed=seed, rname=rname, saveat=saveat, error_fn=error_fn,
                           dt=dt, ifdrag=ifdrag, stride=stride, trainm=trainm, grid=grid, mpass=mpass, lr=lr,
                           withdata=withdata, datapoints=datapoints, batch_size=batch_size, if_noisy_data=if_noisy_data)

def main(N=3, epochs=10000, seed=42, rname=True, saveat=10, error_fn="L2error",
         dt=1.0e-3, ifdrag=0, stride=100, trainm=1, grid=False, mpass=1, lr=0.001, withdata=None, datapoints=None, batch_size=1000, config=None, if_noisy_data=0):

    # print("Configs: ")
    # pprint(N, epochs, seed, rname,
    #        dt, stride, lr, ifdrag, batch_size,
    #        namespace=locals())

    randfilename = datetime.now().strftime(
        "%m-%d-%Y_%H-%M-%S") + f"_{datapoints}"

    PSYS = f"{N}-body"
    TAG = f"fgnn"

    if (if_noisy_data == 1):
        out_dir = f"../noisy_data"
    else:
        out_dir = f"../results"

    def _filename(name, tag=TAG):
        # rstring = randfilename if (rname and (tag != "data")) else (
        #     "1" if (tag == "data") or (withdata == None) else f"{withdata}")
        rstring = "1" if (tag == "data") else "0"

        if (tag == "data"):
            filename_prefix = f"../results/{PSYS}-{tag}/{1}/"
        else:
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

    # if len(dataset_states)*model_states.position.shape[0] != 10000:
    #     raise Exception("Invalid number of data points")

    N, dim = model_states.position.shape[-2:]
    species = jnp.zeros((N, 1), dtype=int)
    masses = jnp.ones((N, 1))

    # Rs, Vs, Fs = States().fromlist(dataset_states).get_array()
    # Rs = Rs.reshape(-1, N, dim)
    # Vs = Vs.reshape(-1, N, dim)
    # Fs = Fs.reshape(-1, N, dim)

    Rs, Vs, Fs, Rds, Vds = States_modified().fromlist(dataset_states).get_array()
    Rs = Rs.reshape(-1, N, dim)
    Vs = Vs.reshape(-1, N, dim)
    Fs = Fs.reshape(-1, N, dim)
    Rds = Rds.reshape(-1, N, dim)
    Vds = Vds.reshape(-1, N, dim)

    if (if_noisy_data == 1):
        Rs = np.array(Rs)
        Rds = np.array(Rds)
        Fs = np.array(Fs)
        Vs = np.array(Vs)
        Vds = np.array(Vds)

        np.random.seed(100)
        for i in range(len(Rs)):
            Rs[i] += np.random.normal(0,1,1)
            Rds[i] += np.random.normal(0,1,1)
            Vs[i] += np.random.normal(0,1,1)
            Vds[i] += np.random.normal(0,1,1)
            Fs[i] += np.random.normal(0,1,1)

        Rs = jnp.array(Rs)
        Rds = jnp.array(Rds)
        Fs = jnp.array(Fs)
        Vs = jnp.array(Vs)
        Vds = jnp.array(Vds)

    # mask = np.random.choice(len(Rs), len(Rs), replace=False)
    # allRs = Rs[mask]
    # allVs = Vs[mask]
    # allFs = Fs[mask]

    mask = np.random.choice(len(Rs), len(Rs), replace=False)
    allRs = Rs[mask]
    allVs = Vs[mask]
    allFs = Fs[mask]
    allRds = Rds[mask]
    allVds = Vds[mask]

    # Ntr = int(0.75*len(Rs))
    # Nts = len(Rs) - Ntr

    # Rs = allRs[:Ntr]
    # Vs = allVs[:Ntr]
    # Fs = allFs[:Ntr]

    # Rst = allRs[Ntr:]
    # Vst = allVs[Ntr:]
    # Fst = allFs[Ntr:]

    Ntr = int(0.75*len(Rs))
    Nts = len(Rs) - Ntr

    Rs = allRs[:Ntr]
    Vs = allVs[:Ntr]
    Fs = allFs[:Ntr]
    Rds = allRds[:Ntr]
    Vds = allVds[:Ntr]

    Rst = allRs[Ntr:]
    Vst = allVs[Ntr:]
    Fst = allFs[Ntr:]
    Rdst = allRds[Ntr:]
    Vdst = allVds[Ntr:]

    ################################################
    ################## SYSTEM ######################
    ################################################

    # pot_energy_orig = PEF
    # kin_energy = partial(lnn._T, mass=masses)

    # def Lactual(x, v, params):
    #     return kin_energy(v) - pot_energy_orig(x)

    # def constraints(x, v, params):
    #     return jax.jacobian(lambda x: hconstraints(x.reshape(-1, dim)), 0)(x)

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

    # if grid:
    #     print("It's a grid?")
    #     a = int(np.sqrt(N))
    #     senders, receivers = get_connections(a, a)
    #     eorder = edge_order(len(senders))
    # else:
    #     print("It's a random?")
    #     # senders, receivers = get_fully_connected_senders_and_receivers(N)
    #     print("Creating Chain")
    #     _, _, senders, receivers = chain(N)
    #     eorder = edge_order(len(senders))
    senders, receivers = get_fully_connected_senders_and_receivers(N)
    eorder = get_fully_edge_order(N)

    R, V = Rs[0], Vs[0]

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

    # if trainm:
    #     print("kinetic energy: learnable")

    #     def L_energy_fn(params, graph):
    #         L = fgn.cal_energy(params, graph, mpass=mpass)
    #         return L

    # else:
    #     print("kinetic energy: 0.5mv^2")

    #     kin_energy = partial(lnn._T, mass=masses)

    #     raise Warning("KE = 0.5mv2 not implemented")

    #     # def L_energy_fn(params, graph):
    #     #     g, V, T = cal_graph(params, graph, mpass=mpass, eorder=eorder,
    #     #                         useT=True, useonlyedge=True)
    #     #     return kin_energy(graph.nodes["velocity"]) - V

    hidden_dim = [16, 16]
    edgesize = 1
    nodesize = 1 + 2*dim
    ee = 8
    ne = 8
    Lparams = dict(
        ee_params=initialize_mlp([edgesize, ee], key),
        ne_params=initialize_mlp([nodesize, ne], key),
        e_params=initialize_mlp([ee+2*ne, *hidden_dim, ee], key),
        n_params=initialize_mlp([2*ee+ne, *hidden_dim, ne], key),
        g_params=initialize_mlp([ne, *hidden_dim, 1], key),
        acc_params=initialize_mlp([ne, *hidden_dim, dim], key),
        delta_params = initialize_mlp([ne, *hidden_dim, dim*2], key),
    )

    def acceleration_fn(params, graph):
        acc = fgn.cal_delta(params, graph, mpass=1)
        return acc

    def acc_fn(species):
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

    apply_fn = acc_fn(species)
    v_apply_fn = vmap(apply_fn, in_axes=(None, 0))

    def acceleration_fn_model(x, v, params): return apply_fn(x, v, params["L"])

    params = {"L": Lparams}

    print(acceleration_fn_model(R, V, params))

    # print("lag: ", Lmodel(R, V, params))

    # def nndrag(v, params):
    #     return - jnp.abs(models.forward_pass(params, v.reshape(-1), activation_fn=models.SquarePlus)) * v

    # if ifdrag == 0:
    #     print("Drag: 0.0")

    #     def drag(x, v, params):
    #         return 0.0
    # elif ifdrag == 1:
    #     print("Drag: nn")

    #     def drag(x, v, params):
    #         return vmap(nndrag, in_axes=(0, None))(v.reshape(-1), params["drag"]).reshape(-1, 1)

    # params["drag"] = initialize_mlp([1, 5, 5, 1], key)

    # acceleration_fn_model = jit(accelerationFull(N, dim,
    #                                              lagrangian=Lmodel,
    #                                              constraints=None,
    #                                              non_conservative_forces=drag))

    v_acceleration_fn_model = vmap(acceleration_fn_model, in_axes=(0, 0, None))

    ################################################
    ################## ML Training #################
    ################################################

    #LOSS = getattr(src.models, error_fn)

    @jit
    def loss_fn(params, Rs, Vs, Rds, Vds):
        pred = v_acceleration_fn_model(Rs, Vs, params)
        return MSE(pred, jnp.concatenate([Rds,Vds], axis=2))

    @jit
    def gloss(*args):
        return value_and_grad(loss_fn)(*args)

    opt_init, opt_update_, get_params = optimizers.adam(lr)

    @ jit
    def opt_update(i, grads_, opt_state):
        grads_ = jax.tree_map(jnp.nan_to_num, grads_)
        grads_ = jax.tree_map(
            partial(jnp.clip, a_min=-1000.0, a_max=1000.0), grads_)
        return opt_update_(i, grads_, opt_state)

    @jit
    def update(i, opt_state, params, loss__, *data):
        """ Compute the gradient for a batch and update the parameters """
        value, grads_ = gloss(params, *data)
        opt_state = opt_update(i, grads_, opt_state)
        return opt_state, get_params(opt_state), value

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

    bRs, bVs, bRds, bVds = batching(Rs, Vs, Rds, Vds,
                             size=min(len(Rs), batch_size))

    print(f"training ...")

    opt_state = opt_init(params)
    epoch = 0
    optimizer_step = -1
    larray = []
    ltarray = []
    last_loss = 1000

    start = time.time()
    train_time_arr = []

    larray += [loss_fn(params, Rs, Vs, Rds, Vds)]
    ltarray += [loss_fn(params, Rst, Vst, Rdst, Vdst)]

    def print_loss():
        print(
            f"Epoch: {epoch}/{epochs} Loss (mean of {error_fn}):  train={larray[-1]}, test={ltarray[-1]}")

    print_loss()

    for epoch in range(epochs):
        l = 0.0
        count = 0
        for data in zip(bRs, bVs, bRds, bVds):
            optimizer_step += 1
            opt_state, params, l_ = step(
                optimizer_step, (opt_state, params, 0), *data)

            l += l_
            count += 1

        # optimizer_step += 1
        # opt_state, params, l_ = step(
        #     optimizer_step, (opt_state, params, 0), Rs, Vs, Fs)
        l = l/count
        if epoch % 1 == 0:
            larray += [l]
            ltarray += [loss_fn(params, Rst, Vst, Rdst, Vdst)]
            print_loss()

        if epoch % saveat == 0:
            metadata = {
                "savedat": epoch,
                "mpass": mpass,
                "grid": grid,
                "ifdrag": ifdrag,
                "trainm": trainm,
            }
            savefile(f"trained_model_{ifdrag}_{trainm}.dil",
                     params, metadata=metadata)
            savefile(f"loss_array_{ifdrag}_{trainm}.dil",
                     (larray, ltarray), metadata=metadata)
            if last_loss > larray[-1]:
                last_loss = larray[-1]
                savefile(f"trained_model_low_{ifdrag}_{trainm}.dil",
                         params, metadata=metadata)

        now = time.time()
        train_time_arr.append((now - start))

        fig, axs = panel(1, 1)
        plt.semilogy(larray[1:], label="Training")
        plt.semilogy(ltarray[1:], label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(_filename(f"training_loss_{ifdrag}_{trainm}.png"))

    metadata = {
        "savedat": epoch,
        "mpass": mpass,
        "grid": grid,
        "ifdrag": ifdrag,
        "trainm": trainm,
    }
    params = get_params(opt_state)
    savefile(f"trained_model_{ifdrag}_{trainm}.dil",
             params, metadata=metadata)
    savefile(f"loss_array_{ifdrag}_{trainm}.dil",
             (larray, ltarray), metadata=metadata)

    np.savetxt(f"../{N}-body-training-time/hgnn.txt", train_time_arr, delimiter = "\n")
    np.savetxt(f"../{N}-body-training-loss/hgnn-train.txt", larray, delimiter = "\n")
    np.savetxt(f"../{N}-body-training-loss/hgnn-test.txt", ltarray, delimiter = "\n")

fire.Fire(Main)



