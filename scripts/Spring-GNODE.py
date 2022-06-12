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
from shadow.plot import *
#from sklearn.metrics import r2_score
# from sympy import LM
# from torch import batch_norm_gather_stats_with_counts

from psystems.nsprings import (chain, edge_order, get_connections,
                               get_fully_connected_senders_and_receivers,
                               get_fully_edge_order)

MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8

import jraph
import src
from jax.config import config
from src import lnn
from src.graph1 import *
from src.lnn1 import acceleration, accelerationFull, accelerationTV, acceleration_GNODE
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

N=5
epochs=10000
seed=42
rname=True
dt=1.0e-3
ifdrag=0
stride=100
trainm=1
# mpass=1
lr=0.001
withdata=None
datapoints=None
batch_size=100
ifDataEfficiency = 1

# def main(N=3, epochs=100, seed=42, rname=True, saveat=1,
#          dt=1.0e-5, ifdrag=0, stride=100, trainm=1, grid=False, mpass=1, lr=0.001, withdata=None, datapoints=None, batch_size=1000):

if (ifDataEfficiency == 1):
    data_points = int(sys.argv[1])
    batch_size = int(data_points/100)

print("Configs: ")
pprint(N, epochs, seed, rname,
        dt, stride, lr, ifdrag, batch_size,
        namespace=locals())

randfilename = datetime.now().strftime(
    "%m-%d-%Y_%H-%M-%S") + f"_{datapoints}"

PSYS = f"{N}-Spring"
TAG = f"gnode"

if (ifDataEfficiency == 1):
    out_dir = f"../data-efficiency"
else:
    out_dir = f"../results"

def _filename(name, tag=TAG):
    rstring = randfilename if (rname and (tag != "data")) else (
        "0" if (tag == "data") or (withdata == None) else f"0_{withdata}")
    rstring = "0"
    if (ifDataEfficiency == 1):
            rstring = "0_" + str(data_points)
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

print(
    f"Total number of data points: {len(dataset_states)}x{model_states.position.shape[0]}")

N, dim = model_states.position.shape[-2:]
species = jnp.zeros(N, dtype=int)
masses = jnp.ones(N)

Rs, Vs, Fs = States().fromlist(dataset_states).get_array()
Rs = Rs.reshape(-1, 1, N, dim)
Vs = Vs.reshape(-1, 1, N, dim)
Fs = Fs.reshape(-1, 1, N, dim)

mask = np.random.choice(len(Rs), len(Rs), replace=False)
allRs = Rs[mask]
allVs = Vs[mask]
allFs = Fs[mask]

Ntr = int(0.75*len(Rs))
Nts = len(Rs) - Ntr

Rs = allRs[:Ntr]
Vs = allVs[:Ntr]
Fs = allFs[:Ntr]

Rst = allRs[Ntr:]
Vst = allVs[Ntr:]
Fst = allFs[Ntr:]

print(f"training data shape(Rs): {Rs.shape}")
print(f"test data shape(Rst): {Rst.shape}")

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

print("Creating Chain")
_, _, senders, receivers = chain(N)
eorder = edge_order(len(senders))

Ef = dim  # eij dim
Nf = dim
Oh = 1

Eei = 5
Nei = 5
Nei_ = 5  ##Nei for mass

hidden = 5
nhidden = 2

def get_layers(in_, out_):
    return [in_] + [hidden]*nhidden + [out_]

def mlp(in_, out_, key, **kwargs):
    return initialize_mlp(get_layers(in_, out_), key, **kwargs)

fneke_params = initialize_mlp([Oh, Nei], key)
fne_params = initialize_mlp([Oh, Nei], key)  #

# Nei = Nei+dim+dim
fb_params = mlp(Ef, Eei, key)  #
fv_params = mlp(Nei+Eei, Nei, key)  #
fe_params = mlp(Nei, Eei, key)  #

ff1_params = mlp(Eei, dim, key)
ff2_params = mlp(Nei, dim, key) #
ff3_params = mlp(Nei+dim+dim, dim, key)
ke_params = initialize_mlp([1+Nei, 10, 10, 1], key, affine=[True])
mass_params = initialize_mlp([Nei_, 5, 1], key, affine=[True]) #

Fparams = dict(fb=fb_params,
                fv=fv_params,
                fe=fe_params,
                ff1=ff1_params,
                ff2=ff2_params,
                ff3=ff3_params,
                fne=fne_params,
                fneke=fneke_params,
                ke=ke_params,
                mass=mass_params)

params = {"Fqqdot": Fparams}

def graph_force_fn(params, graph):
    _GForce = a_cdgnode_cal_force_q_qdot(params, graph, eorder=None,
                        useT=True)
    return _GForce


R, V = Rs[0][0], Vs[0][0]

def _force_fn(species):
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
        return graph_force_fn(params, state_graph)
    return apply

apply_fn = _force_fn(species)
# v_apply_fn = vmap(apply_fn, in_axes=(None, 0))

def F_q_qdot(x, v, params): return apply_fn(x, v, params["Fqqdot"])

# x=R
# v=V
# print(F_q_qdot(x, v, params))

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
acceleration_fn_model = F_q_qdot
# acceleration_fn_model = acceleration_GNODE(N, dim,F_q_qdot,
#                                             constraints=None,
#                                             non_conservative_forces=None)
v_acceleration_fn_model = vmap(acceleration_fn_model, in_axes=(0, 0, None))
v_v_acceleration_fn_model = vmap(v_acceleration_fn_model, in_axes=(0, 0, None))


# x=Rs[0]
# v=Vs[0]
# F_q_qdot(x[0], v[0], params)
# acceleration_fn_model(x[0], v[0], params)
# hhhh = v_v_acceleration_fn_model(Rs, Vs, params)
# print(hhhh)
# print(hhhh.shape)

################################################
################## ML Training #################
################################################

@jit
def loss_fn(params, Rs, Vs, Fs):
    pred = v_v_acceleration_fn_model(Rs, Vs, params)
    return MSE(pred, Fs)

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
    grads_ = jax.tree_map(partial(jnp.clip, a_min=-1000.0, a_max=1000.0), grads_)
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

bRs, bVs, bFs = batching(Rs, Vs, Fs,
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
for epoch in range(epochs):
    l = 0.0
    count = 0
    for data in zip(bRs, bVs, bFs):
        optimizer_step += 1
        opt_state, params, l_ = step(
            optimizer_step, (opt_state, params, 0), *data)
        l += l_
        count+=1

    # opt_state, params, l_ = step(
    #     optimizer_step, (opt_state, params, 0), Rs, Vs, Fs)
    l = l/count
    larray += [l]
    ltarray += [loss_fn(params, Rst, Vst, Fst)]

    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch}/{epochs} Loss (MSE):  train={larray[-1]}, test={ltarray[-1]}")
    if epoch % 100 == 0:
        metadata = {
            "savedat": epoch,
            # "mpass": mpass,
            "ifdrag": ifdrag,
            "trainm": trainm,
        }
        savefile(f"cgnode_trained_model_{ifdrag}_{trainm}.dil",
                    params, metadata=metadata)
        savefile(f"loss_array_{ifdrag}_{trainm}.dil",
                    (larray, ltarray), metadata=metadata)
        if last_loss > larray[-1]:
            last_loss = larray[-1]
            savefile(f"cgnode_trained_model_low_{ifdrag}_{trainm}.dil",
                        params, metadata=metadata)

        plt.clf()
        fig, axs = panel(1, 1)
        plt.semilogy(larray, label="Training")
        plt.semilogy(ltarray, label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(_filename(f"training_loss_{ifdrag}_{trainm}.png"))
    now = time.time()
    train_time_arr.append((now - start))

plt.clf()
fig, axs = panel(1, 1)
plt.semilogy(larray, label="Training")
plt.semilogy(ltarray, label="Test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(_filename(f"training_loss_{ifdrag}_{trainm}.png"))

params = get_params(opt_state)
savefile(f"cgnode_trained_model_{ifdrag}_{trainm}.dil",
            params, metadata=metadata)
savefile(f"loss_array_{ifdrag}_{trainm}.dil",
            (larray, ltarray), metadata=metadata)

if last_loss > larray[-1]:
    last_loss = larray[-1]
    savefile(f"cgnode_trained_model_low_{ifdrag}_{trainm}.dil",
                params, metadata=metadata)
if (ifDataEfficiency == 0):
    np.savetxt("../5-spring-training-time/cgnode.txt", train_time_arr, delimiter = "\n")
    np.savetxt("../5-spring-training-loss/cgnode-train.txt", larray, delimiter = "\n")
    np.savetxt("../5-spring-training-loss/cgnode-test.txt", ltarray, delimiter = "\n")

# fire.Fire(main)
