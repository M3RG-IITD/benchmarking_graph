################################################
################## IMPORT ######################
################################################

import json
import sys
from datetime import datetime
from functools import partial, wraps
from statistics import mode
import time

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

MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8

import jraph
import src
from jax.config import config
from src import fgn, lnn
from src.graph import *
from src.lnn import acceleration, accelerationFull, accelerationTV, acceleration_GNODE
from src.md import *
from src.models import MSE, initialize_mlp
from src.nve import nve
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


# import pickle
# data = pickle.load(open('../results/LJ-data/0/graphs_dicts.pkl','rb'))[0]
# dd = data[0]['nodes']['position']
# data[1]


acceleration = []
damage = []
id = []
mass = []
position = []
type = []
velocity = []
volume = []
import pandas as pd
for num in (np.linspace(0,5000,251).astype('int')):
    dataf_name = f"env_1_step_{num}.jld.data"
    df = pd.read_csv(f'../results/peridynamics-data/datafiles/{dataf_name}')    
    split_df = df.iloc[1:,0].str.split(expand=True)
    acceleration += [(np.array(split_df[[0,1,2]]).astype('float64'))]
    damage += [np.array(split_df[[3]]).astype('float64')]
    id += [np.array(split_df[[4]]).astype('float64')]
    mass += [np.array(split_df[[5]]).astype('float64')]
    position += [np.array(split_df[[6,7,8]]).astype('float64')]
    type += [np.array(split_df[[9]]).astype('float64')]
    velocity += [np.array(split_df[[10,11,12]]).astype('float64')]
    volume += [np.array(split_df[[13]]).astype('float64')]


Rs = jnp.array(position)
Vs = jnp.array(velocity)
Fs = jnp.array(acceleration)
print(Rs.shape)
print(Fs.shape)
Zs_dot = jnp.concatenate([Vs,Fs], axis=1)
print(Zs_dot.shape)
#sys.exit()


o_position = position[0]/1.1
N,dim = o_position.shape
species = jnp.zeros(N, dtype=int)

def displacement(a, b):
    return a - b

# make_graph(o_position,displacement[0],species=species,atoms={0: 125},V=velocity[0],A=acceleration[0],mass=mass[0],cutoff=3.0)
my_graph0_disc = make_graph(o_position,displacement,atoms={0: 125},cutoff=3.0)


epochs=10000
seed=42
rname=False
dt=1.0e-3
ifdrag=0
stride=100
trainm=1
lr=0.001
withdata=None
datapoints=None
batch_size=20
ifDataEfficiency = 0


# def main(N=5, epochs=10000, seed=42, rname=True, dt=1.0e-3, ifdrag=0, stride=100, trainm=1, lr=0.001, withdata=None, datapoints=None, batch_size=100):
# print("Configs: ")
# pprint(N, epochs, seed, rname,
#         dt, stride, lr, ifdrag, batch_size,
#         namespace=locals())

randfilename = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + f"_{datapoints}"

PSYS = f"peridynamics"
TAG = f"HGN"
out_dir = f"../results"

def _filename(name, tag=TAG):
    rstring = randfilename if (rname and (tag != "data")) else (
        "0" if (tag == "data") or (withdata == None) else f"0_{withdata}")
    filename_prefix = f"{out_dir}/{PSYS}-{tag}/{rstring}/"
    file = f"{filename_prefix}/{name}"
    os.makedirs(os.path.dirname(file), exist_ok=True)
    filename = f"{filename_prefix}/{name}".replace("//", "/")
    print("===", filename, "===")
    return filename

# def displacement(a, b):
#     return a - b

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

# try:
#     graphs = loadfile(f"env_1_step_0.jld.data", tag="data")
# except:
#     raise Exception("Generate dataset first.")



species = jnp.zeros(N, dtype=int)
masses = jnp.ones(N)

# Rs, Vs, Fs = States(graphs).get_array()


mask = np.random.choice(len(Rs), len(Rs), replace=False)
allRs = Rs[mask]
allVs = Vs[mask]
allFs = Fs[mask]
allZs_dot = Zs_dot[mask]

Ntr = int(0.75*len(allRs))
Nts = len(allRs) - Ntr

Rs = allRs[:Ntr]
Vs = allVs[:Ntr]
Fs = allFs[:Ntr]
Zs_dot = allZs_dot[:Ntr]

Rst = allRs[Ntr:]
Vst = allVs[Ntr:]
Fst = allFs[Ntr:]
Zst_dot = allZs_dot[Ntr:]

print(f"training data shape(Rs): {Rs.shape}")
print(f"test data shape(Rst): {Rst.shape}")

################################################
################## SYSTEM ######################
################################################

# peridynamics_sim

################################################
################### ML Model ###################
################################################

dim = 3
# Ef = 1  # eij dim
# Nf = dim
# Oh = 1

# Eei = 8
# Nei = 8
# Nei_ = 5  ##Nei for mass

# hidden = 8
# nhidden = 2

# def get_layers(in_, out_):
#     return [in_] + [hidden]*nhidden + [out_]

# def mlp(in_, out_, key, **kwargs):
#     return initialize_mlp(get_layers(in_, out_), key, **kwargs)

# fneke_params = initialize_mlp([Oh, Nei], key)
# fne_params = initialize_mlp([Oh, Nei], key)

# fb_params = mlp(Ef, Eei, key)
# fv_params = mlp(Nei+Eei, Nei, key)
# fe_params = mlp(Nei, Eei, key)

# ff1_params = mlp(Eei, 1, key)
# ff2_params = mlp(Nei, 1, key)
# ff3_params = mlp(dim+Nei, 1, key)
# ke_params = initialize_mlp([1+Nei, 10, 10, 1], key, affine=[True])
# mass_params = initialize_mlp([Nei_, 5, 1], key, affine=[True]) #

# Hparams = dict(fb=fb_params,
#                 fv=fv_params,
#                 fe=fe_params,
#                 ff1=ff1_params,
#                 ff2=ff2_params,
#                 ff3=ff3_params,
#                 fne=fne_params,
#                 fneke=fneke_params,
#                 ke=ke_params,
#                 mass=mass_params)

# #params = {"Fqqdot": Fparams}

# def H_energy_fn(params, graph):
#         g, V, T = cal_graph(params, graph, eorder=None,
#                             useT=True)
#         return T + V

# def graph_force_fn(params, graph):
#     _GForce = a_gnode_cal_force_q_qdot(params, graph, eorder=None,
#                         useT=True)
#     return _GForce

R, V = Rs[0], Vs[0]

my_graph0_disc.pop("e_order")
my_graph0_disc.pop("atoms")
my_graph0_disc.update({"globals": None})

mask = my_graph0_disc['senders'] != my_graph0_disc['receivers']

my_graph0_disc.update({"senders": my_graph0_disc['senders'][mask]})

my_graph0_disc.update({"receivers": my_graph0_disc['receivers'][mask]})

my_graph0_disc.update({"n_edge": mask.sum()})

senders = my_graph0_disc['senders']
receivers = my_graph0_disc['receivers']

graph = jraph.GraphsTuple(**my_graph0_disc)

def dist(*args):
        disp = displacement(*args)
        return jnp.sqrt(jnp.square(disp).sum())

dij = vmap(dist, in_axes=(0, 0))(R[senders], R[receivers])


hidden_dim = [16, 16]
edgesize = 1
nodesize = 1 + 2*dim
ee = 8
ne = 8
Hparams = dict(
    ee_params=initialize_mlp([edgesize, ee], key),
    ne_params=initialize_mlp([nodesize, ne], key),
    e_params=initialize_mlp([ee+2*ne, *hidden_dim, ee], key),
    n_params=initialize_mlp([2*ee+ne, *hidden_dim, ne], key),
    g_params=initialize_mlp([ne, *hidden_dim, 1], key),
    acc_params=initialize_mlp([ne, *hidden_dim, dim], key),
    lgn_params = initialize_mlp([ne, *hidden_dim, 1], key),
)

def acceleration_fn(params, graph):
    acc = fgn.cal_lgn(params, graph, mpass=1)
    return acc

def acc_fn(species):
    state_graph = graph

    def apply(R, V, params):
        state_graph.nodes.update(position=R)
        state_graph.nodes.update(velocity=V)
        state_graph.edges.update(dij=vmap(dist, in_axes=(0, 0))(R[senders], R[receivers])
                                    )
        return acceleration_fn(params, state_graph)
    return apply


apply_fn = acc_fn(species)
v_apply_fn = vmap(apply_fn, in_axes=(None, 0))

def Hmodel(x, v, params): return apply_fn(x, v, params["H"])

params = {"H": Hparams}

# def energy_fn(species):
#     state_graph = graph
#     def apply(R, V, params):
#         state_graph.nodes.update(position=R)
#         state_graph.nodes.update(velocity=V)
#         return H_energy_fn(params, state_graph)
#     return apply


# apply_fn = energy_fn(species)
# v_apply_fn = vmap(apply_fn, in_axes=(None, 0))

# def Hmodel(x, v, params):
#     return apply_fn(x, v, params["H"])

# params = {"H": Hparams}


def nndrag(v, params):
    return - jnp.abs(models.forward_pass(params, v.reshape(-1), activation_fn=models.SquarePlus)) * v


if ifdrag == 0:
    print("Drag: 0.0")

    def drag(x, v, params):
        return 0.0
elif ifdrag == 1:
    print("Drag: -0.1*v")

    def drag(x, v, params):
        return vmap(nndrag, in_axes=(0, None))(v.reshape(-1), params["drag"]).reshape(-1, 1)

params["drag"] = initialize_mlp([1, 5, 5, 1], key)

zdot_model, lamda_force_model = get_zdot_lambda(
    N, dim, hamiltonian=Hmodel, drag=drag, constraints=None)


v_zdot_model = vmap(zdot_model, in_axes=(0, 0, None))

# def F_q_qdot(x, v, params): return apply_fn(x, v, params["Fqqdot"])

# acceleration_fn_model = F_q_qdot
# # acceleration_fn_model = acceleration_GNODE(N, dim, F_q_qdot,
# #                                             constraints=None)



# v_acceleration_fn_model = vmap(acceleration_fn_model, in_axes=(0, 0, None))

# v_acceleration_fn_model(Rs[:10], Vs[:10], params)

print(zdot_model(R,V, params))
# sys.exit()


################################################
################## ML Training #################
################################################
@jit
def loss_fn(params, Rs, Vs, Zs_dot):
    pred = v_zdot_model(Rs, Vs, params)
    return MSE(pred, Zs_dot)

# loss_fn(params, Rs[:1], Vs[:1], Fs[:1])

def gloss(*args):
    return value_and_grad(loss_fn)(*args)

def update(i, opt_state, params, loss__, *data):
    """ Compute the gradient for a batch and update the parameters """
    value, grads_ = gloss(params, *data)
    opt_state = opt_update(i, grads_, opt_state)
    return opt_state, get_params(opt_state), value

@jit
def step(i, ps, *args):
    return update(i, *ps, *args)

opt_init, opt_update_, get_params = optimizers.adam(lr)

@jit
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

bRs, bVs, bZs_dot = batching(Rs, Vs, Zs_dot,
                                size=min(len(Rs), batch_size))

print(f"training ...")

start = time.time()
train_time_arr = []

opt_state = opt_init(params)
epoch = 0
optimizer_step = -1
larray = []
ltarray = []
last_loss = 1000
for epoch in range(epochs):
    l = 0.0
    for data in zip(bRs, bVs, bZs_dot):
        optimizer_step += 1
        opt_state, params, l_ = step(
            optimizer_step, (opt_state, params, 0), *data)
        l += l_
    
    opt_state, params, l_ = step(
        optimizer_step, (opt_state, params, 0), Rs, Vs, Zs_dot)
    larray += [l_]
    ltarray += [loss_fn(params, Rst, Vst ,Zst_dot)]
    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch}/{epochs} Loss (MSE):  train={larray[-1]}, test={ltarray[-1]}")
    if epoch % 10 == 0:
        metadata = {
            "savedat": epoch,
            # "mpass": mpass,
            "ifdrag": ifdrag,
            "trainm": trainm,
        }
        savefile(f"perignode_trained_model_{ifdrag}_{trainm}.dil",
                    params, metadata=metadata)
        savefile(f"loss_array_{ifdrag}_{trainm}.dil",
                    (larray, ltarray), metadata=metadata)
        if last_loss > larray[-1]:
            last_loss = larray[-1]
            savefile(f"perignode_trained_model_{ifdrag}_{trainm}_low.dil",
                        params, metadata=metadata)
        fig, axs = panel(1, 1)
        plt.semilogy(larray, label="Training")
        plt.semilogy(ltarray, label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(_filename(f"training_loss_{ifdrag}_{trainm}.png"))

        if (ifDataEfficiency == 0):
            np.savetxt("../peridynamics-training-time/hgn.txt", train_time_arr, delimiter = "\n")
            np.savetxt("../peridynamics-training-loss/hgn-train.txt", larray, delimiter = "\n")
            np.savetxt("../peridynamics-training-loss/hgn-test.txt", ltarray, delimiter = "\n")

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
savefile(f"perignode_trained_model_{ifdrag}_{trainm}.dil",
            params, metadata=metadata)
savefile(f"loss_array_{ifdrag}_{trainm}.dil",
            (larray, ltarray), metadata=metadata)

if last_loss > larray[-1]:
    last_loss = larray[-1]
    savefile(f"perignode_trained_model_{ifdrag}_{trainm}_low.dil",
                params, metadata=metadata)

if (ifDataEfficiency == 0):
        np.savetxt("../peridynamics-training-time/hgn.txt", train_time_arr, delimiter = "\n")
        np.savetxt("../peridynamics-training-loss/hgn-train.txt", larray, delimiter = "\n")
        np.savetxt("../peridynamics-training-loss/hgn-test.txt", ltarray, delimiter = "\n")


# fire.Fire(main)
