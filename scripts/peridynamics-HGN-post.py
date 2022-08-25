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
from sklearn.metrics import r2_score
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
import time


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
Zs_dot = jnp.concatenate([Vs,Fs], axis=1)

o_position = position[0]/1.1
N,dim = o_position.shape
species = jnp.zeros(N, dtype=int)

def displacement(a, b):
    return a - b

# make_graph(o_position,displacement[0],species=species,atoms={0: 125},V=velocity[0],A=acceleration[0],mass=mass[0],cutoff=3.0)
my_graph0_disc = make_graph(o_position,displacement,atoms={0: 125},cutoff=3.0)


dt=1.0e-3
# useN=None
withdata=None
datapoints=None
# mpass=1
# grid=False
stride=100
ifdrag=0
seed=42
rname=0
saveovito=1
trainm=1
runs=100
semilog=1
maxtraj=10
plotthings=True
redo=0



# def main(N=5, epochs=10000, seed=42, rname=True, dt=1.0e-3, ifdrag=0, stride=100, trainm=1, lr=0.001, withdata=None, datapoints=None, batch_size=100):
# print("Configs: ")
# pprint(N, epochs, seed, rname,
#         dt, stride, lr, ifdrag, batch_size,
#         namespace=locals())



PSYS = f"peridynamics"
TAG = f"HGN"
out_dir = f"../results"

randfilename = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + f"_{datapoints}"

def _filename(name, tag=TAG, trained=None):
    if tag == "data":
        part = f"_{ifdrag}."
    else:
        part = f"_{ifdrag}_{trainm}."
    if trained is not None:
        psys = f"{trained}-{PSYS.split('-')[0]}"
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

################################################
################## SYSTEM ######################
################################################

# peridynamics_sim

origin_acceleration = []
origin_mass = []
origin_position = []
origin_velocity = []
import pandas as pd
for num in range(1000):
    dataf_name = f"env_1_step_{num}.jld.data"
    df = pd.read_csv(f'../results/peridynamics-MCGNODE/test/{dataf_name}')
    split_df = df.iloc[1:,0].str.split(expand=True)
    
    origin_acceleration += [(np.array(split_df[[0,1,2]]).astype('float64'))]
    origin_mass += [np.array(split_df[[5]]).astype('float64')]
    origin_position += [np.array(split_df[[6,7,8]]).astype('float64')]
    origin_velocity += [np.array(split_df[[10,11,12]]).astype('float64')]


origin_Rs = jnp.array(origin_position)
origin_Vs = jnp.array(origin_velocity)
origin_Fs = jnp.array(origin_acceleration)
origin_mass = jnp.array(origin_mass)
origin_Zs_dot = jnp.concatenate([origin_Vs,origin_Fs], axis=1)


################################################
################### ML Model ###################
################################################

def H_energy_fn(params, graph):
        g, V, T = cal_graph(params, graph, eorder=None,
                            useT=True)
        return T + V

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

R = jnp.array(R)
V = jnp.array(V)
species = jnp.array(species).reshape(-1, 1)

dij = vmap(dist, in_axes=(0, 0))(R[senders], R[receivers])



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

#params["drag"] = initialize_mlp([1, 5, 5, 1], key)

zdot_model, lamda_force_model = get_zdot_lambda(
    N, dim, hamiltonian=Hmodel, drag=drag, constraints=None)

def zdot_model_func(z, t, params):
        x, p = jnp.split(z, 2)
        return zdot_model(x, p, params)


v_zdot_model = vmap(zdot_model, in_axes=(0, 0, None))
# acceleration_fn_model = acceleration_GNODE(N, dim, F_q_qdot,
#                                             constraints=None)

# def force_fn_model(R, V, params, mass=None):
#     if mass is None:
#         return acceleration_fn_model(R, V, params)
#     else:
#         return acceleration_fn_model(R, V, params)
        # return acceleration_fn_model(R, V, params)*mass.reshape(-1, 1)

params = loadfile(f"perignode_trained_model_low.dil")[0]

def z0(x, p):
    return jnp.vstack([x, p])

def get_forward_sim(params=None, zdot_func=None, runs=10):
        def fn(R, V):
            t = jnp.linspace(0.0, runs*stride*dt, runs*stride)
            _z_out = ode.odeint(zdot_func, z0(R, V), t, params)
            return _z_out[0::stride]
        return fn

sim_model = get_forward_sim(
        params=params, zdot_func=zdot_model_func, runs=runs)

# my_sim = sim_model(R, V)

# v_acceleration_fn_model = vmap(acceleration_fn_model, in_axes=(0, 0, None))

# v_acceleration_fn_model(Rs[:10], Vs[:10], params)


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

def AbsErr(ya, yp):
    return norm(ya-yp)


nexp = {
        "z_pred": [],
        "z_actual": [],
        "Zerr": [],
        "AbsZerr":[],
        "Perr": [],
        "AbsPerr": []
    }

t=0.0
for ind in range(maxtraj):
    print(f"Simulating trajectory {ind}/{maxtraj} ...")
    R, V = Rs[runs*ind], Vs[runs*ind]
    
    start = time.time()
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
    end = time.time()
    t += end - start
    # ll = [state for state in NVEStates(pred_traj)]
    # save_ovito(f"pred_{ind}.data",[state for state in NVEStates(pred_traj)], lattice="")
    # if ind>20:
    #     break
    
    sim_size = runs
    nexp["z_pred"] += [pred_traj.position]
    nexp["z_actual"] += [origin_Rs[runs*ind:runs+runs*ind]]
    nexp["Zerr"] += [RelErr(origin_Rs[runs*ind:runs+runs*ind], pred_traj.position)+1e-30]
    # nexp["AbsZerr"] += [AbsErr(origin_Rs[runs*ind:runs+runs*ind], pred_traj.position)]
    nexp["AbsZerr"] += [jnp.abs(norm(origin_Rs[runs*ind:runs+runs*ind]) - norm(pred_traj.position))]

    ac_mom = jnp.square(origin_Vs[runs*ind:runs+runs*ind].sum(1)).sum(1)
    pr_mom = jnp.square(pred_traj.velocity.sum(1)).sum(1)
    # nexp["Perr"] += ([RelErr(origin_Vs[runs*ind:runs+runs*ind], pred_traj.velocity)])
    nexp["Perr"] += ([RelErr(origin_Vs[runs*ind:runs+runs*ind][6:], pred_traj.velocity[6:])+1e-30])
    nexp["AbsPerr"] += ([jnp.abs(ac_mom - pr_mom)+1e-30])
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
    plt.ylabel(yl)
    plt.xlabel("Time")
    plt.savefig(_filename(f"RelError_std_{key}.png"))

make_plots(nexp, "Zerr",
            yl=r"$\frac{||\hat{z}-z||_2}{||\hat{z}||_2+||z||_2}$")

make_plots(nexp, "Perr",
            yl=r"$\frac{||\hat{p}-p||_2}{||\hat{p}||_2+||p||_2}$")

np.savetxt(f"../peridynamics-simulation-time/hgn.txt", [t/maxtraj], delimiter = "\n")
# make_plots(nexp, "AbsZerr", yl=r"${||\hat{z}-z||_2}$")
# make_plots(nexp, "Herr",
#             yl=r"$\frac{||H(\hat{z})-H(z)||_2}{||H(\hat{z})||_2+||H(z)||_2}$")
# make_plots(nexp, "AbsHerr", yl=r"${||H(\hat{z})-H(z)||_2}$")

