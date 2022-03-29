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
from torch import ne

from psystems.npendulum import (PEF, edge_order, get_init, hconstraints,
                                pendulum_connections)

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
from src.nve import NVEState, NVEStates, nve
from src.utils import *

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
# jax.config.update('jax_platform_name', 'gpu')


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def pprint(*args, namespace=globals()):
    for arg in args:
        print(f"{namestr(arg, namespace)[0]}: {arg}")


def main(N=2, dim=2, dt=1.0e-5, ifdrag=0, saveovito=0, semilog=1, stride=1000, seed=42, rname=0, runs=10, maxtraj=1, plotthings=False):

    print("Configs: ")
    pprint(dt, stride,
           namespace=locals())

    PSYS = f"{N}-Pendulum"
    TAG = f"lnn"
    out_dir = f"../results"

    def _filename(name, tag=TAG):
        rstring = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") if rname else "0"
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

    # dataset_states = loadfile("model_states_{}.pkl", tag="data")[0]
    # model_states = dataset_states[0]

    # R = model_states.position[0]
    # V = model_states.velocity[0]

    # print(
    #     f"Total number of training data points: {len(dataset_states)}x{model_states.position.shape[0]}")

    # N, dim = model_states.position.shape[-2:]

    R, V = get_init(N, dim=dim, angles=(-90, 90))
    species = jnp.zeros(N, dtype=int)
    masses = jnp.ones(N)

    ################################################
    ################## SYSTEM ######################
    ################################################

    pot_energy_orig = PEF
    kin_energy = partial(lnn._T, mass=masses)

    def Lactual(x, v, params):
        return kin_energy(v) - pot_energy_orig(x)

    def constraints(x, v, params):
        return jax.jacobian(lambda x: hconstraints(x.reshape(-1, dim)), 0)(x)

    def external_force(x, v, params):
        F = 0*R
        F = jax.ops.index_update(F, (1, 1), -1.0)
        return F.reshape(-1, 1)

    if ifdrag:
        print("Drag: -0.1v")

        def drag(x, v, params):
            return -0.1*v.reshape(-1, 1)
    else:
        def drag(x, v, params):
            return 0.0

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

    def get_forward_sim(params=None, force_fn=None, runs=10):
        @jit
        def fn(R, V):
            return predition(R,  V, params, force_fn, shift, dt, masses, stride=stride, runs=runs)
        return fn

    sim_orig = get_forward_sim(params=None, force_fn=force_fn_orig, runs=runs)

    ################################################
    ################### ML Model ###################
    ################################################

    def Lmodel(x, v, params):
        return ((params["lnn_ke"] * jnp.square(v).sum(axis=1)).sum() -
                forward_pass(params["lnn_pe"], x.flatten(), activation_fn=SquarePlus)[0])

    def nndrag(v, params):
        return - jnp.abs(models.forward_pass(params, v.reshape(-1), activation_fn=models.SquarePlus)) * v

    if ifdrag:
        print("Drag: learnable")

        def drag(x, v, params):
            return vmap(nndrag, in_axes=(0, None))(v.reshape(-1), params["drag"]).reshape(-1, 1)
    else:
        print("No drag.")

        def drag(x, v, params):
            return 0.0

    acceleration_fn_model = src.lnn.accelerationFull(N, dim,
                                                     lagrangian=Lmodel,
                                                     constraints=constraints,
                                                     non_conservative_forces=drag)

    v_acceleration_fn_model = vmap(acceleration_fn_model, in_axes=(0, 0, None))

    def force_fn_model(R, V, params, mass=None):
        if mass is None:
            return acceleration_fn_model(R, V, params)
        else:
            return acceleration_fn_model(R, V, params)*mass.reshape(-1, 1)

    params = loadfile(f"lnn_trained_model_{ifdrag}.dil")[0]

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

    for ind in range(maxtraj):

        if ind >= maxtraj:
            break

        print(f"Simulating trajectory {ind}/{maxtraj} ...")

        # R = model_states.position[0]
        # V = model_states.velocity[0]

        R, V = get_init(N, dim=dim, angles=(-90, 90))

        actual_traj = sim_orig(R, V)
        pred_traj = sim_model(R, V)

        if saveovito:
            save_ovito(f"lnn_pred_{ind}_{ifdrag}.ovito", [
                state for state in NVEStates(pred_traj)], lattice="")
            save_ovito(f"lnn_actual_{ind}_{ifdrag}.ovito", [
                state for state in NVEStates(actual_traj)], lattice="")

        trajectories += [(actual_traj, pred_traj)]
        savefile(f"trajectories_{ifdrag}.pkl", trajectories)

        if plotthings:
            for key, traj in {"actual": actual_traj, "pred": pred_traj}.items():

                print(f"plotting energy ({key})...")

                Es = Es_fn(traj)
                Es_pred = Es_pred_fn(traj)

                Es_pred = Es_pred - Es_pred[0] + Es[0]

                fig, axs = panel(1, 2, figsize=(20, 5))
                axs[0].plot(Es, label=["PE", "KE", "L", "TE"], lw=6, alpha=0.5)
                axs[1].plot(Es_pred, "--", label=["PE", "KE", "L", "TE"])
                plt.legend(bbox_to_anchor=(1, 1), loc=2)
                axs[0].set_facecolor("w")

                xlabel("Time step", ax=axs)
                ylabel("Energy", ax=axs)

                title = f"LNN {N}-Pendulum Exp {ind}"
                plt.title(title)
                plt.savefig(_filename(title.replace(
                    " ", "-")+f"_{key}_{ifdrag}.png"))

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
                        ax.plot(net_force_model.sum(axis=1), "--", color="k")
                        ax.plot([], "--", c="k", label="Predicted")
                    else:
                        ax.text(0.6, 0.8, f"For particle {i}",
                                transform=ax.transAxes, color="k")
                        ax.plot(net_force_orig[:, i-1, :], lw=6, label=[r"$F_x$",
                                r"$F_y$", r"$F_z$"][:R.shape[1]], alpha=0.5)
                        ax.plot(net_force_model[:, i-1, :], "--", color="k")
                        ax.plot([], "--", c="k", label="Predicted")

                    ax.legend(loc=2, bbox_to_anchor=(1, 1),
                              labelcolor="markerfacecolor")
                    ax.set_ylabel("Net force")
                    ax.set_xlabel("Time step")
                    ax.set_title(f"{N}-Pendulum Exp {ind}")
                plt.savefig(
                    _filename(f"net_force_Exp_{ind}_{key}_{ifdrag}.png"))

        Es = Es_fn(actual_traj)
        Eshat = Es_fn(pred_traj)
        H = Es[:, -1]
        Hhat = Eshat[:, -1]

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

        title = f"LNN {N}-Pendulum Exp {ind} Lmodel"
        axs[1].set_title(title)
        title = f"LNN {N}-Pendulum Exp {ind} Lactual"
        axs[0].set_title(title)

        plt.savefig(_filename(title.replace(" ", "-")+f"_{ifdrag}.png"))

    savefile(f"lnn_error_parameter_{ifdrag}.pkl", nexp)

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
        plt.savefig(_filename(f"RelError_{key}_{ifdrag}.png"))

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
        plt.savefig(_filename(f"lnn_RelError_std_{key}_{ifdrag}.png"))

    make_plots(nexp, "Zerr",
               yl=r"$\frac{||\hat{z}-z||_2}{||\hat{z}||_2+||z||_2}$")
    make_plots(nexp, "Herr",
               yl=r"$\frac{||H(\hat{z})-H(z)||_2}{||H(\hat{z})||_2+||H(z)||_2}$")


fire.Fire(main)
