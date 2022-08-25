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
import time

from psystems.nsprings import (chain, edge_order, get_connections,
                               get_fully_connected_senders_and_receivers,
                               get_fully_edge_order, get_init)

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


def main(N=5, dt=1.0e-3, useN=5, withdata=None, datapoints=100, mpass=1, grid=False, stride=100, ifdrag=0, seed=42, rname=0, saveovito=1, trainm=1, runs=100, semilog=1, maxtraj=100, plotthings=False, redo=0, ifDataEfficiency = 0, if_noisy_data=0):
    if (ifDataEfficiency == 1):
        data_points = int(sys.argv[1])
        batch_size = int(data_points/100)

    if useN is None:
        useN = N

    print("Configs: ")
    pprint(dt, stride, ifdrag,
           namespace=locals())

    PSYS = f"{N}-Spring"
    TAG = f"hgn"
    
    if (ifDataEfficiency == 1):
        out_dir = f"../data-efficiency"
    elif (if_noisy_data == 1):
        out_dir = f"../noisy_data"
    else:
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
        #     "0" if (tag == "data") or (withdata == None) else f"{withdata}")
        rstring  = "0" if (tag != "data" ) else "2"
        if (ifDataEfficiency == 1):
            rstring = "2_" + str(data_points)

        if (tag == "data"):
            filename_prefix = f"../results/{PSYS}-{tag}/{2}/"
        else:
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
    z_out, zdot_out = dataset_states[0]
    xout, pout = jnp.split(z_out, 2, axis=1)

    R = xout[0]
    V = pout[0]

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

    senders = jnp.array(senders)
    receivers = jnp.array(receivers)

    # R = model_states.position[0]
    # V = model_states.velocity[0]

    print(
        f"Total number of training data points: {len(dataset_states)}x{z_out.shape}")

    N, dim = xout.shape[-2:]
    species = jnp.zeros(N, dtype=int)
    masses = jnp.ones(N)

    ################################################
    ################## SYSTEM ######################
    ################################################

    # parameters = [[dict(length=1.0)]]
    # pot_energy_orig = map_parameters(
    #     lnn.SPRING, displacement, species, parameters)
    def pot_energy_orig(x):
        dr = jnp.square(x[senders, :] - x[receivers, :]).sum(axis=1)
        return jax.vmap(partial(src.hamiltonian.SPRING, stiffness=1.0, length=1.0))(dr).sum()

    kin_energy = partial(src.hamiltonian._T, mass=masses)

    def Hactual(x, p, params):
        return kin_energy(p) + pot_energy_orig(x)

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

    # if trainm:
    #     print("kinetic energy: learnable")

    #     def L_energy_fn(params, graph):
    #         g, V, T = cal_graph(params, graph, mpass=mpass, eorder=eorder,
    #                             useT=True, useonlyedge=True)
    #         return T - V

    # else:
    #     print("kinetic energy: 0.5mv^2")

    #     kin_energy = partial(lnn._T, mass=masses)

    #     def L_energy_fn(params, graph):
    #         g, V, T = cal_graph(params, graph, mpass=mpass, eorder=eorder,
    #                             useT=True, useonlyedge=True)
    #         return kin_energy(graph.nodes["velocity"]) - V

    def dist(*args):
        disp = displacement(*args)
        return jnp.sqrt(jnp.square(disp).sum())

    R = jnp.array(R)
    V = jnp.array(V)
    species = jnp.array(species).reshape(-1, 1)

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
        acc = fgn.cal_lgn(params, graph, mpass=1)
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

    def Hmodel(x, v, params): return apply_fn(x, v, params["L"])

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

    # acceleration_fn_model = accelerationFull(N, dim,
    #                                          lagrangian=Lmodel,
    #                                          constraints=None,
    #                                          non_conservative_forces=drag)

    zdot_model, lamda_force_model = get_zdot_lambda(
        N, dim, hamiltonian=Hmodel, drag=drag, constraints=None)

    def zdot_model_func(z, t, params):
        x, p = jnp.split(z, 2)
        return zdot_model(x, p, params)

    params = loadfile(f"trained_model_low.dil", trained=useN)[0]

    
    sim_model = get_forward_sim(
        params=params, zdot_func=zdot_model_func, runs=runs)

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
        "Perr": [],
    }

    trajectories = []

    sim_orig2 = get_forward_sim(params=None, zdot_func=zdot_func, runs=runs)

    skip = 0

    t = 0.0

    for ind in range(maxtraj):
        if ind > maxtraj+skip:
            break

        _ind = ind*runs

        print(f"Simulating trajectory {ind}/{maxtraj} ...")

        # R = full_traj[_ind].position
        # V = full_traj[_ind].velocity
        # start_ = _ind+1
        # stop_ = start_+runs

        z_out, _ = dataset_states[ind]
        xout, pout = jnp.split(z_out, 2, axis=1)

        R = xout[0]
        V = pout[0]

        try:
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

            if saveovito:
                save_ovito(f"pred_{ind}.data", [
                    state for state in NVEStates(pred_traj)], lattice="")
                save_ovito(f"actual_{ind}.data", [
                    state for state in NVEStates(actual_traj)], lattice="")

            trajectories += [(actual_traj, pred_traj)]
            savefile("trajectories.pkl", trajectories)

            if plotthings:
                #raise Warning("Cannot calculate energy in FGN")
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

                    title = f"HGN {N}-Spring Exp {ind}"
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
            Eshat = Es_fn(pred_traj)
            H = Es[:, -1]
            Hhat = Eshat[:, -1]

            nexp["Herr"] += [RelErr(H, Hhat)+1e-30]
            nexp["E"] += [Es, Eshat]
            
            nexp["z_pred"] += [pred_traj.position]
            nexp["z_actual"] += [actual_traj.position]
            nexp["Zerr"] += [RelErr(actual_traj.position,
                                    pred_traj.position)+1e-30]
            ac_mom = jnp.square(actual_traj.velocity.sum(1)).sum(1)
            pr_mom = jnp.square(pred_traj.velocity.sum(1)).sum(1)
            nexp["Perr"] += [jnp.absolute(ac_mom - pr_mom)]
            
            savefile(f"error_parameter.pkl", nexp)

            fig, axs = panel(1, 2, figsize=(20, 5))
            axs[0].plot(Es, label=["PE", "KE", "L", "TE"], lw=6, alpha=0.5)
            axs[1].plot(Eshat, "--", label=["PE", "KE", "L", "TE"])
            plt.legend(bbox_to_anchor=(1, 1), loc=2)
            axs[0].set_facecolor("w")

            xlabel("Time step", ax=axs[0])
            xlabel("Time step", ax=axs[1])
            ylabel("Energy", ax=axs[0])
            ylabel("Energy", ax=axs[1])

            title = f"HGN {N}-Spring Exp {ind} pred traj"
            axs[1].set_title(title)
            title = f"HGN {N}-Spring Exp {ind} actual traj"
            axs[0].set_title(title)

            plt.savefig(
                _filename(f"HGN {N}-Spring Exp {ind}".replace(" ", "-")+f"_actualH.png"))
        except:
            print("skipped")
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
               yl=r"$\frac{||z_1-z_2||_2}{||z_1||_2+||z_2||_2}$")
    make_plots(nexp, "Herr",
               yl=r"$\frac{||H(z_1)-H(z_2)||_2}{||H(z_1)||_2+||H(z_2)||_2}$")

    make_plots(nexp, "Perr",
               yl=r"$\frac{||P(z_1)-P(z_2)||_2}{||P(z_1)||_2+||P(z_2)||_2}$")

    gmean_zerr = jnp.exp( jnp.log(jnp.array(nexp["Zerr"])).mean(axis=0) )
    gmean_herr = jnp.exp( jnp.log(jnp.array(nexp["Herr"])).mean(axis=0) )
    gmean_perr = jnp.exp( jnp.log(jnp.array(nexp["Perr"])).mean(axis=0) )

    if (ifDataEfficiency == 0):
        np.savetxt(f"../{N}-spring-zerr/hgn.txt", gmean_zerr, delimiter = "\n")
        np.savetxt(f"../{N}-spring-herr/hgn.txt", gmean_herr, delimiter = "\n")
        np.savetxt(f"../{N}-spring-perr/hgn.txt", gmean_perr, delimiter = "\n")
        np.savetxt(f"../{N}-spring-simulation-time/hgn.txt", [t/maxtraj], delimiter = "\n")

# main(N = 20)
main(N = 5)


