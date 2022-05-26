# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Tuple, TypeVar, Union

import jax.numpy as np
from jax import random
from jax_md import dataclasses, interpolate, quantity, simulate, space, util

static_cast = util.static_cast
# Types
Array = util.Array
f32 = util.f32
f64 = util.f64
ShiftFn = space.ShiftFn
T = TypeVar('T')
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Simulator = Tuple[InitFn, ApplyFn]
NVEState = simulate.NVEState

# NVEState2 = simulate.NVEState2

Schedule = Union[Callable[..., float], float]

# pylint: disable=invalid-nameNVEState


class NVEStates():
    def __init__(self, states):
        self.position = states.position
        self.velocity = states.velocity
        self.force = states.force
        self.mass = states.mass
        self.index = 0

    def __len__(self):
        return len(self.position)

    def __getitem__(self, key):
        if isinstance(key, int):
            return NVEState(self.position[key], self.velocity[key],
                            self.force[key], self.mass[key])
        else:
            return NVEState(self.position[key],
                            self.velocity[key],
                            self.force[key],
                            self.mass[key])

    def __iter__(self,):
        return (self.__getitem__(i) for i in range(len(self)))

# class NVEStates2():
#     def __init__(self, states):
#         self.position = states.position
#         self.velocity = states.velocity
#         # self.force = states.force
#         self.mass = states.mass
#         self.index = 0

#     def __len__(self):
#         return len(self.position)

#     def __getitem__(self, key):
#         if isinstance(key, int):
#             return NVEState2(self.position[key], self.velocity[key], self.mass[key])
#         else:
#             return NVEState2(self.position[key], self.velocity[key], self.mass[key])

#     def __iter__(self,):
#         return (self.__getitem__(i) for i in range(len(self)))

def nve(energy_or_force_fn: Callable[..., Array],
        shift_fn: ShiftFn,
        dt: float) -> Simulator:
    """Simulates a system in the NVE ensemble.
    Samples from the microcanonical ensemble in which the number of particles
    (N), the system volume (V), and the energy (E) are held constant. We use a
    standard velocity verlet integration scheme.
    Args:
      energy_or_force: A function that produces either an energy or a force from
        a set of particle positions specified as an ndarray of shape
        [n, spatial_dimension].
      shift_fn: A function that displaces positions, R, by an amount dR. Both R
        and dR should be ndarrays of shape [n, spatial_dimension].
      dt: Floating point number specifying the timescale (step size) of the
        simulation.
      quant: Either a quantity.Energy or a quantity.Force specifying whether
        energy_or_force is an energy or force respectively.
    Returns:
      See above.
    """
    force_fn = energy_or_force_fn

    dt_2 = 0.5 * dt ** 2

    def init_fun(R: Array,
                 V: Array,
                 mass=f32(1.0),
                 **kwargs) -> NVEState:
        mass = quantity.canonicalize_mass(mass)
        return NVEState(R, V, force_fn(R, V, **kwargs), mass)

    def apply_fun(state: NVEState, **kwargs) -> NVEState:
        R, V, F, mass = dataclasses.astuple(state)
        A = F / mass
        dR = V * dt + A * dt_2
        R, V = shift_fn(R, dR, V)
        F = force_fn(R, V, **kwargs)
        A_prime = F / mass
        V = V + f32(0.5) * (A + A_prime) * dt
        return NVEState(R, V, F, mass)

    return init_fun, apply_fun

def nve4(energy_or_force_fn: Callable[..., Array],
        shift_fn: ShiftFn,
        dt: float) -> Simulator:
    """Simulates a system in the NVE ensemble.
    Samples from the microcanonical ensemble in which the number of particles
    (N), the system volume (V), and the energy (E) are held constant. We use a
    standard velocity verlet integration scheme.
    Args:
      energy_or_force: A function that produces either an energy or a force from
        a set of particle positions specified as an ndarray of shape
        [n, spatial_dimension].
      shift_fn: A function that displaces positions, R, by an amount dR. Both R
        and dR should be ndarrays of shape [n, spatial_dimension].
      dt: Floating point number specifying the timescale (step size) of the
        simulation.
      quant: Either a quantity.Energy or a quantity.Force specifying whether
        energy_or_force is an energy or force respectively.
    Returns:
      See above.
    """
    force_fn = energy_or_force_fn

    dt_2 = 0.5 * dt ** 2

    def init_fun(R: Array,
                 V: Array,
                 mass=f32(1.0),
                 **kwargs) -> NVEState:
        mass = quantity.canonicalize_mass(mass)
        return NVEState(R, V, force_fn(R, V, **kwargs), mass)

    def apply_fun(state: NVEState, **kwargs) -> NVEState:
        R, V, F, mass = dataclasses.astuple(state)
        A = F / mass
        dR = V * dt + A * dt_2
        R, V = shift_fn(R, dR, V)
        F = force_fn(R, V, **kwargs)
        A_prime = F / mass
        V = V + f32(0.5) * (A + A_prime) * dt
        return NVEState(R, V, F, mass)
        A = F/mass
        k1 = force_fn(R, V, **kwargs)

        k2 = force_fn(R+V)

    return init_fun, apply_fun


def nve2(params, change_R_V, dt: float) -> Simulator:
    """Simulates a system in the NVE ensemble.
    Samples from the microcanonical ensemble in which the number of particles
    (N), the system volume (V), and the energy (E) are held constant. We use a
    standard velocity verlet integration scheme.
    Args:
      energy_or_force: A function that produces either an energy or a force from
        a set of particle positions specified as an ndarray of shape
        [n, spatial_dimension].
      shift_fn: A function that displaces positions, R, by an amount dR. Both R
        and dR should be ndarrays of shape [n, spatial_dimension].
      dt: Floating point number specifying the timescale (step size) of the
        simulation.
      quant: Either a quantity.Energy or a quantity.Force specifying whether
        energy_or_force is an energy or force respectively.
    Returns:
      See above.
    """

    def init_fun(R: Array, V: Array, mass=f32(1.0), **kwargs) -> NVEState:
        mass = quantity.canonicalize_mass(mass)
        return NVEState(R, V, V, mass)

    def apply_fun(state: NVEState, **kwargs) -> NVEState:
        R, V, F, mass = dataclasses.astuple(state)
        change = change_R_V(R, V, params)
        #print(change.shape)
        change_  = np.split(change,2,axis=1)
        dR = change_[0]
        dV = change_[1]
        return NVEState(R + dR, V + dV, V, mass)

    return init_fun, apply_fun

def nve3(params, change_Acc, dt: float) -> Simulator:
    """Simulates a system in the NVE ensemble.
    Samples from the microcanonical ensemble in which the number of particles
    (N), the system volume (V), and the energy (E) are held constant. We use a
    standard velocity verlet integration scheme.
    Args:
      energy_or_force: A function that produces either an energy or a force from
        a set of particle positions specified as an ndarray of shape
        [n, spatial_dimension].
      shift_fn: A function that displaces positions, R, by an amount dR. Both R
        and dR should be ndarrays of shape [n, spatial_dimension].
      dt: Floating point number specifying the timescale (step size) of the
        simulation.
      quant: Either a quantity.Energy or a quantity.Force specifying whether
        energy_or_force is an energy or force respectively.
    Returns:
      See above.
    """
    dt_2 = 0.5 * dt ** 2

    def init_fun(R: Array, V: Array, mass=f32(1.0), **kwargs) -> NVEState:
        mass = quantity.canonicalize_mass(mass)
        return NVEState(R, V, change_Acc(R,V,params), mass)

    def apply_fun(state: NVEState, **kwargs) -> NVEState:
        R, V, F, mass = dataclasses.astuple(state)
        A = F / mass
        dR = V * dt + A * dt_2
        R = R + dR
        #R, V = shift_fn(R, dR, V)
        F = change_Acc(R, V, params)
        A_prime = F / mass
        V = V + f32(0.5) * (A + A_prime) * dt
        return NVEState(R, V, F, mass)
        # R, V, F, mass = dataclasses.astuple(state)
        # A = F/mass
        # A_prime = change_Acc(R, V, params)
        # R = R + V * dt + f32(0.5) * A * dt * dt
        # V = V + f32(0.5) * (A + A_prime) * dt
        # return NVEState(R, V, A_prime * mass, mass)

    return init_fun, apply_fun
