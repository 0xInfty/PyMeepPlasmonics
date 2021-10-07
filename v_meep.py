#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The `v_meep` module contains tools used to complement Meep.

Some of its most useful tools are...

verify_stability_freq_res : function
    Verifies stability via temporal resolution and resonant frequencies.
verify_stability_dim_index : function
    Verifies stability via dimensions, refractive index and Courant factor.
MeepUnitsManager : class
    Incomplete class to manage units in Meep.

It's widely based on Meep Materials Library.

@author: vall
"""

import h5py as h5
try:
    from mpi4py import MPI
except ModuleNotFoundError:
    print("Importing without module 'mpi4py'")
import meep as mp
import numpy as np
import os
import resource as res
from shutil import copy
from time import sleep, time
import v_materials as vmt
import v_save as vs
import v_utilities as vu
import v_meep_analysis as vma

sysname = vs.get_sys_name()
syshome = vs.get_sys_home()
home = vs.get_home()

tupachome = "/scratch/"

midflux_key_params = ["from_um_factor", "resolution", "courant",
                      "wlen_range", "cutoff", "nfreq",
                      "submerged_index", "surface_index", "overlap",
                      "cell_width", "pml_width", "source_center", "flux_box_size",
                      "until_after_sources",
                      "parallel", "n_processes", "n_cores", "n_nodes",
                      "split_chunks_evenly", "near2far"]

chunks_key_params = ["from_um_factor", "resolution", "courant",
                     "wlen_range", "cutoff", "nfreq",
                     "r", "material", "paper", "reference",
                     "submerged_index", "surface_index", "overlap",
                     "cell_width", "pml_width", "source_center", "flux_box_size",
                     "until_after_sources",
                     "parallel", "n_processes", "n_cores", "n_nodes",
                     "split_chunks_evenly", "near2far"]

normfield_key_params = ["from_um_factor", "resolution", "courant",
                        "submerged_index", "wlen", "surface_index", "overlap",
                        "cell_width", "pml_width", "source_center",
                        "norm_until_time", "period_line", 
                        "parallel", "n_processes", "n_cores", "n_nodes",
                        "split_chunks_evenly", "hfield"]

# %%


def verify_stability_freq_res(medium, resolution, courant=0.5, print_log=True):
    """Verifies stability via temporal resolution and resonant frequencies.

    Parameters
    ----------
    medium : mp.Medium
        The mp.Medium instance of the material.
    resolution : int
        The resolution that defines spatial discretization dx = 1/resolution 
        in Meep units.
    Courant=0.5 : float
        The Courant factor that defines temporal discretization dt = Courant*dx 
        in Meep units.
    print_log=True : bool
        Whether to print the result or not.

    Returns
    -------
    stable : bool
        True if the simulation turns out stable for that medium.        
    max_courant : float
        Maximum value of Courant factor for the FDTD method to be stable.
    """

    def log(string):
        if print_log:
            print(string)

    resonant_frequencies = [Es.frequency for Es in medium.E_susceptibilities]
    max_courant = resolution / (np.pi * np.max(resonant_frequencies))

    dt = courant/resolution
    stable = True
    error = []
    for i, f in enumerate(resonant_frequencies):
        if f >= 1 / (np.pi * dt):
            stable = False
            error.append(i)
    if stable:
        answer = "Medium should be stable according to frequency and resolution criteria.\n"
        answer += "All resonant frequencies are small enough for this resolution."
        log(answer)
    else:
        answer = [str(i) + vu.counting_sufix(i) for i in error]
        if len(error) > 1:
            answer = vu.enumerate_string(answer) + " frequencies are"
        else:
            answer = answer + " frequency is"
        log("Medium could be unstable according to frequency and resolution criteria:")
        log(f"{answer} too large.")
        log(f"Maximum Courant to be stable is {max_courant}")

    return stable, max_courant

# %%


def verify_stability_dim_index(medium, freq, ndims=3, courant=0.5, print_log=True):
    """Verifies stability via dimensions, refractive index and Courant factor.

    Parameters
    ----------
    medium : The mp.Medium instance of the material.
        The mp.Medium instance of the material.
    freq : float, array of floats
        Frequency in Meep units.
    ndims=3 : int, optional
        Number of dimensions of simulation.
    courant=0.5 : float, optional
        Courant factor that defines temporal discretization from spatial 
        discretization as dt = Courant * dx.
    print_log=True : bool
        Whether to print the result or not.

    Returns
    -------
    stable : bool
        True if the simulation turns out to be stable for that medium.
    max_courant : float
        Maximum value of Courant factor for the FDTD method to be stable.
    """

    def log(string):
        if print_log:
            print(string)

    try:
        freq = [*freq]
    except:
        freq = [freq]

    index = np.array(
        [np.sqrt(medium.epsilon(f)[0, 0]*medium.mu(f)[0, 0]) for f in freq])

    min_index = np.min(np.real(index))

    stable = (courant < min_index / np.sqrt(ndims))

    max_courant = min_index / np.sqrt(ndims)

    if stable:
        log("Simulation should be stable according to dimensions and index criteria")
    else:
        log("Simulation could be unstable according to dimensions and index criteria")
        log(f"Maximum Courant to be stable is {max_courant}")

    return stable, max_courant

# %% STABILITY CHECK


def check_stability(params):

    medium = vmt.import_medium(params["material"],
                               from_um_factor=params["from_um_factor"],
                               paper=params["paper"])
    # Importing material constants dependant on frequency from Meep Library

    stable_freq_res, max_courant_freq_res = verify_stability_freq_res(
        medium, params["resolution"], courant=params["courant"], print_log=True)

    freqs = np.linspace(
        1/max(params["wlen_range"]), 1/min(params["wlen_range"]), params["nfreq"])

    stable_dim_index = []
    max_courant_dim_index = []
    for f in freqs:
        stable, max_courant = verify_stability_dim_index(medium, f,
                                                         courant=params["courant"],
                                                         print_log=False)
        stable_dim_index.append(stable)
        max_courant_dim_index.append(max_courant)

    stable_dim_index = all(stable_dim_index)
    max_courant_dim_index = min(max_courant_dim_index)

    if stable_dim_index:
        print("Medium should be stable according to dimensions and index criteria.")
    else:
        print("Medium could be unstable according to dimensions and index criteria.")
        print(f"Maximum Courant factor should be {max_courant_dim_index}")

    stable = all([stable_freq_res, stable_dim_index])
    max_courant = min([max_courant_dim_index, max_courant_freq_res])

    return stable, max_courant

# %%


class ParallelManager:

    def __init__(self, n_cores=0, n_nodes=0):

        n_processes = mp.count_processors()

        parallel_specs = np.array([n_processes, n_cores, n_nodes], dtype=int)

        max_index = np.argmax(parallel_specs)
        for index, item in enumerate(parallel_specs):
            if item == 0:
                parallel_specs[index] = 1

        parallel_specs[0:max_index] = np.full(parallel_specs[0:max_index].shape,
                                              max(parallel_specs))

        n_processes, n_cores, n_nodes = parallel_specs
        parallel = n_processes > 1

        if parallel and n_nodes == 1 and n_cores == 1:
            n_cores = n_processes

        self._n_processes = n_processes
        self._n_cores = n_cores
        self._n_nodes = n_nodes
        self._parallel = parallel

    @property
    def n_processes(self):
        return self._n_processes

    @property
    def n_cores(self):
        return self._n_cores

    @property
    def n_nodes(self):
        return self._n_nodes

    @property
    def parallel(self):
        return self._parallel

    @property
    def specs(self):
        return self.n_processes, self.n_cores, self.n_nodes

    @n_processes.setter
    @n_cores.setter
    @n_nodes.setter
    @parallel.setter
    @specs.setter
    def negator(self):
        raise AttributeError("This attribute cannot be changed this way!")

    def assign(self, process_number):

        if self.parallel and self.n_processes > 1:
            if process_number == 0:
                return mp.am_master()
            else:
                return mp.my_rank() == process_number
        else:
            return True

    def log(self, string):

        if self.assign(0): print(string)

    def hdf_file(self, filename, mode="r"):

        if self.parallel:
            f = h5.File(filename, mode, driver='mpio', comm=MPI.COMM_WORLD)
        else:
            f = h5.File(filename, mode)

        return f

# %%


def parallel_manager(process_total_number, parallel):

    def parallel_assign(process_number):

        if parallel and process_total_number > 1:
            if process_number == 0:
                return mp.am_master()
            else:
                return mp.my_rank() == process_number
        else:
            return True

    def parallel_log(string):

        if parallel_assign(0):
            print(string)
        return

    return parallel_assign, parallel_log

# %%


def parallel_hdf_file(filename, mode, parallel):

    if parallel:
        f = h5.File(filename, mode, driver='mpio', comm=MPI.COMM_WORLD)
    else:
        f = h5.File(filename, mode)

    return f

# %%


def ram_manager():

    used_ram = []
    swapped_ram = []

    def measure_ram():
        ram = res.getrusage(res.RUSAGE_THREAD).ru_maxrss  # / (1024**2)
        swap = res.getrusage(res.RUSAGE_THREAD).ru_nswap
        used_ram.append(ram)
        swapped_ram.append(swap)

    return used_ram, swapped_ram, measure_ram

# %%


class RAManager:

    def __init__(self):

        self._used_ram = []
        self._swapped_ram = []

    @property
    def used_ram(self):
        return self._used_ram

    @property
    def swapped_ram(self):
        return self._swapped_ram

    @used_ram.setter
    @swapped_ram.setter
    def negator(self):
        raise AttributeError("This attribute cannot be changed this way!")

    def measure(self):

        ram = res.getrusage(res.RUSAGE_THREAD).ru_maxrss  # / (1024**2)
        swap = res.getrusage(res.RUSAGE_THREAD).ru_nswap
        self._used_ram.append(ram)
        self._swapped_ram.append(swap)

    def reset(self):

        self._used_ram = []
        self._swapped_ram = []

# %%


class TimeManager:

    def __init__(self):

        self._elapsed_time = []
        self._instant = None

    @property
    def elapsed_time(self):
        return self._elapsed_time

    @elapsed_time.setter
    def negator(self):
        raise AttributeError("This attribute cannot be changed this way!")

    def start_measure(self):

        self._instant = time()

    def end_measure(self):

        new_instant = time()
        try:
            self._elapsed_time.append(new_instant - self._instant)
        except TypeError:
            print("Must start measurement first!")
        self._instant = None

    def reset(self):
        self._elapsed_time = []

# %%


class ResourcesMonitor:

    def __init__(self):

        self._elapsed_time = []
        self._used_ram = []
        self._swapped_ram = []

    @property
    def elapsed_time(self):
        return self._elapsed_time

    @property
    def used_ram(self):
        return self._used_ram

    @property
    def swapped_ram(self):
        return self._swapped_ram

    @elapsed_time.setter
    @used_ram.setter
    @swapped_ram.setter
    def negator(self):
        raise AttributeError("This attribute cannot be changed this way!")

    def start_measure_time(self):

        self._instant = time()

    def end_measure_time(self):

        new_instant = time()
        try:
            self._elapsed_time.append(new_instant - self._instant)
        except TypeError:
            print("Must start measurement first!")
        self._instant = None

    def measure_ram(self):

        ram = res.getrusage(res.RUSAGE_THREAD).ru_maxrss  # / (1024**2)
        swap = res.getrusage(res.RUSAGE_THREAD).ru_nswap
        self._used_ram.append(ram)
        self._swapped_ram.append(swap)

    def save(self, filename, parameters={}):

        n_processes = mp.count_processors()
        parallel = (n_processes > 1)

        if mp.am_master():
            print(f"np={n_processes}")
            print(f"parallel={parallel}")
            print(f"filename={filename}")
            print(f"parameters={parameters}")

        if os.path.isfile(filename) and mp.am_master():
            os.remove(filename)
            print("Removed file")

        f = parallel_hdf_file(filename, "w", parallel)
        print("Opened file with parallel={parallel}")

        if parallel:
            current_process = mp.my_rank()
            print(f"Current rank: {current_process}")
            f.create_dataset(
                "RAM", (len(self.used_ram), n_processes), dtype="int")
            f["RAM"][:, current_process] = self.used_ram
            print("Saved RAM")
            f.create_dataset(
                "SWAP", (len(self.used_ram), n_processes), dtype="int")
            f["SWAP"][:, current_process] = self.swapped_ram
            print("Saved SWAP")
        else:
            f.create_dataset("RAM", data=self.used_ram, dtype="int")
            print("Saved RAM")
            f.create_dataset("SWAP", data=self.swapped_ram, dtype="int")
            print("Saved SWAP")

        for key in parameters.keys():
            f["RAM"].attrs[key] = parameters[key]
        for key in parameters.keys():
            f["SWAP"].attrs[key] = parameters[key]
        print("Saved parameters in RAM and SWAP")

        f.close()
        print("Closed file")

        if mp.am_master():

            f = parallel_hdf_file(filename, "r+", False)
            print("Opened file just with master")

            f.create_dataset("ElapsedTime", data=self.elapsed_time)
            print("Saved elapsed time")

            for key in parameters.keys():
                f["ElapsedTime"].attrs[key] = parameters[key]
            print("Saved parameters in elapsed time")

            f.close()
            print("Closed file")

        return

    def load(self, filename):

        n_processes = mp.count_processors()
        parallel = n_processes > 1

        f = parallel_hdf_file(filename, "r", parallel)

        if parallel:
            current_process = mp.my_rank()
            self._used_ram = list(f["RAM"][:, current_process])
            self._swapped_ram = list(f["SWAP"][:, current_process])
        else:
            self._used_ram = list(f["RAM"])
            self._swapped_ram = list(f["SWAP"])

        self._elapsed_time = list(f["ElapsedTime"])

        f.close()
        del f

    def reset(self):

        self._elapsed_time = []
        self._used_ram = []
        self._swapped_ram = []

# %%

class SavingAssistant:

    def __init__(self, series=None, folder=None):

        if series is None:
            self._series = "Test"
        else:
            self._series = series
        if folder is None:
            self._folder = "Test"
        else:
            self._folder = folder

        self._home = vs.get_home()
        self._syshome = vs.get_sys_home()
        self._sysname = vs.get_sys_name()

        self._path = os.path.join(self._home, self._folder, self._series)
        if not os.path.isdir(self._path) and mp.am_master():
            os.makedirs(self._path)

    @property
    def series(self):
        return self._series

    @series.setter
    def series(self, value):
        self._series = value
        self._clear_last()
        self._path = os.path.join(self.home, self.folder, self.series)
        self._open_new()
        return

    @property
    def folder(self):
        return self._folder

    @folder.setter
    def folder(self, value):
        self._folder = value
        self._clear_last()
        self._path = os.path.join(self.home, self.folder, self.series)
        self._open_new()
        return

    @property
    def path(self):
        return self._path

    @property
    def home(self):
        return self._home

    @property
    def syshome(self):
        return self._syshome

    @property
    def sysname(self):
        return self._sysname

    @path.setter
    @home.setter
    @syshome.setter
    @sysname.setter
    def negator(self, value):
        raise AttributeError("Cannot set this attribute")

    def go_folder(self):

        os.chdir(self.path)

    def go_home(self):

        os.chdir(self.home)

    def go_syshome(self):

        os.chdir(self.syshome)

    def file(self, filename):
        
        if not os.path.isdir(self.path):
            self._open_new()
        return os.path.join(self.path, filename)

    def _clear_last(self):

        if mp.am_master():
            if os.path.isdir(self.path) and not os.listdir(self.path):
                os.rmdir(self.path)
                print("Removed dir")
        return

    def _open_new(self):

        if not os.path.isdir(self.path) and mp.am_master():
            os.makedirs(self.path)
            print("New dir")
        return


# %%

def save_midflux(sim, box_x1, box_x2, box_y1, box_y2, box_z1, box_z2,
                 near2far_box, params, path):

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()

    n_processes = mp.count_processors()
    parallel = n_processes > 1
    parallel_assign = parallel_manager(n_processes, parallel)[0]

    near2far = params["near2far"]

    dir_file = os.path.join(home, "FluxData/FluxDataDirectory.txt")
    dir_backup = os.path.join(home, f"FluxData/FluxDataDir{sysname}Backup.txt")

    if mpi_rank == 0:
        new_flux_path = vs.datetime_dir(os.path.join(home, "FluxData/MidFlux"),
                                        strftime="%Y%m%d%H%M%S")
        broadcasted_data = {'path': new_flux_path}
        os.makedirs(new_flux_path)
    else:
        broadcasted_data = None
    broadcasted_data = comm.bcast(broadcasted_data, root=0)
    new_flux_path = broadcasted_data["path"]

    filename_prefix = sim.filename_prefix
    sim.filename_prefix = "MidFlux"
    if sysname != "TC":
        os.chdir(new_flux_path)
    else:
        os.chdir(tupachome)
    sim.save_flux("X1", box_x1)
    sim.save_flux("X2", box_x2)
    sim.save_flux("Y1", box_y1)
    sim.save_flux("Y2", box_y2)
    sim.save_flux("Z1", box_z1)
    sim.save_flux("Z2", box_z2)
    if near2far:
        sim.save_near2far("Near2Far", near2far_box)
    os.chdir(syshome)
    sim.filename_prefix = filename_prefix

    database = vs.retrieve_footer(dir_file)
    if parallel_assign(1):
        vs.savetxt(dir_backup, np.array([]), footer=database, overwrite=True)

    database["flux_path"].append(os.path.split(new_flux_path)[-1])
    database["path"].append(path)
    for key in midflux_key_params:
        try:
            if isinstance(params[key], np.ndarray):
                database[key].append(list(params[key]))
            else:
                database[key].append(params[key])
        except:
            raise ValueError(f"Missing key parameter: {key}")

    if parallel_assign(0):
        vs.savetxt(dir_file, np.array([]), footer=database, overwrite=True)
        if sysname == "TC":
            for field in ["X1", "X2", "Y1", "Y2", "Z1", "Z2"]:
                copy(os.path.join(tupachome, f"Midfield-{field}.h5"), 
                     os.path.join(new_flux_path, f"Midfield-{field}.h5"))
                os.remove(os.path.join(tupachome, f"Midfield-{field}.h5"))

    return new_flux_path

# %%


def check_midflux(params):

    dir_file = os.path.join(home, "FluxData/FluxDataDirectory.txt")

    database = vs.retrieve_footer(dir_file)

    try:
        database_array = []
        for key in midflux_key_params:
            if key in params.keys():
                if isinstance(database[key][0], bool):
                    aux_data = [int(data) for data in database[key]]
                    database_array.append(aux_data)
                else:
                    try:
                        if len(list(database[key][0])) > 1:
                            for i in range(len(list(database[key][0]))):
                                aux_data = [data[i] for data in database[key]]
                                database_array.append(aux_data)
                        else:
                            database_array.append(database[key])
                    except:
                        database_array.append(database[key])
        database_array = np.array(database_array)

        desired_array = []
        for key in midflux_key_params:
            if key in params.keys():
                if isinstance(params[key], bool):
                    desired_array.append(int(params[key]))
                else:
                    try:
                        if len(list(params[key])) > 1:
                            for i in range(len(list(params[key]))):
                                desired_array.append(params[key][i])
                        else:
                            desired_array.append(params[key])
                    except:
                        desired_array.append(params[key])
        desired_array = np.array(desired_array)

        boolean_array = []
        for array in database_array.T:
            boolean_array.append(
                np.all(array - desired_array.T == np.zeros(desired_array.T.shape)))
        index = [i for i, boolean in enumerate(boolean_array) if boolean]

        if len(index) == 0:
            print("No coincidences where found at the midflux database!")
        elif len(index) == 1:
            print(
                f"You could use midflux data from '{database['path'][index[-1]]}'")
        else:
            print("More than one coincidence was found at the midflux database!")
            print(
                f"You could use midflux data from '{database['path'][index[-1]]}'")

        try:
            flux_path_list = [os.path.join(
                home, "FluxData", database['flux_path'][i]) for i in index]
        except:
            flux_path_list = []

        return flux_path_list

    except IndexError:
        print("Midflux database must be empty!")
        return []

# %%


def load_midflux(sim, box_x1, box_x2, box_y1, box_y2, box_z1, box_z2,
                 near2far_box, flux_path):

    print(f"Loading flux from '{flux_path}'")

    filename_prefix = sim.filename_prefix
    sim.filename_prefix = "MidFlux"
    os.chdir(flux_path)
    sim.load_flux("X1", box_x1)
    sim.load_flux("X2", box_x2)
    sim.load_flux("Y1", box_y1)
    sim.load_flux("Y2", box_y2)
    sim.load_flux("Z1", box_z1)
    sim.load_flux("Z2", box_z2)
    if near2far_box is not None:
        sim.load_near2far("Near2Far", near2far_box)
    os.chdir(syshome)
    sim.filename_prefix = filename_prefix

    return

# %%


def save_chunks(sim, params, path):

    n_processes = mp.count_processors()
    parallel = n_processes > 1
    parallel_assign = parallel_manager(n_processes, parallel)[0]

    dir_file = os.path.join(home, "ChunksData/ChunksDataDirectory.txt")
    dir_backup = os.path.join(
        home, f"ChunksData/ChunksDataDir{sysname}Backup.txt")
    new_chunks_path = vs.datetime_dir(os.path.join(home, "ChunksData/Chunks"),
                                      strftime="%Y%m%d%H%M%S")
    if parallel_assign(0):
        os.makedirs(new_chunks_path)
    else:
        sleep(.2)

    filename_prefix = sim.filename_prefix
    sim.filename_prefix = "Chunks"
    os.chdir(new_chunks_path)
    sim.dump_chunk_layout("Layout.h5")
    sim.dump_structure("Structure.h5")
    os.chdir(syshome)
    sim.filename_prefix = filename_prefix

    database = vs.retrieve_footer(dir_file)
    if parallel_assign(1):
        vs.savetxt(dir_backup, np.array([]), footer=database, overwrite=True)

    database["chunks_path"].append(os.path.split(new_chunks_path)[-1])
    database["path"].append(path)
    for key in chunks_key_params:
        try:
            if isinstance(params[key], np.ndarray):
                database[key].append(list(params[key]))
            else:
                database[key].append(params[key])
        except:
            raise ValueError(f"Missing key parameter: {key}")

    if parallel_assign(0):
        vs.savetxt(dir_file, np.array([]), footer=database, overwrite=True)

    return new_chunks_path

# %%


def check_chunks(params):

    dir_file = os.path.join(home, "ChunksData/ChunksDataDirectory.txt")

    database = vs.retrieve_footer(dir_file)

    try:
        database_array = []
        database_strings = {}
        for key in chunks_key_params:
            if key in params.keys():
                if isinstance(database[key][0], bool):
                    aux_data = [int(data) for data in database[key]]
                    database_array.append(aux_data)
                elif isinstance(database[key][0], str):
                    database_strings[key] = database[key]
                else:
                    try:
                        if len(list(database[key][0])) > 1:
                            for i in range(len(list(database[key][0]))):
                                aux_data = [data[i] for data in database[key]]
                                database_array.append(aux_data)
                        else:
                            database_array.append(database[key])
                    except:
                        database_array.append(database[key])
        database_array = np.array(database_array)

        desired_array = []
        desired_strings = {}
        for key in chunks_key_params:
            if key in params.keys():
                if isinstance(params[key], bool):
                    desired_array.append(int(params[key]))
                elif isinstance(params[key], str):
                    desired_strings[key] = params[key]
                else:
                    try:
                        if len(list(params[key])) > 1:
                            for i in range(len(list(params[key]))):
                                desired_array.append(params[key][i])
                        else:
                            desired_array.append(params[key])
                    except:
                        desired_array.append(params[key])
        desired_array = np.array(desired_array)

        boolean_array = []
        for array in database_array.T:
            boolean_array.append(
                np.all(array - desired_array.T == np.zeros(desired_array.T.shape)))
        index = [i for i, boolean in enumerate(boolean_array) if boolean]

        for key, values_list in database_strings.items():
            for i, value in enumerate(values_list):
                if value == desired_strings[key]:
                    index.append(i)

        index_in_common = []
        for i in index:
            if index.count(i) == len(list(desired_strings.keys())) + 1:
                if i not in index_in_common:
                    index_in_common.append(i)

        if len(index_in_common) == 0:
            print("No coincidences where found at the chunks database!")
        elif len(index_in_common) == 1:
            print(
                f"You could use chunks data from '{database['path'][index[0]]}'")
        else:
            print("More than one coincidence was found at the chunks database!")
            print(
                f"You could use chunks data from '{database['path'][index[0]]}'")

        try:
            chunks_path_list = [os.path.join(
                home, "ChunksData", database['chunks_path'][i]) for i in index_in_common]
        except:
            chunks_path_list = []

        return chunks_path_list

    except IndexError:
        print("Chunks database must be empty!")
        return []

# %%

def save_normfield(params, path):

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()

    n_processes = mp.count_processors()
    parallel = n_processes > 1
    parallel_assign = parallel_manager(n_processes, parallel)[0]

    dir_file = os.path.join(home, "FieldData/FieldDataDirectory.txt")
    dir_backup = os.path.join(home, f"FieldData/FieldDataDir{sysname}Backup.txt")

    if mpi_rank == 0:
        new_norm_path = vs.datetime_dir(os.path.join(home, "FieldData/NormField"),
                                        strftime="%Y%m%d%H%M%S")
        broadcasted_data = {'path': new_norm_path}
        os.makedirs(new_norm_path)
    else:
        broadcasted_data = None
    broadcasted_data = comm.bcast(broadcasted_data, root=0)
    new_norm_path = broadcasted_data["path"]

    # os.path.chdir(new_norm_path)
    if parallel_assign(0):
        copy(os.path.join(path, "Field-Lines-Norm.h5"),
             os.path.join(new_norm_path, "Field-Lines-Norm.h5"))
    # os.chdir(syshome)

    database = vs.retrieve_footer(dir_file)
    if parallel_assign(1):
        vs.savetxt(dir_backup, np.array([]), footer=database, overwrite=True)

    database["norm_path"].append(os.path.split(new_norm_path)[-1])
    database["path"].append(path)
    for key in normfield_key_params:
        try:
            if isinstance(params[key], np.ndarray):
                database[key].append(list(params[key]))
            else:
                database[key].append(params[key])
        except:
            raise ValueError(f"Missing key parameter: {key}")

    if parallel_assign(0):
        vs.savetxt(dir_file, np.array([]), footer=database, overwrite=True)

    return new_norm_path

# %%

def check_normfield(params):

    dir_file = os.path.join(home, "FieldData/FieldDataDirectory.txt")

    database = vs.retrieve_footer(dir_file)

    try:
        database_array = []
        for key in normfield_key_params:
            if key in params.keys():
                if isinstance(database[key][0], bool):
                    aux_data = [int(data) for data in database[key]]
                    database_array.append(aux_data)
                else:
                    try:
                        if len(list(database[key][0])) > 1:
                            for i in range(len(list(database[key][0]))):
                                aux_data = [data[i] for data in database[key]]
                                database_array.append(aux_data)
                        else:
                            database_array.append(database[key])
                    except:
                        database_array.append(database[key])
        database_array = np.array(database_array)

        desired_array = []
        for key in normfield_key_params:
            if key in params.keys():
                if isinstance(params[key], bool):
                    desired_array.append(int(params[key]))
                else:
                    try:
                        if len(list(params[key])) > 1:
                            for i in range(len(list(params[key]))):
                                desired_array.append(params[key][i])
                        else:
                            desired_array.append(params[key])
                    except:
                        desired_array.append(params[key])
        desired_array = np.array(desired_array)

        boolean_array = []
        for array in database_array.T:
            boolean_array.append(
                np.all(array - desired_array.T == np.zeros(desired_array.T.shape)))
        index = [i for i, boolean in enumerate(boolean_array) if boolean]

        if len(index) == 0:
            print("No coincidences where found at the normfield database!")
        elif len(index) == 1:
            print(
                f"You could use normfield data from '{database['path'][index[-1]]}'")
        else:
            print("More than one coincidence was found at the normfield database!")
            print(
                f"You could use normfield data from '{database['path'][index[-1]]}'")

        try:
            norm_path_list = [os.path.join(
                home, "FieldData", database['norm_path'][i]) for i in index]
        except:
            norm_path_list = []

        return norm_path_list

    except IndexError:
        print("Normfield database must be empty!")
        return []

# %%

def load_normfield(norm_path):

    print(f"Loading field from '{norm_path}'")
    
    n_processes = mp.count_processors()
    parallel = n_processes > 1
    parallel_assign = parallel_manager(n_processes, parallel)[0]
    
    norm_file = parallel_hdf_file(os.path.join(norm_path, "Field-Lines-Norm.h5"), 
                                  "r", parallel)
    
    results_line_norm = norm_file["Ez"]
    source_center = norm_file["Ez"].attrs["source_center"]
    
    t_line_norm = np.asarray(norm_file["T"])
    x_line_norm = np.asarray(norm_file["X"])
    
    t_line_norm_index = vma.def_index_function(t_line_norm)
    x_line_norm_index = vma.def_index_function(x_line_norm)
    
    source_results = vma.get_source_from_line(results_line_norm, x_line_norm_index, source_center)
    norm_period = vma.get_period_from_source(source_results, t_line_norm)[-1]
    norm_amplitude = vma.get_amplitude_from_source(source_results)[-1]
    
    norm_file.close()
            
    return norm_amplitude, norm_period

# %%


def recognize_component(component):

    components_dict = {"Ex": mp.Ex,
                       "Ey": mp.Ey,
                       "Ez": mp.Ez,
                       "Hx": mp.Hx,
                       "Hy": mp.Hy,
                       "Hz": mp.Hz}
    components_keys = list(components_dict.keys())
    components_values = list(components_dict.values())

    if isinstance(component, str):
        try:
            return components_dict[vu.camel(component)]
        except:
            raise ValueError("Unrecognized component")

    elif isinstance(component, int):
        try:
            return components_keys[components_values.index(component)]
        except:
            raise ValueError("Unrecognized component")

    else:
        raise ValueError("Unrecognized format for component")

# %%


def recognize_direction(direction):

    direction_dict = {"X": mp.X,
                      "Y": mp.Y,
                      "Z": mp.Z}
    direction_vect_dict = {"X": mp.Vector3(1, 0, 0),
                           "Y": mp.Vector3(0, 1, 0),
                           "Z": mp.Vector3(0, 0, 1)}
    direction_keys = list(direction_dict.keys())
    direction_values = list(direction_dict.values())
    direction_vect_values = list(direction_vect_dict.values())

    if isinstance(direction, mp.Vector3):
        try:
            return direction_keys[direction_vect_values.index(direction)]
        except:
            return direction

    elif isinstance(direction, str):
        try:
            return direction_dict[vu.camel(direction)]
        except:
            raise ValueError("Unrecognized direction")

    elif isinstance(direction, int):
        try:
            return direction_keys[direction_values.index(direction)]
        except:
            raise ValueError("Unrecognized direction")

    else:
        raise ValueError("Unrecognized format for direction")

# %%


class Line(mp.Volume):

    """A Meep Volume subclass that holds a line instead of a whole volume"""

    def __init__(self, center=mp.Vector3(), size=mp.Vector3(),
                 is_cylindrical=False, vertices=[]):

        super().__init__(center=center, size=size, dims=1,
                         is_cylindrical=is_cylindrical, vertices=vertices)

        nvertices = len(self.get_vertices())
        if nvertices > 2:
            raise TypeError(f"Must have 2 vertices and not {nvertices}")

# %%


class Plane(mp.Volume):

    """A Meep Volume subclass that holds a line instead of a whole volume"""

    def __init__(self, center=mp.Vector3(), size=mp.Vector3(),
                 is_cylindrical=False, vertices=[]):

        super().__init__(center=center, size=size, dims=1,
                         is_cylindrical=is_cylindrical, vertices=vertices)

        nvertices = len(self.get_vertices())
        if nvertices < 3 or nvertices > 4:
            raise TypeError(f"Must have 3 or 4 vertices and not {nvertices}")

# %%


class SimpleUnitsConverter:

    def __init__(self, from_um_factor):
        self.from_um_factor = from_um_factor

    def to_nm(self, mp_length):
        return mp_length * (1e3 * self.from_um_factor)

    def from_nm(self, nm_length):
        return nm_length / (1e3 * self.from_um_factor)

# %%


class MeepUnitsManager:
    """Depricated class to manage units in Meep"""

    def __init__(self, from_um_factor=1):

        self._from_um_factor = from_um_factor
        self._a = from_um_factor * 1e-6  # Meep length unit [m]

        self.constants = vu.DottableWrapper(**dict(
            c=299792458,  # Speed of light in vacuum c [m/s]
            e=1.6021892 * 10e-19,  # Electron charge e [C]
            me=9.109534 * 10e-31,  # Electron rest mass [kg]
            mp=1.6726485 * 10e-27  # Proton rest mass [kg]
        ))

        self.constants.add(**dict(
            # Vacuum Permitivity ε0 [F/m]
            epsilon0=1/(4*np.pi*self.constants.c**2) * 10e7,
            # Vacuum Permeability μ0 [H/m])
            mu0=4*np.pi * 10e-7
        ))

        self.Meep_to_SI = vu.DottableWrapper()

        self.SI_to_Meep = vu.DottableWrapper()

    @property
    def from_um_factor(self):
        """Conversion factor from um to Meep getter"""
        return self._from_um_factor

    @from_um_factor.setter
    def from_um_factor(self, value):
        """Conversion factor from um to Meep setter (also updates a)"""
        self._from_um_factor = value
        self._a = value * 1e-6  # Meep length unit [m]

    @property
    def a(self):
        """Length unit a getter"""
        return self._a

    @a.setter
    def a(self, value):
        """Length unit a setter (also updates from_um_factor)"""
        self._a = value
        self._from_um_factor = value * 1e6

    def len_Meep_to_SI(self, len_Meep):
        """Converts Meep length to SI units [m]"""

        return self.a * len_Meep

    def len_SI_to_Meep(self, len_SI):
        """Converts SI length [m] to Meep units"""

        return len_SI / self.a

    def time_Meep_to_SI(self, time_Meep):
        """Converts Meep time to SI units [s]"""

        return time_Meep * self.a / self.constants.c

    def time_SI_to_Meep(self, time_SI):
        """Converts SI time [s] to Meep units"""

        return time_SI * self.constants.c / self.a

    def vel_Meep_to_SI(self, vel_Meep):
        """Converts Meep velocity to SI units [m/s]"""

        return vel_Meep * self.constants.c

    def vel_SI_to_Meep(self, vel_SI):
        """Converts SI velocity [m/s] to Meep units"""

        return vel_SI / self.constants.c

    def freq_Meep_to_SI(self, freq_Meep):
        """Converts Meep frequency to SI units [Hz]"""

        return freq_Meep * self.constants.c / self.a

    def freq_SI_to_Meep(self, freq_SI):
        """Converts SI frequency [Hz] to Meep units"""

        return freq_SI * self.a / self.constants.c
