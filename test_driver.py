#!/usr/bin/env python3

###############################################################################
#
# Test driver to compute the linear thermal expansion of a cubic crystal at
# finite temperature and pressure.
#
# The baisc idea with this driver is to create a lammps template script that will 
# be used to run NPT simulations of a 10x10x10 unit cell slab of material at 
# finite temperature, by replacing specified variables in the template. This produces 5
# seperate lammps scripts for different temperatures, which are then run simultaneously
# via multiprocessing.  The results of this are then stored in  files that specify 
# the system volume and the final positions of all atoms. The system
# volume and box parameters are used to compute the coeffient of thermal expansion (alpha),
# and the positions of the atoms are averaged and used to check that the crytstal
# structure did not change during the simulation.
#
###############################################################################
import os
import multiprocessing
import copy
import re

import numpy as np
import numpy.typing as npt
from scipy.stats import linregress
from ast import literal_eval
from collections import deque
from typing import Iterable, List, Optional, Tuple

from ase.build import bulk
from ase.io.lammpsdata import write_lammps_data
from ase.io.lammpsrun import construct_cell
from ase.io import write
from ase import Atoms
from ase.io.lammpsrun import lammps_data_to_ase_atoms
from ase.calculators.lammps import convert

from kim_python_utils.ase import CrystalGenomeTest, KIMASEError
from kim_test_utils import test_driver
from kim_test_utils.test_driver import CrystalGenomeTestDriver


###############################################################################
#
# read_params
#
# read parameters from stdin
#
# Argument:
# x: input class instance
#
# Only used if run directly, not called in test driver
#
###############################################################################


def read_params(x):

    def is_num(asking, name):
        try:
            value_str = eval(input(asking))
            print(value_str)
            value = float(value_str)
        except ValueError:
            raise ValueError(
                "Incorrect input `%s = %s'; `%s' should be a numerical value."
                % (name, value_str, name)
            )
        return value

    def is_positive(var, name):
        if var < 0.0:
            raise ValueError(
                "Incorrect input `%s = %s'; `%s' should be positive."
                % (name, str(var), name)
            )

    x.modelname = input("Please enter a KIM Model extended-ID:\n")
    print("Modelname = {}".format(x.modelname))

    x.species = input("Please enter a list of species symbol (e.g. [Si, Au, Al], etc.):\n")
    print("Species = {}".format(x.species))

    x.mass = literal_eval(input("Please enter a list of the atomic masses of the species (g/mol):\n"))
    print("Mass = {} g/mol".format(x.mass))

    msg = "Please enter the temperature (Kelvin):\n"
    x.temperature = is_num(msg, "temperature")
    is_positive(x.temperature, "temperature")
    print("Temperature = {} Kelvin".format(x.temperature))

    msg = "Please enter the hydrostatic pressure (MPa):\n"
    x.press = is_num(msg, "stress")
    print("Pressure = {} MPa".format(x.press))

    # convert unit of pressure from MPa to Bar
    x.press = x.press * 10


###############################################################################
#
# get_input_file
#
# Generate LAMMPS input files from the template input file.
#
# Arguments:
# template: name of lammps template script
# lmpinput: name string of lammps input file
# x: dictionary of runtime parameters
# T: array of temperatures
# vol_vile: filename to record final box volume in
# position_file: array of file names to save final atom positions in (same len as T)
# cell_file: array of file names to save final simulation cell info in (same len as T)
#
###############################################################################

def get_input_file(template, lmpinput, x, T, vol_file, position_file, cell_file):
    TDdirectory = os.path.dirname(os.path.realpath(__file__))
    species_list=x["stoichiometric_species"]
    species_string=""
    for element in species_list:
        species_string=species_string+element+" "
    replacements = {
        "rpls_modelname": x["modelname"],
        "rpls_species": species_string,
        "rpls_temp": str(T),
        "rpls_press": str(x["press"]),
        "rpls_vol_file": vol_file,
        "average_position_filename":position_file,
        "average_cell_filename": cell_file
    }

    with open(template, "r") as readfile, open(lmpinput, "w") as writefile:
        writefile.write(readfile.read().format(**replacements))


###############################################################################
#
# run_lammps
#
# use the specified input template to run a lammps simulation
#
# Run lammps with `lmpinput' as the input.
#
# Arguments:
# lmpinput: list of arguments passed to lammps to be replaced in the
#
###############################################################################

def run_lammps(lmpinput):
    cmd = "lammps -in " + lmpinput + " > output/screen.out"
    os.system(cmd)

###############################################################################
#
# create_jobs
# create N = len(Temp) LAMMPS input(s) and run them simultaenously with multiprocessing
#
# Arguments:
# Temp: temperature list
# volfle: list of name strings of volume file
# param: list of parameters to be replaced
# position_file: file name to save final atom positions in
# cell_file: file name to save final simulation cell info in
#
###############################################################################


def create_jobs(volfile, Temp, param, position_file, cell_file):

    # lammps input template
    TDdirectory = os.path.dirname(os.path.realpath(__file__))
    tempfile = TDdirectory + "/lammps.in.template"

    # create N = len(Temp) jobs at N = len(Temp) different temperatures
    jobs = []
    infile = []
    for i in range(len(Temp)):
        # LAMMPS input file names
        infile.append("output/lmpinfile_T" + str(Temp[i]) + ".in")
        # get input file from
        get_input_file(tempfile, infile[i], param, Temp[i], volfile[i], "output/"+position_file[i], cell_file[i])
        # record jobs
        p = multiprocessing.Process(target=run_lammps, args=(infile[i],))
        jobs.append(p)

    # submit the jobs
    for p in jobs:
        p.start()
    # wait for the jobs to complete
    for p in jobs:
        p.join()

    return jobs


###############################################################################
#
# check_lammps_log_for_wrong_structure_format
#
# return a boolean corresponding to whether the lammps simulation failed
# because of an incorrectly formatted input structure file
#
# Arguments:
# log_file: path to the relevant lammps log file
#
###############################################################################
def check_lammps_log_for_wrong_structure_format(log_file):
    wrong_format_in_structure_file=False

    try:
        with open(log_file,"r") as logfile:
            data=logfile.read()
            data=data.split("\n")
            final_line=data[-2]

            if final_line == "Last command: read_data structure_template":
                wrong_format_in_structure_file=True
    except FileNotFoundError:
        pass

    return wrong_format_in_structure_file


###############################################################################
#
#  _split_dump_file_by_frame
#  
# Split a lammps dump file containing a series of frames at different timesteps
# into a series of files each containing a single timestep's data.
#
# Likely not needed after merging with other Finite Temperature tests, right now this
# driver doesn't use kim-convergence, the convergence criteria are in the lammps script,
# just Temperature and Pressure within a tolerance of a target value. So unlike the 
# Heat Capacity driver, the lammps script runs an internal loop instead of restarting.
# This results in one long dump file, instead of several small ones, which needs to
# be split into individual timesteps for the averaging.
#
# Arguments:
# path: path to where the dump file is
# filename: name of the dump file
#
###############################################################################

def _split_dump_file_by_frame(path, filename):
    with open(os.path.join(path,filename),"r") as dumpfile:
        data=dumpfile.readlines()

    # read the dump, and split into individual frames by timestamp
    previous_frame_line=0
    frames=[]
    for i,line in enumerate(data):
        if "TIMESTEP" in line:
            frame=data[previous_frame_line:i]
            if frame!=[]:
                frames.append(frame)
            previous_frame_line=i

    # write each frame into its own seperate file
    for i, frame in enumerate(frames):
        for j, line in enumerate(frame):
            if "TIMESTEP" in line:
                timestep=int(frame[j+1])
                break
        
        new_filename=filename+"."+str(timestep)

        with open(os.path.join(path,new_filename), "w") as outfile:
            outfile.writelines(frame)


###############################################################################
#
# compute_alpha
#
# compute the linear thermal expansion tensor from the simulation results
#
# Arguments:
# volfile: list of name strings of the volume files
# Temp: temperature list
# prototype_label: aflow crystal structure designation
#
###############################################################################

def compute_alpha(volfile, Temp, prototype_label):
    a=[]
    b=[]
    c=[]
    alpha_angle=[]
    beta_angle=[]
    gamma_angle=[]

    space_group=int(prototype_label.split("_")[2])

    for fin in volfile:
        with open(fin, "r") as f:
            tmp = f.read().strip()
            if tmp == "not_converged":
                return "not_converged"
            elif "Volume keeps increasing" in tmp:
                return "unstable"
            else:
                aa,bb,cc,alph_ang,bet_ang,gam_ang = tmp.split()
                a.append(float(aa))
                b.append(float(bb))
                c.append(float(cc))
                alpha_angle.append(float(alph_ang))
                beta_angle.append(float(bet_ang))
                gamma_angle.append(float(gam_ang))
    
    # needed for all space groups
    #(aslope, ycut) = np.polyfit(Temp, a, 1)
    resulta = linregress(Temp, a)
    aslope = resulta.slope

    # if the space group is cubic, only compute alpha11
    if space_group>=195:
        alpha11=aslope
        alpha22=alpha11
        alpha33=alpha11
        alpha12=0
        alpha13=0
        alpha23=0

    # hexagona, trigonal, tetragonal space groups also compute alpha33
    elif space_group>=75 and space_group<=194:
        resultc = linregress(Temp, c)
        cslope = resultc.slope
        alpha11=aslope
        alpha33=cslope
        alpha22=alpha33
        alpha12=0
        alpha13=0
        alpha23=0

    # orthorhombic, also compute alpha22
    elif space_group>=16 and space_group<=74:
        resultb = linregress(Temp, b)
        bslope = resultb.slope
        resultc = linregress(Temp, c)
        cslope = resultc.slope
        alpha11=aslope
        alpha22=bslope
        alpha33=cslope
        alpha12=0
        alpha13=0
        alpha23=0

    # monoclinic or triclinic
    elif space_group<=15:

        resultb = linregress(Temp, b)
        bslope = resultb.slope
        resultc = linregress(Temp, c)
        cslope = resultc.slope

        result_alpha_angle = linregress(Temp,alpha_angle)
        alpha_angle_slope = result_alpha_angle.slope
        result_beta_angle = linregress(Temp,beta_angle)
        beta_angle_slope = result_beta_angle.slope

        aval=a[2]
        bval=b[2]
        cval=c[2]
        alphaval=alpha_angle[2]
        betaval=beta_angle[2]
        gammaval=gamma_angle[2]

        alpha11 = (1/aval) * aslope
        alpha22 = (1/bval) * bslope + gamma_angle_slope * (1/np.tan(np.radians(gammaval)))
        alpha33 = (1/cval) * cslope
        alpha12 = (-1/2) * ((1/aval) * aslope - (1/bval) * bslope) * (1/np.tan(np.radians(gammaval))) - (1/2) * gamma_angle_slope
        alpha13 = (-1/2) * beta_angle_slope
        alpha23 = (1/2) * ((-1/np.sin(np.radians(gammaval))) * alpha_angle_slope + beta_angle_slope * (1/np.tan(np.radians(gammaval)))) 
    
        # triclinic
        if space_group<=2:

            a=np.asarray(a)
            b=np.asarray(b)
            c=np.asarray(c)
            alpha_angle=np.asarray(alpha_angle)
            beta_angle=np.asarray(beta_angle)
            gamma_angle=np.asarray(gamma_angle)

            # calculating reciprocal lattice angle gamma_star, and its temperature derivitive
            gamma_star_array=np.arccos((np.cos(np.radians(alpha_angle))*np.cos(np.radians(beta_angle)) - np.cos(np.radians(gamma_angle)))/(np.sin(np.radians(alpha_angle))*np.sin(np.radians(beta_angle))))
            gamma_star=gamma_star_array[2]
            # (gamma_star_prime,ycut) = np.polyfit(Temp, gamma_star_array, 1)

            gamma_star_result = linregress(Temp,gamma_star_array)
            gamma_start_prime = gamma_star_result.slope

            alpha11 = (1/aval) * aslope + beta_angle_slope * (1/np.tan(np.radians(betaval)))
            alpha22 = (1/bval) * bslope + alpha_angle_slope * (1/np.tan(np.radians(alphaval))) + gamma_star_prime * (1/np.tan(gamma_star))
            alpha33 = (1/cval) * cslope
            alpha12 = (1/2) * (1/np.tan(gamma_star)) * ((1/aval) * aslope - (1/bval) * bslope - alpha_angle_slope * (1/np.tan(alphaval)) + beta_angle_slope * (1/np.tan(betaval))) + (1/2) * gamma_star_prime
            alpha13 = (1/2) * ((1/aval) * aslope - (1/cval) * cslope) * (1/np.tan(betaval)) - (1/2) * beta_angle_slope
            alpha23 = (1/2) * (((1/aval) * aslope - (1/cval) * cslope) * (1/np.tan(gamma_star)) * (1/np.tan(betaval)) + ((1/bval) * bslope - (1/cval) * cslope) * (1/(np.tan(alphaval)*np.sin(gamma_star))) - ((1/np.sin(gamma_star)) * alpha_angle_slope + beta_angle_slope * (1/np.tan(gamma_star))))

    else:
        raise RuntimeError("invalid space group in prototype label")
    
    # enforce tensor symmetries
    alpha21=alpha12
    alpha31=alpha13
    alpha32=alpha23

    alpha=np.array([[alpha11,alpha12,alpha13],
                    [alpha21,alpha22,alpha23],
                    [alpha31,alpha32,alpha33]])

    # thermal expansion coeff tensor
    return alpha


###############################################################################
#
# main function
#
###############################################################################

class TestDriver(CrystalGenomeTestDriver):
    """Right now this test driver doesn't run, because its inputs are wrong. 
    Previously, it inhereted from kim-python-utils.ase.CrystalGenomeTest like
    Ilia's BindingEnergy example, but I was in the process of changing it to 
    inherit from kim-test-utils.CrystalGenomeTestDriver like the Heat Capacity TD,
    which handles inputs somewhat differently.
    
    Probably the way to move forward is to use this driver's method to initialize
    multiple lammps simulations that differ only in temperature from a template, 
    run them simultaneously with multiprocessing, but they will use a lammps script
    more like the Heat Capacity driver's, just running at 5 temperatures instead of 2.
    Then, use the Heat Capacity driver's infrastructure to verify that the crystal 
    structure hasn't changed.
    
    Just make sure that the lammps simulations still output a version of the volume file,
    with time-averaged parameters cella, cellb, cellc and cellalpha, cellbeta, cellgamma 
    like this driver's template does so that compute_alpha has the inputs it needs.
    
    I left comments # NEEDED TO COMPUTE THERMAL EXPANSION TENSOR in the lammps script
    where computations related to this are set up."""
    def _calculate(self, model_name: str, structure_index: int, temp: float, press: float, mass: list):

        self.model_name=model_name
    
        atoms=self.atoms[structure_index]

        original_cell = atoms.get_cell() # do this instead of deepcopy
        original_atoms = copy.copy(atoms)

        atoms_new = self.atoms.copy()

        # set up slab 10 unit cells thick
        repeat=(10,10,10)
        atoms_new=atoms.repeat(repeat)

        proto=self.prototype_label

        # set up path for structure template file used to initialize lammps simulation
        TDdirectory = os.path.dirname(os.path.realpath(__file__))
        structure_file=os.path.join(TDdirectory,"structure_template")

        # bundle parameters into a dictionary
        param={"modelname":self.model_name,
               "stoichiometric_species":self.stoichiometric_species,
               "press":press,
               }
        
        # # create temperature list
        dT = 20  # temperature interval
        ntemp=5
        if temp - 2 * dT < 0.0:
            T_lowest = 0.0
        else:
            T_lowest = temp - 2 * dT
        T = [round(T_lowest + i * dT, 2) for i in range(ntemp)]

        # filename arrays for temperature and structure index
        # lmpvolfile stores the final box parameters/volume
        # avg_pos_file and avg_cell_file store atom positions for structure change detection
        lmpvolfile = ["output/vol_T" + str(T[i]) + "_"+ str(structure_index) + ".out" for i in range(ntemp)]
        avg_pos_file = ["average_positions_" + str(T[i]) + "_"+ str(structure_index) + ".dump" for i in range(ntemp)]
        avg_cell_file = ["output/average_cell_" + str(T[i]) + "_"+ str(structure_index) + ".dump" for i in range(ntemp)]
        
        # some models want atom_style="charge", others want "atomic"
        # try to run with atomic, if it fails, try charge
        try:
            write_lammps_data(structure_file,atoms_new,atom_style="atomic",masses=True)

            # run LAMMPS at N = len(T) temperatures simutaneously
            jobs = create_jobs(lmpvolfile, T, param, avg_pos_file, avg_cell_file)
        
        except FileNotFoundError as e:
            wrong_format_error=False
            for t in T:
                filename="output/lmp_T"+str(t)+".log"
                log_file=os.path.join(TDdirectory,filename)
                
                wrong_format_error=check_lammps_log_for_wrong_structure_format(log_file)

            if wrong_format_error:

                write_lammps_data(structure_file,atoms_new,atom_style="charge",masses=True)

                # run LAMMPS at N = len(T) temperatures simutaneously
                jobs = create_jobs(lmpvolfile, T, param, avg_pos_file, avg_cell_file)
            
            else :

                raise e

        # use lmpvolfile data to compute the thermal expansion tensor after the lammps simulation
        alpha = compute_alpha(lmpvolfile,T,proto)
        
        # verify that the crystal structure hasn't changed
        for outfile in avg_pos_file:

            _split_dump_file_by_frame("output",outfile)
            equilibration_time=1000
            self._compute_average_positions_from_lammps_dump("output", outfile,
                                                        "output/average_position_equilibration_over_dump.out",
                                                        skip_steps=equilibration_time)
            atoms_new.set_cell(self._get_cell(self._average_cell_over_steps("output/average_cell_253.15_0.dump",
                                                                            skip_steps=equilibration_time)))
            atoms_new.set_scaled_positions(
                self._get_positions_from_lammps_dump("output/average_position_equilibration_over_dump.out"))
            reduced_atoms = self._reduce_and_avg(atoms_new, repeat)
            # AFLOW Symmetry check
            _get_crystal_genome_designation_from_atoms_and_verify_unchanged_symmetry(
                reduced_atoms, loose_triclinic_and_monoclinic=loose_triclinic_and_monoclinic)

            final_params=test_driver.get_crystal_genome_designation_from_atoms(atoms_new)
            final_species=final_params["stoichiometric_species"]
            final_prototype_label=final_params["prototype_label"]
            test_driver.verify_unchanged_symmetry(self.stoichiometric_species,
                                                        self.prototype_label,
                                                        final_species,
                                                        final_prototype_label)
        # write final test result/error
        if isinstance(alpha,str):
            if alpha == "not_converged":
                print(
                    "Error: the temperature or pessure has not converged within the simulation "
                    "steps specified in the Test Driver. Linear thermal expansion coefficient "
                    "cannot be obtained."
                )
            elif alpha == "unstable":
                print(
                    "Error: the system may be unstable since the volume keeps increasing. "
                    "Linear thermal expansion coefficient cannot be obtained."
                )
        else:
            self._add_property_instance("thermal-expansion-coefficient-npt")
            self._add_common_crystal_genome_keys_to_current_property_instance(structure_index,write_stress=True,write_temp=True)
            self._add_key_to_current_property_instance("alpha11",alpha[0,0],"1/K")
            self._add_key_to_current_property_instance("alpha22",alpha[1,1],"1/K")
            self._add_key_to_current_property_instance("alpha33",alpha[2,2],"1/K")
            self._add_key_to_current_property_instance("alpha12",alpha[0,1],"1/K")
            self._add_key_to_current_property_instance("alpha13",alpha[0,2],"1/K")
            self._add_key_to_current_property_instance("alpha23",alpha[1,2],"1/K")
            self._add_key_to_current_property_instance("thermal-expansion-coefficient",alpha,"1/K")
            self._add_key_to_current_property_instance("temperature",temp,"K")

    # Methods borrowed from Phil's heat capacity driver below here

    @staticmethod
    def _reduce_and_avg(atoms: Atoms, repeat: Tuple[int, int, int]) -> Atoms:
        '''
        Function to reduce all atoms to the original unit cell position.
        '''
        new_atoms = atoms.copy()

        cell = new_atoms.get_cell()

        # Divide each unit vector by its number of repeats.
        # See https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element.
        cell = cell / np.array(repeat)[:, None]

        # Decrease size of cell in the atoms object.
        new_atoms.set_cell(cell)
        new_atoms.set_pbc((True, True, True))

        # Set averaging factor
        M = np.prod(repeat)

        # Wrap back the repeated atoms on top of the reference atoms in the original unit cell.
        positions = new_atoms.get_positions(wrap=True)

        number_atoms = len(new_atoms)
        original_number_atoms = number_atoms // M
        assert number_atoms == original_number_atoms * M
        positions_in_prim_cell = np.zeros((original_number_atoms, 3))

        # Start from end of the atoms because we will remove all atoms except the reference ones.
        for i in reversed(range(number_atoms)):
            if i >= original_number_atoms:
                # Get the distance to the reference atom in the original unit cell with the
                # minimum image convention.
                distance = new_atoms.get_distance(i % original_number_atoms, i,
                                                  mic=True, vector=True)
                # Get the position that has the closest distance to the reference atom in the
                # original unit cell.
                position_i = positions[i % original_number_atoms] + distance
                # Remove atom from atoms object.
                new_atoms.pop()
            else:
                # Atom was part of the original unit cell.
                position_i = positions[i]
            # Average.
            positions_in_prim_cell[i % original_number_atoms] += position_i / M

        new_atoms.set_positions(positions_in_prim_cell)

        return new_atoms

    @staticmethod
    def _plot_property_from_lammps_log(in_file_path: str, property_names: Iterable[str]) -> None:
        '''
        The function to get the value of the property with time from ***.log 
        the extracted data are stored as ***.csv and ploted as property_name.png
        data_dir --- the directory contains lammps_equilibration.log 
        property_names --- the list of properties
        '''
        def get_table(in_file):
            if not os.path.isfile(in_file):
                raise FileNotFoundError(in_file + "not found")
            elif not ".log" in in_file:
                raise FileNotFoundError("The file is not a *.log file")
            is_first_header = True
            header_flags = ["Step", "v_pe_metal", "v_temp_metal", "v_press_metal"]
            eot_flags = ["Loop", "time", "on", "procs", "for", "steps"]
            table = []
            with open(in_file, "r") as f:
                line = f.readline()
                while line:  # not EOF
                    is_header = True
                    for _s in header_flags:
                        is_header = is_header and (_s in line)
                    if is_header:
                        if is_first_header:
                            table.append(line)
                            is_first_header = False
                        content = f.readline()
                        while content:
                            is_eot = True
                            for _s in eot_flags:
                                is_eot = is_eot and (_s in content)
                            if not is_eot:
                                table.append(content)
                            else:
                                break
                            content = f.readline()
                    line = f.readline()
            return table

        def write_table(table, out_file):
            with open(out_file, "w") as f:
                for l in table:
                    f.writelines(l)

        dir_name = os.path.dirname(in_file_path)
        in_file_name = os.path.basename(in_file_path)
        out_file_path = os.path.join(dir_name, in_file_name.replace(".log", ".csv"))

        table = get_table(in_file_path)
        write_table(table, out_file_path)
        df = np.loadtxt(out_file_path, skiprows=1)

        for property_name in property_names:
            with open(out_file_path) as file:
                first_line = file.readline().strip("\n")
            property_index = first_line.split().index(property_name)
            properties = df[:, property_index]
            step = df[:, 0]
            plt.plot(step, properties)
            plt.xlabel("step")
            plt.ylabel(property_name)
            img_file = os.path.join(dir_name, in_file_name.replace(".log", "_")+property_name + ".png")
            plt.savefig(img_file, bbox_inches="tight")
            plt.close()

    @staticmethod
    def _compute_average_positions_from_lammps_dump(data_dir: str, file_str: str, output_filename: str, skip_steps: int) -> None:
        '''
        This function compute the average position over *.dump files which contains the file_str in data_dir and output it
        to data_dir/[file_str]_over_dump.out

        input:
        data_dir -- the directory contains all the data e.g average_position.dump.* files
        file_str -- the files whose names contain the file_str are considered
        output_filename -- the name of the output file
        skip_steps -- dump files with steps <= skip_steps are ignored
        '''

        def get_id_pos_dict(file_name):
            '''
            input: 
            file_name--the file_name that contains average postion data
            output:
            the dictionary contains id:position pairs e.g {1:array([x1,y1,z1]),2:array([x2,y2,z2])}
            for the averaged positions over files
            '''
            id_pos_dict = {}
            header4N = ["NUMBER OF ATOMS"]
            header4pos = ["id", "f_avePos[1]", "f_avePos[2]", "f_avePos[3]"]
            is_table_started = False
            is_natom_read = False
            with open(file_name, "r") as f:
                line = f.readline()
                count_content_line = 0
                N = 0
                while line:
                    if not is_natom_read:
                        is_natom_read = np.all([flag in line for flag in header4N])
                        if is_natom_read:
                            line = f.readline()
                            N = int(line)
                    if not is_table_started:
                        contain_flags = np.all([flag in line for flag in header4pos])
                        is_table_started = contain_flags
                    else:
                        count_content_line += 1
                        words = line.split()
                        id = int(words[0])
                        # pos = np.array([float(words[2]),float(words[3]),float(words[4])])
                        pos = np.array([float(words[2]), float(words[3]), float(words[4])])
                        id_pos_dict[id] = pos
                    if count_content_line > 0 and count_content_line >= N:
                        break
                    line = f.readline()
            if count_content_line < N:
                print("The file " + file_name +
                      " is not complete, the number of atoms is smaller than " + str(N))
            return id_pos_dict

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(data_dir + " does not exist")
        if not ".dump" in file_str:
            raise ValueError("file_str must be a string containing .dump")

        # extract and store all the data
        pos_list = []
        max_step, last_step_file = -1, ""
        for file_name in os.listdir(data_dir):
            if file_str in file_name:
                step = int(re.findall(r'\d+', file_name)[-1])
                if step <= skip_steps:
                    continue
                file_path = os.path.join(data_dir, file_name)
                id_pos_dict = get_id_pos_dict(file_path)
                id_pos = sorted(id_pos_dict.items())
                id_list = [pair[0] for pair in id_pos]
                pos_list.append([pair[1] for pair in id_pos])
                # check if this is the last step
                if step > max_step:
                    last_step_file, max_step = os.path.join(data_dir, file_name), step
        if max_step == -1 and last_step_file == "":
            raise RuntimeError("Found no files to average over.")
        pos_arr = np.array(pos_list)
        avg_pos = np.mean(pos_arr, axis=0)
        # get the lines above the table from the file of the last step
        with open(last_step_file, "r") as f:
            header4pos = ["id", "f_avePos[1]", "f_avePos[2]", "f_avePos[3]"]
            line = f.readline()
            description_str = ""
            is_table_started = False
            while line:
                description_str += line
                is_table_started = np.all([flag in line for flag in header4pos])
                if is_table_started:
                    break
                else:
                    line = f.readline()
        # Write the output to the file
        with open(output_filename, "w") as f:
            f.write(description_str)
            for i in range(len(id_list)):
                f.write(str(id_list[i]))
                f.write("  ")
                for dim in range(3):
                    f.write('{:3.6}'.format(avg_pos[i, dim]))
                    f.write("  ")
                f.write("\n")

    @staticmethod
    def _average_cell_over_steps(input_file: str, skip_steps: int) -> List[float]:
        '''
        average cell properties over time steps
        args:
        input_file: the input file e.g "./output/average_cell_low_temperature.dump"
        return:
        the dictionary contains the property_name and its averaged value
        e.g. {v_lx_metal:1.0,v_ly_metal:2.0 ...}
        '''
        with open(input_file, "r") as f:
            f.readline()  # skip the first line
            header = f.readline()
            header = header.replace("#", "")
        property_names = header.split()
        data = np.loadtxt(input_file, skiprows=2)
        time_step_index = property_names.index("TimeStep")
        time_step_data = data[:, time_step_index]
        cutoff_index = np.argmax(time_step_data > skip_steps)
        assert time_step_data[cutoff_index] > skip_steps
        assert cutoff_index == 0 or time_step_data[cutoff_index - 1] <= skip_steps
        mean_data = data[cutoff_index:].mean(axis=0).tolist()
        property_dict = {property_names[i]: mean_data[i] for i in range(len(mean_data)) if property_names[i] != "TimeStep"}
        return [property_dict["v_lx_metal"], property_dict["v_ly_metal"], property_dict["v_lz_metal"], 
                property_dict["v_xy_metal"], property_dict["v_xz_metal"], property_dict["v_yz_metal"]]

    @staticmethod
    def _get_positions_from_lammps_dump(filename: str) -> List[Tuple[float, float, float]]:
        lines = sorted(np.loadtxt(filename, skiprows=9).tolist(), key=lambda x: x[0])
        return [(line[1], line[2], line[3]) for line in lines]
    
    @staticmethod
    def _get_cell(cell_list: List[float]) -> npt.NDArray[np.float64]:
        assert len(cell_list) == 6
        cell = np.empty(shape=(3, 3))
        cell[0, :] = np.array([cell_list[0], 0.0, 0.0])
        cell[1, :] = np.array([cell_list[3], cell_list[1], 0.0])
        cell[2, :] = np.array([cell_list[4], cell_list[5], cell_list[2]])
        return cell

if __name__ == "__main__":

    # read parameters from stdin
    class Input:
        pass


    param = Input()
    read_params(param)

    # This queries for equilibrium structures in this prototype and builds atoms
    # test = TestDriver(model_name="Sim_LAMMPS_ReaxFF_ManzanoMoeiniMarinelli_2012_CaSiOH__SM_714124634215_000", stoichiometric_species=["Ca","Si","O"], prototype_label='AB3C_aP30_2_3i_9i_3i')

    # Alternatively, for debugging, give it atoms object or a list of atoms objects
    # atoms = bulk("TiN",'rocksalt',a=4.00)
    # test = TestDriver(model_name="MEAM_LAMMPS_KimLee_2008_TiN__MO_070542625990_002", atoms=atoms)
    test = TestDriver(model_name="MEAM_LAMMPS_KimLee_2008_TiN__MO_070542625990_002", stoichiometric_species=["Ti","N"], prototype_label='AB_cF8_225_a_b')
    test(temp=293.15,press=0,mass=param.mass)