"""
Generate a test_generator.json for testgenie to be able to
generate Tests with

Date: 11/19/2018
Author: Daniel S. Karls (karl0100 |AT| umn DOT edu), University of Minnesota

"""

import json
from ase.data import chemical_symbols
from ase.data import reference_states
from random import randint

# Read periodic table
periodic_table = {}
with open("periodic_table.txt") as f:
    for line in f.read().splitlines():
        atomic_number, symbol, fullname, atomic_mass = line.split()
        periodic_table[symbol] = {
            "atomic_number": atomic_number,
            "fullname": fullname,
            "atomic_mass": atomic_mass,
        }

lattices = ["bcc", "diamond", "fcc", "sc"]

species = {}
species["bcc"] = [
    "Ba",
    "Cr",
    "Cs",
    "Eu",
    "Fe",
    "K",
    "Li",
    "Mo",
    "Na",
    "Rb",
    "Ta",
    "V",
    "W",
]

species["diamond"] = ["C", "Ge", "Si", "Sn"]

species["fcc"] = [
    "Ac",
    "Ag",
    "Al",
    "Ar",
    "Au",
    "Ca",
    "Ce",
    "Cu",
    "Ir",
    "Kr",
    "Ne",
    "Ni",
    "Pb",
    "Pd",
    "Pt",
    "Rh",
    "Sr",
    "Th",
    "Xe",
    "Yb",
]

species["sc"] = ["Po"]

existing_kim_nums = {}
existing_kim_nums["bcc"] = {}
existing_kim_nums["diamond"] = {}
existing_kim_nums["fcc"] = {}
existing_kim_nums["sc"] = {}

existing_kim_nums["bcc"]["Ba"] = "132553522497"
existing_kim_nums["bcc"]["Cr"] = "435511432078"
existing_kim_nums["bcc"]["Cs"] = "124842053505"
existing_kim_nums["bcc"]["Eu"] = "883193159339"
existing_kim_nums["bcc"]["Fe"] = "506786620750"
existing_kim_nums["bcc"]["K"] = "293947541816"
existing_kim_nums["bcc"]["Li"] = "940119952339"
existing_kim_nums["bcc"]["Mo"] = "653330286461"
existing_kim_nums["bcc"]["Na"] = "398765858860"
existing_kim_nums["bcc"]["Rb"] = "027845558943"
existing_kim_nums["bcc"]["Ta"] = "537849920850"
existing_kim_nums["bcc"]["V"] = "417640301289"
existing_kim_nums["bcc"]["W"] = "489123578653"
existing_kim_nums["diamond"]["C"] = "640411322333"
existing_kim_nums["diamond"]["Ge"] = "778011010022"
existing_kim_nums["diamond"]["Si"] = "782453122212"
existing_kim_nums["diamond"]["Sn"] = "836750009088"
existing_kim_nums["fcc"]["Ac"] = "611064980563"
existing_kim_nums["fcc"]["Ag"] = "016048498506"
existing_kim_nums["fcc"]["Al"] = "957040092249"
existing_kim_nums["fcc"]["Ar"] = "732820333279"
existing_kim_nums["fcc"]["Au"] = "173429922932"
existing_kim_nums["fcc"]["Ca"] = "389870929270"
existing_kim_nums["fcc"]["Ce"] = "716914137201"
existing_kim_nums["fcc"]["Cu"] = "335019190158"
existing_kim_nums["fcc"]["Ir"] = "657654753530"
existing_kim_nums["fcc"]["Kr"] = "080995775402"
existing_kim_nums["fcc"]["Ne"] = "243550005184"
existing_kim_nums["fcc"]["Ni"] = "127978642829"
existing_kim_nums["fcc"]["Pb"] = "051450577568"
existing_kim_nums["fcc"]["Pd"] = "728704926608"
existing_kim_nums["fcc"]["Pt"] = "325427650920"
existing_kim_nums["fcc"]["Rh"] = "500354655277"
existing_kim_nums["fcc"]["Sr"] = "380277273371"
existing_kim_nums["fcc"]["Th"] = "780283590779"
existing_kim_nums["fcc"]["Xe"] = "246024074275"
existing_kim_nums["fcc"]["Yb"] = "033069610010"
existing_kim_nums["sc"]["Po"] = "573533540838"

with open("test_generator.json", "w") as f:
    for lattice in lattices:

        for elem in species[lattice]:

            f.write(
                json.dumps(
                    {
                        "symbol": elem,
                        "elemnumber": periodic_table[elem]["atomic_number"],
                        "mass": periodic_table[elem]["atomic_mass"],
                        "fullname": periodic_table[elem]["atomic_mass"],
                        "lattice": lattice,
                        "kimnum": existing_kim_nums[lattice][elem],
                        "version": "002",
                        "temperature": "293.15",
                        "pressure": "0",
                    }
                )
                + "\n"
            )
