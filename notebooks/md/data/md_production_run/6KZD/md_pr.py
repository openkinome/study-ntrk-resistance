import argparse
import os
import sys
from sys import stdout

import mdtraj as md
import numpy as np
import parmed
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
from openforcefield.topology import Molecule, Topology
from simtk.openmm import XmlSerializer
from simtk.openmm.app import CheckpointReporter, ForceField, PDBFile
from simtk.openmm.app.pdbreporter import PDBReporter
from simtk.openmm.app.statedatareporter import StateDataReporter

# Read arguments to get ligand
parser = argparse.ArgumentParser()
parser.add_argument(
    "-ligand",
    help="the docked ligand to be prepared for simulation",
    choices=["larotrectinib", "selitrectinib", "repotrectinib"],
    type=str,
)
parser.add_argument(
    "-input_pdb",
    help="the equilibrated system: a .pdb file",
    type=str,
)
parser.add_argument(
    "-input_system",
    help="the equilibrated system: a .xml file",
    type=str,
)
parser.add_argument(
    "-input_state",
    help="the equilibrated state: a .xml file",
    type=str,
)
args = parser.parse_args()
chosen_ligand = args.ligand
input_pdb = args.input_pdb
input_system = args.input_system
input_state = args.input_state

# Parameters
print("--> Reading parameters")
pressure = 1.0 * unit.bar
temperature = 300 * unit.kelvin
nonbonded_method = app.PME
constraints = app.HBonds
remove_cm_motion = True
collision_rate = 1.0 / unit.picoseconds
timestep = 0.002 * unit.picoseconds
solvent_padding = 10.0 * unit.angstrom
ionic_strength = 150 * unit.millimolar

# Forcefield
protein_forcefield = "amber14/protein.ff14SB.xml"
small_molecule_forcefield = "openff-1.1.0"
solvation_forcefield = "amber14/tip3p.xml"
forcefield = ForceField(protein_forcefield, solvation_forcefield)

# Set steps and frequencies
nsteps = 250000000  # 500 ns
report_freq = 500  # Report at 1 ns intervals
chk_freq = 500  #  Create checkpoint file at 1 ns intervals
traj_freq = 25000  # Each frame = 0.05 ns with 10,000 frames in total 

# Create output directory
output_prefix = "./output/" + chosen_ligand
os.makedirs(output_prefix, exist_ok=True)
print("--> Directory ", output_prefix, " created ")

# Set file names
integrator_xml_filename = "integrator_2fs.xml"
state_xml_filename = "pr_output_state.xml"
state_pdb_filename = "pr_output_state.pdb"
system_xml_filename = "pr_output_system.xml"
checkpoint_filename = "pr_output_checkpoint.chk"
traj_output_filename = "pr_output_traj.xtc"

# Define the barostat for the system
barostat = mm.MonteCarloBarostat(pressure, temperature)

# Read in equilibrated system (PDB)
pdb = PDBFile(input_pdb)

# Deserialize system file and load system
with open(input_system, 'r') as f:
    system = XmlSerializer.deserialize(f.read())

# Make and serialize integrator - Langevin dynamics
print(
    "Serializing integrator to %s"
    % os.path.join(output_prefix, integrator_xml_filename)
)
integrator = mm.LangevinIntegrator(
    temperature, collision_rate, timestep  # Friction coefficient
)
with open(os.path.join(output_prefix, integrator_xml_filename), "w") as outfile:
    xml = mm.XmlSerializer.serialize(integrator)
    outfile.write(xml)

# Define the platform to use; CUDA, OpenCL, CPU, or Reference. Or do not specify
# the platform to use the default (fastest) platform
# platform = mm.Platform.getPlatformByName("OpenCL")
# prop = dict(OpenCLPrecision="mixed")  # Use mixed single/double precision

# Create the Simulation object
sim = app.Simulation(pdb.topology, system, integrator)  # , platform, prop)

# Load state and set box vectors, positions, and velocities
with open(input_state, 'r') as infile:
    state_xml = infile.read()
state = XmlSerializer.deserialize(state_xml)
sim.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
sim.context.setPositions(state.getPositions())
sim.context.setVelocities(state.getVelocities())
sim.context.setTime(state.getTime())

# write limited state information to standard out:
sim.reporters.append(
    StateDataReporter(
        stdout,
        reportInterval=report_freq,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        temperature=True,
        speed=True,
        progress=True,
        remainingTime=True,
        totalSteps=nsteps,
        separator="\t",
    )
)

# Write to checkpoint files regularly:
sim.reporters.append(
    CheckpointReporter(
        file=os.path.join(output_prefix, checkpoint_filename), reportInterval=chk_freq
    )
)

# Write out the trajectory
sim.reporters.append(
    md.reporters.XTCReporter(
        file=os.path.join(output_prefix, traj_output_filename), reportInterval=traj_freq
    )
)

# Run NPT dynamics
print("--> Running dynamics in the NPT ensemble for the 6KZD:" + chosen_ligand + " complex")
sim.step(nsteps)

# Save and serialize the final state
print("--> Serializing state to %s" % os.path.join(output_prefix, state_xml_filename))
state = sim.context.getState(
    getPositions=True, getVelocities=True, getEnergy=True, getForces=True
)
with open(os.path.join(output_prefix, state_xml_filename), "w") as outfile:
    xml = mm.XmlSerializer.serialize(state)
    outfile.write(xml)

# Save the final state as a PDB
print("--> Saving final state as %s" % os.path.join(output_prefix, state_pdb_filename))
with open(os.path.join(output_prefix, state_pdb_filename), "w") as outfile:
    PDBFile.writeFile(
        sim.topology,
        sim.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(),
        file=outfile,
        keepIds=True,
    )

# Save and serialize system
print("--> Serializing system to %s" % os.path.join(output_prefix, system_xml_filename))
system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
with open(os.path.join(output_prefix, system_xml_filename), "w") as outfile:
    xml = mm.XmlSerializer.serialize(system)
    outfile.write(xml)