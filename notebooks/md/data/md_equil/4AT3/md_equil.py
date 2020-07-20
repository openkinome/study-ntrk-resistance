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
from openmmforcefields.generators import SystemGenerator
from perses.utils.openeye import OEMol_to_omm_ff, createOEMolFromSDF
from simtk.openmm import MonteCarloBarostat, XmlSerializer
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
args = parser.parse_args()
chosen_ligand = args.ligand

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
nsteps = 2500000  # 5 ns
report_freq = 100
chk_freq = 500
traj_freq = 1000  # 2500 frames

# Set the input file names
input_pdb = "4AT3_prepped.pdb"
input_ligands_sdf = "../../structures_from_docking/4AT3_hybrid_docking.sdf"

# Create output directory
output_prefix = "./output/" + chosen_ligand
os.makedirs(output_prefix, exist_ok=True)
print("--> Directory ", output_prefix, " created ")

# Set file names
integrator_xml_filename = "integrator_2fs.xml"
state_xml_filename = "equilibrated_state_5ns.xml"
state_pdb_filename = "equilibrated_state_5ns.pdb"
system_xml_filename = "equilibrated_system_5ns.xml"
checkpoint_filename = "equilibrated_checkpoint_5ns.chk"
traj_output_filename = "equilibrated_traj_5ns.xtc"

# Define the barostat for the system
barostat = mm.MonteCarloBarostat(pressure, temperature)

# Load and sort ligands
molecules = Molecule.from_file(input_ligands_sdf)
ligand_names = ["larotrectinib", "selitrectinib", "repotrectinib"]
ligand_dict = dict(zip(ligand_names, molecules))  # Create dict for easy access later

# Make the SystemGenerator
system_generator = SystemGenerator(
    forcefields=[protein_forcefield, solvation_forcefield],
    barostat=barostat,
    periodic_forcefield_kwargs={"nonbondedMethod": app.PME},
    small_molecule_forcefield=small_molecule_forcefield,
    molecules=ligand_dict[chosen_ligand],
)

# Read in the PDB and create an OpenMM topology
pdbfile = app.PDBFile(input_pdb)
protein_topology, protein_positions = pdbfile.topology, pdbfile.positions

# Add ligand to topology - credit to @hannahbrucemacdonald for help here
print("--> Combining protein and ligand topologies")
off_ligand_topology = Topology.from_molecules(ligand_dict[chosen_ligand])
ligand_topology = off_ligand_topology.to_openmm()
ligand_positions = ligand_dict[chosen_ligand].conformers[0]

md_protein_topology = md.Topology.from_openmm(
    protein_topology
)  # using mdtraj for protein top
md_ligand_topology = md.Topology.from_openmm(
    ligand_topology
)  # using mdtraj for ligand top
md_complex_topology = md_protein_topology.join(md_ligand_topology)  # add them together

complex_topology = md_complex_topology.to_openmm()  # now back to openmm
total_atoms = len(protein_positions) + len(ligand_positions)
complex_positions = unit.Quantity(np.zeros([total_atoms, 3]), unit=unit.nanometers)
complex_positions[0 : len(protein_positions)] = protein_positions
for index, atom in enumerate(ligand_positions, len(protein_positions)):
    coords = atom / atom.unit
    complex_positions[index] = (
        coords / 10.0
    ) * unit.nanometers  # since openmm works in nm

# Add hydrogens and solvate the system
modeller = app.Modeller(complex_topology, complex_positions)
print("Adding hydrogens to the system...")
modeller.addHydrogens(system_generator.forcefield)
print("Solvating the system...")
modeller.addSolvent(
    forcefield=system_generator.forcefield,
    model="tip3p",
    ionicStrength=ionic_strength,
    padding=solvent_padding,
)

# Create an OpenMM system
print("--> Creating an OpenMM system")
system = system_generator.create_system(modeller.topology)

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
sim = app.Simulation(modeller.topology, system, integrator)  # , platform, prop)

# Set the particle positions
sim.context.setPositions(modeller.positions)

# Minimize the energy
print("--> Minimising energy with docked ligand: " + chosen_ligand)
print(
    "  initial : %8.3f kcal/mol"
    % (
        sim.context.getState(getEnergy=True).getPotentialEnergy()
        / unit.kilocalories_per_mole
    )
)
sim.minimizeEnergy()
print(
    "  final : %8.3f kcal/mol"
    % (
        sim.context.getState(getEnergy=True).getPotentialEnergy()
        / unit.kilocalories_per_mole
    )
)

# set starting velocities:
print("--> Generating random starting velocities")
sim.context.setVelocitiesToTemperature(temperature * unit.kelvin)

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
print("--> Running dynamics in the NPT ensemble for the 4AT3:" + chosen_ligand + " complex")
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
