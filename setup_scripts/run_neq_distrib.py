import argparse
import bz2
import logging
import os
import pickle
import time

import mdtraj as md
import numpy as np
from openmmtools.integrators import PeriodicNonequilibriumIntegrator
from simtk import openmm, unit
from simtk.openmm import XmlSerializer

from openeye import oechem

# Set up logger
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)

# Read args
parser = argparse.ArgumentParser(description="run perses noneq switching: Abl1 kinase")
parser.add_argument("--index", dest="index", type=int, help="the index of the job ID")
parser.add_argument(
    "--output_dir", dest="output_dir", type=str, help="path to output dir"
)
parser.add_argument(
    "--input_dir",
    dest="input_dir",
    type=str,
    help="path to the directory containing the .pickle htf files",
)
parser.add_argument(
    "--phase", dest="phase", type=str, help="apo or complex", required=True
)
parser.add_argument(
    "--mutation",
    dest="mutation",
    type=str,
    help="the mutation in the form X123Y",
    required=True,
)
parser.add_argument(
    "--tki",
    dest="tki",
    type=str,
    help="the name of the inhibitor",
    required=True,
)
parser.add_argument(
    "--save_traj",
    dest="save_traj",
    type=bool,
    help="option to save the trajectory",
    default=False,
    required=False,
)

args = parser.parse_args()

# Define lambda functions
x = "lambda"
DEFAULT_ALCHEMICAL_FUNCTIONS = {
    "lambda_sterics_core": x,
    "lambda_electrostatics_core": x,
    "lambda_sterics_insert": f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
    "lambda_sterics_delete": f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
    "lambda_electrostatics_insert": f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
    "lambda_electrostatics_delete": f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
    "lambda_bonds": x,
    "lambda_angles": x,
    "lambda_torsions": x,
}

# Define simulation parameters
temperature = 300  # 300 K
nsteps_eq = 375000  # 1.5 ns
nsteps_neq = 375000  # 1.5 ns
neq_splitting = "V R H O R V"
timestep = 4.0 * unit.femtosecond
platform_name = "CUDA"

# Read in htf
# assuming the dir path is as: "/data/chodera/glassw/kinoml/NTRK/mutations_htf_output/NTRK1/X123Y"
pdb_code = os.path.basename(
    os.path.dirname(args.input_dir)
)  # For NTRK, pdb_code = NTRKX
mutation = args.mutation

with open(
    os.path.join(
        args.input_dir,
        f"{pdb_code}_{args.tki}_{mutation}_{args.phase}.pickle",
    ),
    "rb",
) as f:
    htf = pickle.load(f)
system = htf.hybrid_system
positions = htf.hybrid_positions

# Set up integrator
integrator = PeriodicNonequilibriumIntegrator(
    DEFAULT_ALCHEMICAL_FUNCTIONS,
    nsteps_eq,
    nsteps_neq,
    neq_splitting,
    timestep=timestep,
)

# Set up context
platform = openmm.Platform.getPlatformByName(platform_name)
if platform_name in ["CUDA", "OpenCL"]:
    platform.setPropertyDefaultValue("Precision", "mixed")
if platform_name in ["CUDA"]:
    platform.setPropertyDefaultValue("DeterministicForces", "true")
context = openmm.Context(system, integrator, platform)
context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

# Minimize
openmm.LocalEnergyMinimizer.minimize(context)

# Run neq
ncycles = 1
forward_works_master, reverse_works_master = list(), list()
forward_eq_old, forward_neq_old, forward_neq_new = list(), list(), list()
reverse_eq_new, reverse_neq_old, reverse_neq_new = list(), list(), list()

for cycle in range(ncycles):
    # Equilibrium (lambda = 0)
    for step in range(nsteps_eq):
        initial_time = time.time()
        integrator.step(1)
        elapsed_time = (time.time() - initial_time) * unit.seconds
        if step % 7500 == 0:
            _logger.info(
                f"Cycle: {cycle}, Step: {step}, equilibrating at lambda = 0, took: {elapsed_time / unit.seconds} seconds"
            )
            pos = context.getState(
                getPositions=True, enforcePeriodicBox=False
            ).getPositions(asNumpy=True)
            old_pos = np.asarray(htf.old_positions(pos))
            forward_eq_old.append(old_pos)

    # Forward (0 -> 1)
    forward_works = [integrator.get_protocol_work(dimensionless=True)]
    for fwd_step in range(nsteps_neq):
        initial_time = time.time()
        integrator.step(1)
        elapsed_time = (time.time() - initial_time) * unit.seconds
        if fwd_step % 300 == 0:
            _logger.info(
                f"Cycle: {cycle}, forward NEQ step: {fwd_step}, took: {elapsed_time / unit.seconds} seconds"
            )
            pos = context.getState(
                getPositions=True, enforcePeriodicBox=False
            ).getPositions(asNumpy=True)
            old_pos = np.asarray(htf.old_positions(pos))
            new_pos = np.asarray(htf.new_positions(pos))
            forward_neq_old.append(old_pos)
            forward_neq_new.append(new_pos)
            forward_works.append(integrator.get_protocol_work(dimensionless=True))

    forward_works_master.append(forward_works)

    # Equilibrium (lambda = 1)
    for step in range(nsteps_eq):
        initial_time = time.time()
        integrator.step(1)
        elapsed_time = (time.time() - initial_time) * unit.seconds
        if step % 7500 == 0:
            _logger.info(
                f"Cycle: {cycle}, Step: {step}, equilibrating at lambda = 1, took: {elapsed_time / unit.seconds} seconds"
            )
            pos = context.getState(
                getPositions=True, enforcePeriodicBox=False
            ).getPositions(asNumpy=True)
            new_pos = np.asarray(htf.new_positions(pos))
            reverse_eq_new.append(new_pos)

    # Reverse work (1 -> 0)
    reverse_works = [integrator.get_protocol_work(dimensionless=True)]
    for rev_step in range(nsteps_neq):
        initial_time = time.time()
        integrator.step(1)
        elapsed_time = (time.time() - initial_time) * unit.seconds
        if rev_step % 300 == 0:
            _logger.info(
                f"Cycle: {cycle}, reverse NEQ step: {rev_step}, took: {elapsed_time / unit.seconds} seconds"
            )
            pos = context.getState(
                getPositions=True, enforcePeriodicBox=False
            ).getPositions(asNumpy=True)
            old_pos = np.asarray(htf.old_positions(pos))
            new_pos = np.asarray(htf.new_positions(pos))
            reverse_neq_old.append(old_pos)
            reverse_neq_new.append(new_pos)
            reverse_works.append(integrator.get_protocol_work(dimensionless=True))

    reverse_works_master.append(reverse_works)

    # Save works
    with open(
        os.path.join(
            args.output_dir,
            f"{pdb_code}_{args.tki}_{mutation}_{args.phase}_{args.index}_forward.npy",
        ),
        "wb",
    ) as f:
        np.save(f, forward_works_master)
    with open(
        os.path.join(
            args.output_dir,
            f"{pdb_code}_{args.tki}_{mutation}_{args.phase}_{args.index}_reverse.npy",
        ),
        "wb",
    ) as f:
        np.save(f, reverse_works_master)

    if args.save_traj:
        # Save trajs without solvent
        traj_forward_eq_old = md.Trajectory(
            np.array(forward_eq_old),
            md.Topology.from_openmm(htf._topology_proposal.old_topology),
        )
        traj_forward_eq_old.remove_solvent()
        traj_forward_eq_old.save(
            os.path.join(
                args.output_dir,
                f"{pdb_code}_{args.tki}_{mutation}_{args.phase}_{args.index}_forward_eq_old.xtc",
            )
        )

        traj_reverse_eq_new = md.Trajectory(
            np.array(reverse_eq_new),
            md.Topology.from_openmm(htf._topology_proposal.new_topology),
        )
        traj_reverse_eq_new.remove_solvent()
        traj_reverse_eq_new.save(
            os.path.join(
                args.output_dir,
                f"{pdb_code}_{args.tki}_{mutation}_{args.phase}_{args.index}_reverse_eq_new.xtc",
            )
        )

        traj_forward_neq_old = md.Trajectory(
            np.array(forward_neq_old),
            md.Topology.from_openmm(htf._topology_proposal.old_topology),
        )
        traj_forward_neq_old.remove_solvent()
        traj_forward_neq_old.save(
            os.path.join(
                args.output_dir,
                f"{pdb_code}_{args.tki}_{mutation}_{args.phase}_{args.index}_forward_neq_old.xtc",
            )
        )

        traj_forward_neq_new = md.Trajectory(
            np.array(forward_neq_new),
            md.Topology.from_openmm(htf._topology_proposal.new_topology),
        )
        traj_forward_neq_new.remove_solvent()
        traj_forward_neq_new.save(
            os.path.join(
                args.output_dir,
                f"{pdb_code}_{args.tki}_{mutation}_{args.phase}_{args.index}_forward_neq_new.xtc",
            )
        )

        traj_reverse_neq_old = md.Trajectory(
            np.array(reverse_neq_old),
            md.Topology.from_openmm(htf._topology_proposal.old_topology),
        )
        traj_reverse_neq_old.remove_solvent()
        traj_reverse_neq_old.save(
            os.path.join(
                args.output_dir,
                f"{pdb_code}_{args.tki}_{mutation}_{args.phase}_{args.index}_reverse_neq_old.xtc",
            )
        )

        traj_reverse_neq_new = md.Trajectory(
            np.array(reverse_neq_new),
            md.Topology.from_openmm(htf._topology_proposal.new_topology),
        )
        traj_reverse_neq_new.remove_solvent()
        traj_reverse_neq_new.save(
            os.path.join(
                args.output_dir,
                f"{pdb_code}_{args.tki}_{mutation}_{args.phase}_{args.index}_reverse_neq_new.xtc",
            )
        )
