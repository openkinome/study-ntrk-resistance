import argparse
import datetime
import logging
import os
import pickle

import simtk.unit as unit
from openmmtools import mcmc
from openmmtools.multistate import MultiStateReporter
from perses.annihilation.lambda_protocol import LambdaProtocol
from perses.samplers.multistate import HybridRepexSampler


# Set up logger
_logger = logging.getLogger()
_logger.setLevel(logging.DEBUG)

# Read args
parser = argparse.ArgumentParser(
    description="run perses protein mutation on protein:ligand complex"
)
parser.add_argument("dir", type=str, help="path to input/output dir")
parser.add_argument("phase", type=str, help="apo or complex")
args = parser.parse_args()

i = os.path.basename(os.path.dirname(args.dir))
htf = pickle.load(open(os.path.join(args.dir, f"{i}_{args.phase}.pickle"), "rb"))

# Build the hybrid repex samplers
suffix = "run"
selection = "not water"
checkpoint_interval = 10
n_states = 11
n_cycles = 5000
lambda_protocol = LambdaProtocol(functions="default")
reporter_file = os.path.join(args.dir, f"{i}_{args.phase}.nc")
reporter = MultiStateReporter(
    reporter_file,
    analysis_particle_indices=htf.hybrid_topology.select(selection),
    checkpoint_interval=checkpoint_interval,
)
hss = HybridRepexSampler(
    mcmc_moves=mcmc.LangevinSplittingDynamicsMove(
        timestep=4.0 * unit.femtoseconds,
        collision_rate=5.0 / unit.picosecond,
        n_steps=250,
        reassign_velocities=False,
        n_restart_attempts=20,
        splitting="V R R R O R R R V",
        constraint_tolerance=1e-06,
    ),
    hybrid_factory=htf,
    online_analysis_interval=10,
)
hss.setup(
    n_states=n_states,
    temperature=300 * unit.kelvin,
    storage_file=reporter,
    lambda_protocol=lambda_protocol,
    endstates=False,
)
hss.extend(n_cycles)