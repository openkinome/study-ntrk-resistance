# imports
import argparse
import logging
import os
import pickle
import shutil
import sys

import mdtraj as md
import numpy as np
import pandas as pd
from openeye import oechem
from perses.app.relative_point_mutation_setup import PointMutationExecutor
from perses.utils.openeye import generate_unique_atom_names
from simtk import unit

parser = argparse.ArgumentParser(
    description="Prepare hybrid topologies for NTRK mutant systems with flattened torsions"
)
parser.add_argument(
    "--index", dest="index", type=int, help="an integer for the system to create"
)
parser.add_argument(
    "--output_path",
    dest="output_path",
    type=str,
    help="the path of where to write out the files",
)
parser.add_argument(
    "--csv_data",
    dest="csv_data",
    type=str,
    help="the path to the CSV data e.g. /path/to/csv_data/",
)
parser.add_argument(
    "--data_path",
    dest="data_path",
    type=str,
    help="the path to both PDB and SDF data e.g. /path/to/data/",
)
parser.add_argument(
    "--small_mol_ff",
    dest="small_mol_ff",
    type=str,
    default="openff-1.3.0",
    help="the forcefield to use for the small molecule",
)
args = parser.parse_args()


def render_protein_residue_atom_mapping(topology_proposal, filename):
    """
    wrap the `render_atom_mapping` method around protein point mutation topologies.
    TODO : make modification to `render_atom_mapping` so that the backbone atoms are not written in the output.

    arguments
        topology_proposal : perses.rjmc.topology_proposal.TopologyProposal object
            topology proposal of protein mutation
        filename : str
            filename to write the map
    """
    from perses.utils.smallmolecules import render_atom_mapping

    oe_res_maps = {}
    for omm_new_idx, omm_old_idx in topology_proposal._new_to_old_atom_map.items():
        if omm_new_idx in topology_proposal._new_topology.residue_to_oemol_map.keys():
            try:
                oe_res_maps[
                    topology_proposal._new_topology.residue_to_oemol_map[omm_new_idx]
                ] = topology_proposal._old_topology.residue_to_oemol_map[omm_old_idx]
            except:
                pass

    render_atom_mapping(
        filename,
        topology_proposal._old_topology.residue_oemol,
        topology_proposal._new_topology.residue_oemol,
        oe_res_maps,
    )


# read in CSV data
df = pd.read_csv(args.csv_data, delimiter=",")

# create reference based on array job number (i)
index_dict = {
    0: ["NTRK1"],
    1: ["NTRK2"],
    2: ["NTRK3"],
}

# get the current system based on the parsed index
i = args.index

# create subset dataframe based on i number
df_i = df[df["ntrk"] == index_dict[i][0]]

# set the paths
data_path = args.data_path
out_path = args.output_path

# check mutations output directory doesn't exist
out_dir = f"{out_path}mutations_htf_output/flattened_torsions/"
print(f"Using {out_path} as the output directory...")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
else:
    sys.exit(f"{out_path}mutations_htf_output/ already exists! Exiting.")

# set three letter to one letter amino acid dictionary
aa = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}

htf_input_prefix = "kinoml_OEKLIFSKinaseHybridDockingFeaturizer_"

# make the systems
for index, row in df_i.iterrows():

    print(row)

    ligand_name = row["tki"]
    protein_name = row["ntrk"]
    resid = str(
        row["resid"]
    )  # The resid has to be a string for PointMutationExecutor input
    mutate_from = row["from"]
    mutate_to = row["to"]

    print(f"info: {protein_name}, {ligand_name}, {mutate_from}, {resid}, {mutate_to}")

    # Create protein directory
    output_prefix_protein = (
        f"{out_path}mutations_htf_output/flattened_torsions/{protein_name}"
    )
    if not os.path.exists(output_prefix_protein):
        os.makedirs(output_prefix_protein)
        print(f"--> Directory {output_prefix_protein} created")

    # Create protein mutant directory
    output_prefix_mutant = f"{out_path}mutations_htf_output/flattened_torsions/{protein_name}/{aa[mutate_from]}{resid}{aa[mutate_to]}"
    if not os.path.exists(output_prefix_mutant):
        os.makedirs(output_prefix_mutant)
        print(f"--> Directory {output_prefix_mutant} created")

    protein_file = (
        f"{data_path}{htf_input_prefix}{protein_name}_{ligand_name}_protein.pdb"
    )
    ligand_file = (
        f"{data_path}{htf_input_prefix}{protein_name}_{ligand_name}_ligand.sdf"
    )
    print(f"Using protein file: {protein_file}")
    print(f"Using ligand file: {ligand_file}")

    # create oemol object for NTRK ligands due to issues with stereochemistry - DEPRECATED
    # ifs = oechem.oemolistream()
    # ifs.open(ligand_file)
    # mol_list = [oechem.OEMol(mol) for mol in ifs.GetOEMols()]
    # mol = mol_list[0]
    # if len([atom.GetName() for atom in mol.GetAtoms()]) > len(set([atom.GetName() for atom in mol.GetAtoms()])):
    #         mol = generate_unique_atom_names(mol)
    # oechem.OEAssignAromaticFlags(mol, oechem.OEAroModelOpenEye)
    # oechem.OEAssignHybridization(mol)
    # oechem.OEAddExplicitHydrogens(mol)
    # oechem.OEPerceiveChiral(mol)

    # make the system
    solvent_delivery = PointMutationExecutor(
        protein_file,
        "1",  # First and only protein chain
        resid,
        mutate_to,
        ligand_file=ligand_file,
        allow_undefined_stereo=True,
        ionic_strength=0.15 * unit.molar,
        small_molecule_forcefields=args.small_mol_ff,
        flatten_torsions=True,  # we're using flattened torsions
        flatten_exceptions=True,
        conduct_endstate_validation=False,
    )

    # make image map of the transformation
    render_protein_residue_atom_mapping(
        solvent_delivery.get_apo_htf()._topology_proposal,
        f"{output_prefix_mutant}/{protein_name}_{ligand_name}_{aa[mutate_from]}{resid}{aa[mutate_to]}_apo_map.png",
    )

    # pickle the output and save
    pickle.dump(
        solvent_delivery.get_apo_htf(),
        open(
            os.path.join(
                output_prefix_mutant,
                f"{protein_name}_{ligand_name}_{aa[mutate_from]}{resid}{aa[mutate_to]}_apo.pickle",
            ),
            "wb",
        ),
    )

    pickle.dump(
        solvent_delivery.get_complex_htf(),
        open(
            os.path.join(
                output_prefix_mutant,
                f"{protein_name}_{ligand_name}_{aa[mutate_from]}{resid}{aa[mutate_to]}_complex.pickle",
            ),
            "wb",
        ),
    )

    # save the coordinates to of apo and complex to check the geometry of the transformation
    htfs_t = [solvent_delivery.get_apo_htf(), solvent_delivery.get_complex_htf()]

    top_old = md.Topology.from_openmm(htfs_t[0]._topology_proposal.old_topology)
    top_new = md.Topology.from_openmm(htfs_t[0]._topology_proposal.new_topology)
    traj = md.Trajectory(
        np.array(htfs_t[0].old_positions(htfs_t[0].hybrid_positions)), top_old
    )
    traj.save(
        f"{output_prefix_mutant}/{protein_name}_{ligand_name}_{aa[mutate_from]}{resid}{aa[mutate_to]}_apo_old.pdb"
    )
    traj = md.Trajectory(
        np.array(htfs_t[0].new_positions(htfs_t[0].hybrid_positions)), top_new
    )
    traj.save(
        f"{output_prefix_mutant}/{protein_name}_{ligand_name}_{aa[mutate_from]}{resid}{aa[mutate_to]}_apo_new.pdb"
    )

    top_old = md.Topology.from_openmm(htfs_t[1]._topology_proposal.old_topology)
    top_new = md.Topology.from_openmm(htfs_t[1]._topology_proposal.new_topology)
    traj = md.Trajectory(
        np.array(htfs_t[1].old_positions(htfs_t[1].hybrid_positions)), top_old
    )
    traj.save(
        f"{output_prefix_mutant}/{protein_name}_{ligand_name}_{aa[mutate_from]}{resid}{aa[mutate_to]}_complex_old.pdb"
    )
    traj = md.Trajectory(
        np.array(htfs_t[1].new_positions(htfs_t[1].hybrid_positions)), top_new
    )
    traj.save(
        f"{output_prefix_mutant}/{protein_name}_{ligand_name}_{aa[mutate_from]}{resid}{aa[mutate_to]}_complex_new.pdb"
    )
