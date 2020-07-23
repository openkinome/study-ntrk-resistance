import MDAnalysis as mda
import MDAnalysis.analysis.rms
import MDAnalysis.transformations as trans
import matplotlib.pyplot as plt
import numpy as np

cb_colour_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]


def centre_protein(dict_of_systs, wrap=False, cent="geometry"):

    # Centre the protein in the box using MDAnalysis
    for syst in dict_of_systs:
        u = dict_of_systs[syst]
        reference = u.copy().select_atoms("protein or resname ACE NME")

        protein = u.select_atoms("protein or resname ACE NME")
        not_protein = u.select_atoms("not protein")

        transforms = [
            trans.center_in_box(protein, wrap=wrap, center=cent),
            trans.wrap(not_protein),
            trans.fit_rot_trans(protein, reference),
        ]

        dict_of_systs[syst].trajectory.add_transformations(*transforms)

    return dict_of_systs


def calc_rmsd(name, dict_of_systs):

    rmsd_store = []

    for syst in dict_of_systs:
        print("Calculating RMSD for " + name + ":" + syst)

        # Load system and reset trajectory to the first frame
        u = dict_of_systs[syst]
        u.trajectory[0]
        ref = u

        R = mda.analysis.rms.RMSD(
            u,
            ref,
            select="backbone",  # superimpose on whole backbone of the whole protein
            groupselections=["backbone"],
        )  # whole protein

        # "backbone and resid 554-566" # C-Helix
        # "backbone and resid 668-670" # DFG motif
        # "backbone and resid 517-522" # Glycine loop
        # "backbone and resid 544-560" # Conserved lysine, glutamate
        # "backbone and resid 648-650" # HRD motif
        # "backbone and resid 668-694" # Activation loop

        R.run()
        rmsd_store.append(R)

    # Plot the RMSDs
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    for i, syst in enumerate(dict_of_systs):

        rmsd = rmsd_store[i].rmsd.T  # transpose makes it easier for plotting
        time = rmsd[1] / 1000
        ax.plot(time, rmsd[2], cb_colour_cycle[i], label=name + ":" + syst, alpha=0.8)
        ax.legend(loc="best")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel(r"RMSD ($\AA$)")


def delta_cog_inhib(name, dict_of_systs):

    # Calculate the distance changes between the original docked ligand's centre of geometry
    # and the centre of geometry at each frame

    inhib_resnames = ["resname lar", "resname sel", "resname rep"]
    inhib_store = {
        "larotrectinib": {"time": [], "cog": []},
        "selitrectinib": {"time": [], "cog": []},
        "repotrectinib": {"time": [], "cog": []},
    }

    for sel, syst in zip(inhib_resnames, dict_of_systs):

        print("current system: " + syst)

        u = dict_of_systs[syst]
        u.trajectory[0]

        inhib_sel = u.select_atoms(sel)
        ref_cog = inhib_sel.center_of_geometry()

        for ts in u.trajectory:

            inhib_store[syst]["time"].append(ts.time / 1000)

            dist = np.linalg.norm(inhib_sel.center_of_geometry() - ref_cog)

            inhib_store[syst]["cog"].append(dist)

    # Plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    i = 0

    for sel, syst in zip(inhib_resnames, dict_of_systs):

        cog = inhib_store[syst]["cog"]
        time = inhib_store[syst]["time"]
        ax.plot(time, cog, cb_colour_cycle[i], label=name + ":" + syst, alpha=0.8)
        ax.legend(loc="best")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel(r"$\Delta$ Distance ($\AA$)")

        i += 1
