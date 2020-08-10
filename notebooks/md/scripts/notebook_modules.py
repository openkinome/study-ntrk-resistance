import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np

import MDAnalysis as mda
import MDAnalysis.analysis.rms
import MDAnalysis.transformations as trans

# Check here for good CB safe palettes: https://venngage.com/blog/color-blind-friendly-palette/
cb_colour_cycle = [
    (245, 105, 58),  # orange
    (169, 90, 161),  # purple
    (133, 192, 249),  # light blue
]


def centre_protein_gh(dict_of_systs, wrap=False, cent="geometry"):

    # The GroupHug class was created by Richard Gowers (https://github.com/richardjgowers)
    # in response to this question on the MDAnalysis forum:
    # https://groups.google.com/forum/#!topic/mdnalysis-discussion/umDpvbCmQiE

    class GroupHug:
        def __init__(self, center, *others):
            self.c = center
            self.o = others

        @staticmethod
        def calc_restoring_vec(ag1, ag2):
            box = ag1.dimensions[:3]
            dist = ag1.center_of_mass() - ag2.center_of_mass()

            return box * np.rint(dist / box)

        def __call__(self, ts):
            # loop over other atomgroups and shunt them into nearest image to center
            for i in self.o:
                rvec = self.calc_restoring_vec(self.c, i)

                i.translate(+rvec)

            return ts

    # Centre the protein in the box using MDAnalysis
    for ligand_name, syst in dict_of_systs.items():
        u = dict_of_systs[ligand_name]
        ligand_resname = ligand_name[:3]
        print(ligand_resname)

        if ligand_resname == "lar":
            print("WARNING: This script should only be used for the NTRK3 6KZD system!")

        # hard code the protein chains for now -> only for 6KZD model
        chainA = u.select_atoms("resid 527-627")
        chainB = u.select_atoms("resid 648-713")
        chainC = u.select_atoms("resid 728-838")
        lig = u.select_atoms("resname " + ligand_resname)
        ions = u.select_atoms("resname NA CL")

        protein = u.select_atoms("protein or resname ACE NME")
        reference = u.copy().select_atoms("protein or resname ACE NME")
        not_protein = u.select_atoms("not protein and not resname ACE NME")
        protein_and_lig = u.select_atoms("protein or resname ACE NME " + ligand_resname)

        transforms = [
            trans.unwrap(protein),
            trans.unwrap(lig),
            GroupHug(chainA, chainB, chainC, lig),
            trans.center_in_box(protein_and_lig, wrap=wrap, center="geometry"),
            trans.wrap(ions),
            trans.fit_rot_trans(protein, reference),
        ]

        dict_of_systs[ligand_name].trajectory.add_transformations(*transforms)

    return dict_of_systs


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


def calc_rmsd(name, dict_of_systs, skip=10):

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

        R.run(step=skip)
        rmsd_store.append(R)

    # Plot the RMSDs

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    time = rmsd_store[0].rmsd.T[1] / 1000

    for i, syst in enumerate(dict_of_systs):

        col = tuple(round(c / 255, 2) for c in cb_colour_cycle[i])

        rmsd = rmsd_store[i].rmsd.T  # transpose makes it easier for plotting
        ax.plot(time, rmsd[2], color=col, label=name + ":" + syst, alpha=0.8)

    ax.legend(loc="best")
    ax.set_xlabel("Time (ns)", fontsize=14)
    ax.set_ylabel(r"RMSD ($\AA$)", fontsize=14)


def delta_cog_inhib(name, dict_of_systs, skip=None, hist=False, save=False, **kwargs):

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

        for ts in u.trajectory[::skip]:

            inhib_store[syst]["time"].append(ts.time / 1000)

            dist = np.linalg.norm(inhib_sel.center_of_geometry() - ref_cog)

            inhib_store[syst]["cog"].append(dist)

    # Plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim([kwargs["xmin"], kwargs["xmax"]])
    ax.set_ylim([kwargs["ymin"], kwargs["ymax"]])

    i = 0

    for sel, syst in zip(inhib_resnames, dict_of_systs):

        col = tuple(round(c / 255, 2) for c in cb_colour_cycle[i])

        cog = inhib_store[syst]["cog"]
        time = inhib_store[syst]["time"]

        if hist:
            ax.hist(
                cog,
                density=True,
                bins=100,
                color=col,
                label=name + ":" + syst,
                alpha=0.8,
            )
        else:
            ax.plot(time, cog, color=col, label=name + ":" + syst, alpha=0.8)

        i += 1

    ax.legend(loc="best")

    if hist:
        ax.set_xlabel(r"$\Delta$ Distance ($\AA$)", fontsize=14)
        ax.set_ylabel("Density", fontsize=14)

    else:
        ax.set_xlabel("Time (ns)", fontsize=14)
        ax.set_ylabel(r"$\Delta$ Distance ($\AA$)", fontsize=14)

    if save:
        plt.savefig("inhib_cog_plot.png", format="png")
