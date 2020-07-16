# Adding missing loops

The complexes generated from docking (`4YNE`, `4AT3`, and `6KZD`) contained missing loop residues. A total of 200 models were created using [Modeller] (https://salilab.org/modeller/). Each model was scored using [Discrete Optimized Protein Energy](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2242414/) (DOPE) scoring. The `6KZD` contained a large number of missing residues (i.e > 8) in each loop region and so was capped at each break.

[x] NTRK1: `4YNE` (best model = `4YNE_fill.BL01040001.pdb`)
[x] NTRK2: `4AT3` (best model = `4AT3_fill.BL00850001.pdb`)
[x] NTRK3: `6KZD` (best model = N/A)