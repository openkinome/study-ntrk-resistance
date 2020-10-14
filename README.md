# NTRK resistance example
With the code and notebooks of this repository we aim to explore the impact of point mutations on ligand binding to NTRKs.

### How to use this repository

1. Clone repository

`git clone https://github.com/openkinome/ntrk_resistance_example`

2. Create Conda environment
  
`conda env create -f environment.yml`  
`conda activate ntrk_mutations`

### Structure
- `data/activities.csv`
  - mutational data from [Drilon et al. 2017](https://www.doi.org/10.1158/2159-8290.CD-17-0507) and 
  [Drilon et al. 2018](https://www.doi.org/10.1158/2159-8290.CD-18-0484)
- `notebooks/docking/docking.ipynb`  
  - jupyter notebook explaining how NTRK structures were prepared and how docking poses were generated
- `notebooks/kinoml_modeling/NTRK1_complex_modeling.ipynb.ipynb`  
  - jupyter notebook explaining how NTRK structures can automatically be prepared using KinoML
- `notebooks/md_sims/md.ipynb`  
  - jupyter notebook analysing results from MD simulations of systems prepared with docking.



### Authors

- David Schaller <david.schaller@charite.de>
- William Glass <william.glass@choderalab.org>

### License
This repository is licensed under the MIT license.
