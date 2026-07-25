# ReadMe for: Snake-Swimming-Kinematics

This repository contains the python code and Jupyter notebooks to reproduce the kinematics figures and calculations
in Gregorio et al 2026 (https://doi.org/10.1103/c57h-kx57).

## Data Repository

You can find the data repository associated with this code here:

Gregorio, Elizabeth; Godoy-Diana, Ramiro; Herrel, Anthony, 2026, "Swimming kinematics and volumetric wake measurements for Natrix maura and Nerodia rhombifer", https://doi.org/10.48579/PRO/5Q27ST, data.InDoRES

## Citation

If you use this code, please cite:

Gregorio, E., Godoy-Diana, R., & Herrel, A. (2026). Turning without fins: quantifying the distinct kinematics and vortex dynamics of maneuvering swimming snakes. Physical Review E. https://doi.org/10.1103/c57h-kx57

## Contact

Elizabeth Gregorio: elizabeth.gregorio@espci.fr

## Files
	
- **`MOI-STRAIGHT-withHDF5.ipynb`** — a Jupyter notebook to reproduce figures of forward swimming sequences and calculate associated statistics used in `MOI-Turn-withHDF5.ipynb`

- **`MOI-Turn-withHDF5.ipynb`** — a Jupyter notebook to reproduce figures of turning sequences

- **`utils/snakeFunctions.py`** — python functions to find the snake in the image with thresholding, find endpoints of the skeleton, put the skeleton points in order, and smooth the centerline.

- **`utils/snakeMOI.py`** — python functions to calculate the moment of inertia and convert values from pixels to SI units.

- **`utils/snakeWidth.py`** — python function to estimate the width of the snake at points along the centerline

## Funding

This work is funded by the Agence Nationale de la Recherche (France) through project DRAGON2 (ANR-20- CE02-0010).

## License

Licensed under the MIT License.
