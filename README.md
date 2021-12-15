# Efficient and Robust Discrete Conformal Equivalence with Boundary
Marcel Campen, Ryan Capouellez, Hanxiao Shen, Leyi Zhu, Daniele Panozzo, Denis Zorin. ACM Transactions on Graphics (SIGGRAPH Asia 2021)

![build](https://github.com/geometryprocessing/ConformalIdealDelaunay/workflows/CMake/badge.svg)
![Examples](figures/teaser/teaser.png?raw=true "Title")

## Abstract
Given target discrete curvature per vertex on a triangle mesh, we describe a method to efficiently and robustly compute a discrete conformal deformation
of the mesh’s metric to satisfy this prescription. The method supports closed surfaces as well as surfaces with boundary.
By prescribing the geodesic curvature along the boundary, alignment of the parametrization with the boundary can be enforced. 
Internally, Ideal Delaunay triangulation is used to guarantee reliability.
## Clone
You can clone the repo and submodules with
```
git clone --recursive https://github.com/geometryprocessing/ConformalIdealDelaunay.git
```
## Dependencies
[libigl](https://libigl.github.io/), [spdlog](https://github.com/gabime/spdlog) and [pybind11](https://pybind11.readthedocs.io/en/stable/) are included as submodules, [boost](https://www.boost.org/) is required to be installed by user. [pdflatex](https://www.tug.org/applications/pdftex/) is also required to use the scripts in `figures` folder to reproduce figures in the paper.

MPFR (for the extended precision version) is optionally included from CGAL via libigl. To enable it, make sure the following option is turned on in CMakeLists.txt
```
option(LIBIGL_WITH_CGAL                "Use CGAL"           ON)
```
To generate texture visualizations as shown in the paper, turn on the `RENDER_TEXTURE` option.
```
option(RENDER_TEXTURE                  "Render results"     ON)
```
Embree (to speed up texture visualization) can be enabled by
```
option(LIBIGL_WITH_EMBREE              "Use EMBREE"         ON)
```
## Use (Code)
Include ConformalInterface.hh and use the top-level methods conformal_metric_* or conformal_parametrization_*.

## Use (Command Line)
We provide a top-level python interface `py/script_conformal.py` to run the algorithm.
Use environment.yml to setup the python environment via conda:
```
conda env create --file=environment.yml
conda activate cm_env
mkdir build
cd build
cmake ..
make
```
### Command
```
python py/script_conformal.py -i IN_DIR -f FILENAME [--options]
```
For example, with the following command
```
python py/script_conformal.py -i data/examples -f elephant.obj --error_log --print_summary --output_type param --output_format obj
```
the script produces the following output files in the `out` directory (will be auto-created if non-exist):
- `elephant_float.csv`, records the optimization information at the end of every iteration.
- `summary_delaunay.csv`, records the summary when converged e.g. total edge flips performed, total time spent etc.
- `elephant_out.obj`, records the parametrization result stored as texture coordinates for the overlay mesh.
### Options
```
Required parameters:
  "-i", "--input",      type=str, help="input folder that stores obj files and Th_hat"
  "-f", "--fname",      type=str, help="filename of the obj file"

Optional parameters:
  "-o", "--output",     type=str, help="output folder for stats", default="out"
  "--use_mpf",          type=bool, help="True for enable multiprecision"
  "--do_reduction",     type=bool, help="do reduction for search direction"
  "-p", "--prec",       type=int, help="choose the mantissa value of mpf", type=int
  "-m", "--max_itr",    type=int, help="choose the maximum number of iterations", type=int, default=500
  "--energy_cond",      type=bool, help="True for enable energy computation for line-search"
  "--error_log",        type=bool, help="True for enable writing out the max/ave angle errors per newton iteration"
  "--flip_count",       type=bool, help="True for enable collecting flip type stats"
  "--round_Th_hat",     type=bool, help="True for rounding Th_hat values to multiples of pi/60"
  "--print_summary",    type=bool, help="print a summary table contains target angle range and final max curvature error"
  "--no_plot_result",   type=bool, help="True for NOT rendering the results, used only for reproducing figures to speedup."
  "--no_lm_reset",      type=bool, help="True for using double the previous lambda for line search."
  "--suffix",           type=int, help="id assigned to each model for the random test"
  "--eps",              type=float, help="target error threshold", default=1e-10
  "--lambda0",          type=float, help="initial lambda value", type=float, default=1
  "--bound_norm_thres", type=float, help="threshold to drop the norm bound", type=float, default=1e-10
  "--log_level",        type=int,  help="console logger info level [verbose 0-6]", type=int, default=2
  "--output_type",      type=str,  help="output type selection: 'render', 'he_metric', 'vf_metric', 'param'
  "--output_format",    type=str,  help="output file format selection: 'png', 'pickle', 'obj'
```

### Compatibilty of output type/format
- `render`: render grid texture mapping
- `param`: turning the resulting metric into a map by using a layout of the flat mesh in the plane
- `he_metric`: save result mesh as halfedge structure and attached metric
- `vf_metric`: save result mesh as v, f matrix and attached metric
----
- `png`: compatible with `render` only
- `obj`: compatible with `param`
- `pickle`: compatible with `he_metric`, `vf_metric` and `param`


### Input data format

Input file is expected to be a triangular manifold mesh, possibly with multiple boundaries, in `.obj` format. And an additional text file with suffix `_Th_hat` containing (line by line) the prescribed angle per vertex. We attached three example input in `data/examples`. The compelete input data for our paper can be downloaded here: https://cims.nyu.edu/gcl/papers/2021-Conformal.zip, which also includes the camera configuration needed for rendering grid texture.
## Reproduction
Shell scripts to reproduce (via the python interface) the images and plots shown in the paper can be found in the figures subfolder.

## Citation
```
@article{Campen:2021:Conformal,
  title={Efficient and Robust Discrete Conformal Equivalence with Boundary},
  author={Campen, Marcel and Capouellez, Ryan and Shen, Hanxiao and Zhu, Leyi and Panozzo, Daniele and Zorin, Denis},
  journal={ACM Transactions on Graphics},
  volume={40},
  number={6},
  year={2021}}
```
