Here you can find information about installing and running the Goldilocks app.

## Web Application

The application is currently deployed on Streamlit community cloud and can be accessed here: [https://goldilocks.streamlit.app/](https://goldilocks.streamlit.app/). 

### Running locally
The application can be run locally whether in a Docker container, or in a python environment.

#### Docker container
* Make sure that Docker is installed
* Clone the repository
```
git clone https://github.com/stfc/goldilocks.git
cd goldilocks
```
* Build a docker image from the project folder
```
docker build -t goldilocks-app .
```
* Run the container
```
docker run -p 8501:8501 goldilocks-app
```
* Open the browser and go to
```
http://localhost:8501
```
#### Running in python virtual environment
* Install Python (>=3.10,<3.13)
* Install Poetry
* Create a clean environment (here it is done with venv, but conda, etc. can also be used)
```
python -m venv .venv
source .venv/bin/activate
```
* Clone the repository
```
git clone https://github.com/stfc/goldilocks.git
cd goldilocks
```
* Install PyTorch, torch-scatter, torch-sparse, torch-cluster, torch-spline-conv first (required for torch-geometric) before running poetry install. Pytorch Geometric requires binary wheels for torch_scatter, torch_sparse, torch_cluster, torch_spline_conv. These cannot be installed by Poetry automatically, so they must be installed first.
   * Install PyTorch
      * CPU-only:
      ```
      pip install torch --index-url https://download.pytorch.org/whl/cpu
      ```
      * CUDA-enabled (example: cu124):
      ```
      pip install torch --index-url https://download.pytorch.org/whl/cu124
      ```
   * Install PyTorch Geometric dependencies
      * CPU wheels:
      ```
      pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
            -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
      ```
      * CUDA wheels (example: cu124):
      ```
      pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
         -f https://data.pyg.org/whl/torch-2.7.0+cu124.html
      ```
      * Install torch_geometric
      ```
      pip install torch_geometric
      ```
* Install remaining dependencies with Poetry
```
poetry install
```
* Run the application
```
streamlit run src/qe_input/QE_input.py
```
* Open a browser and go to
```
http://localhost:8501
```