Here you can find information about installing and running the Goldilocks app.

## Web Application

The application is currently deployed on Streamlit community cloud and can be accessed here: https://goldilocks.streamlit.app/. In the near future it will be transferred to https://goldilocks.ac.uk/.

# Running locally

The application can be run locally whether in a Docker container, or in a python environment.

## Docker container

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

## Running in python virtual environment
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
* Install dependencies
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
