# __Benchmarking Graph__

__Requirement__
numpy-1.20.3, jax-0.2.24, jax-md-0.1.20, jaxlib-0.1.73, jraph-0.0.1.dev
create virtual environment using "requirements.txt" file

__How to run__
Run all spcripts from scripts directory
1. First generate data using Data generation files for:
    Spring system:    Spring-data.py, Spring-data-HGNN.py, Spring-data-FGNN.py and
    Pendulum system:  Pendulum-data.py, Pendulum-data-HGNN.py, Pendulum-data-FGNN.py

2. Run respective model using "sys.py" file and post simulation using "sys-post.py" file. (sys: simulation system)
