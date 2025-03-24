FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libopenmpi-dev openmpi-bin \
 && pip install --no-cache-dir \
    pandas numpy matplotlib requests mpi4py python-dotenv

WORKDIR /app
COPY . /app

CMD ["mpirun", "--allow-run-as-root", "-np", "10", "python", "monte_carlo_sim.py"]
