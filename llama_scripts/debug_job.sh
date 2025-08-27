#!/bin/bash
#SBATCH --job-name=debug_job
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=10
#SBATCH --constraint=a100
#SBATCH --time=02:00:00
#SBATCH --output=logs/debug_job%A_%a.out
#SBATCH --error=logs/debug_job%A_%a.err
#SBATCH --account conf-icl-2025.09.24-ghanembs


source /ibex/user/zhuw0b/miniforge/bin/activate /ibex/user/zhuw0b/conda-environments/verl_lora

export CODE_SERVER_CONFIG=~/.config/code-server/config.yaml
export XDG_CONFIG_HOME=$HOME/tmpdir
export CODE_SERVER_EXTENSIONS=/ibex/user/$USER/code-server/extensions
mkdir -p $CODE_SERVER_EXTENSIONS
PROJECT_DIR="$PWD"
ENV_PREFIX="$PROJECT_DIR"/env
PATH="$HOME/.local/bin:$PATH"



# setup ssh tunneling 
COMPUTE_NODE=$(hostname -s) 
CODE_SERVER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "
this is the port from SLURM ${SLURM_STEP_RESV_PORTS}
To connect to the compute node ${COMPUTE_NODE} on Ibex running your Code Server, 
you need to create an ssh tunnel from your local machine to login node on Ibex 
using the following command.

ssh -L localhost:${CODE_SERVER_PORT}:${COMPUTE_NODE}:${CODE_SERVER_PORT} ${USER}@glogin.ibex.kaust.edu.sa 

Next, you need to copy the url provided below and paste it into the browser 
on your local machine.

localhost:${CODE_SERVER_PORT}

" >&2

# launch code server
code-server --auth none --bind-addr ${COMPUTE_NODE}:${CODE_SERVER_PORT} --extensions-dir=${CODE_SERVER_EXTENSIONS} "$PROJECT_DIR"