#!/bin/bash
#SBATCH --job-name=FK_compile
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gp_debug
#SBATCH --account=nct_327

# Pulizia moduli pre-caricati
module purge

# Caricamento dei moduli necessari
module load cmake/3.25.1
module load oneapi/2023.2.0
module load hdf5

# Imposta la directory di dealii_minimal
export DEAL_II_DIR=$HOME/share/dealii
export CMAKE_PREFIX_PATH=$DEAL_II_DIR:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$DEAL_II_DIR/lib:$LD_LIBRARY_PATH

# Creazione della cartella di build e compilazione
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)




