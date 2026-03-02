#!/bin/bash
#SBATCH -t 1-0:00:00 # hours, min, sec
#SBATCH --job-name=gpu_perf_test
#SBATCH --ntasks=1              # Number of processes
#SBATCH --cpus-per-task=1       # Number of CPU cores per process
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --partition=V4V32_CAS40M192_L     # Partition/queue to run the job in. (REQUIRED)
#SBATCH -e slurm-%j.err  # Error file for this job.
#SBATCH -o slurm-%j.out  # Output file for this job.
#SBATCH -A gol_sjpo228_uksr  # Project allocation account name (REQUIRED)
#SBATCH --mail-type ALL    # Send email when job starts/ends
#SBATCH --mail-user ahya224@uky.edu   # Where email is sent to (optional)

echo "Job running on SLURM NODELIST: $SLURM_NODELIST "

# Activate conda environment
module purge
module load ccs/Miniforge3
source activate /project/sjpo228_uksr/AhmedHYassin/isthmus_env_all
which python

python bench_intersection_test.py --repeats 5 --ns 10 100 1e3 1e4 1e5 1e6 1e7 1e8