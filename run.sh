#! /bin/bash

# epsilonarray=(0.1 0.05 0.01 0.005) #Computation of coarse grid and psi10.5
# fractionarray=(0.1 0.05 0.01 0.005)
epsilonarray=(0.1 0.01 0.001) #Computation of coarse grid and psi10.5
fractionarray=(0.1)

# epsilonarray=(0.01) #Computation of coarse grid and psi10.5
# fractionarray=(0.01)

# epsilonarray=(0.2 0.01) #Computation of coarse grid and psi10.5
# fractionarray=(0.2 0.01)

actiontime=1

# python_name="singlecap_ex.py"
# python_name="singlecap_ex_newcab.py"
python_name="singlecap_ex_newcab_newsigmaz.py"
# python_name="singlecap_ex_newcab_newsigmaz2.py"

maxiter=200000

# rhoarray=(0.667 0.750 1.00001 1.333 1.500)
# rhoarray=(0.800 0.850 0.900 0.950 1.00001 1.050 1.100 1.150 1.200 1.250 1.300)
# rhoarray=(0.800 0.850 0.900)
rhoarray=(0.650 0.700 0.750 0.800 0.850 0.900 0.950 1.00001 1.050 1.100 1.150 1.200 1.250 1.300 1.350 1.400 1.450 1.500)

# ellarray=(3.70 3.71 3.72 3.73 3.74 3.75 3.76 3.77 3.78 3.79 3.80 3.81 3.82 3.83 3.84 3.85 3.86 3.87 3.88 3.89 3.9)
# ellarray=(3.50 3.60 3.70 3.80 3.90 4.00)
# ellarray=(2.50 2.60 2.70 2.80 2.90 3.0 3.10 3.20 3.30 3.40 4.10 4.20 4.30 4.40 4.50 4.60 4.70 4.80 4.90 5.00)
# ellarray=(1.00 1.10 1.20 1.30 1.40 1.50 1.60 1.70 1.80 1.90 2.00 2.10 2.20 2.30 2.40)
ellarray=(1.0)
for epsilon in ${epsilonarray[@]}; do
    for fraction in "${fractionarray[@]}"; do
        for rho in "${rhoarray[@]}"; do
            for ell in "${ellarray[@]}"; do
                count=0

                action_name="OneCapital_newcab_newgrid_morepts_opt"

                dataname="${action_name}_${epsilon}_frac_${fraction}"

                mkdir -p ./job-outs/${action_name}/eps_${epsilon}_frac_${fraction}/

                if [ -f ./bash/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}_ell_${ell}.sh ]; then
                    rm ./bash/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}_ell_${ell}.sh
                fi

                mkdir -p ./bash/${action_name}/eps_${epsilon}_frac_${fraction}/

                touch ./bash/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}_ell_${ell}.sh

                tee -a ./bash/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}_ell_${ell}.sh <<EOF
#! /bin/bash

######## login
#SBATCH --job-name=${rho}_${ell}
#SBATCH --output=./job-outs/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}_ell_${ell}.out
#SBATCH --error=./job-outs/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}_ell_${ell}.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=12:00:00

####### load modules
module load python  gcc

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)
# perform a task

python3 -u /project/lhansen/Twist_Bin/$python_name  --rho ${rho} --ell ${ell} --epsilon ${epsilon}  --fraction ${fraction}   --maxiter ${maxiter} --dataname ${dataname} --figname ${dataname}
echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
                count=$(($count + 1))
                sbatch ./bash/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}_ell_${ell}.sh
            done
        done
    done
done
