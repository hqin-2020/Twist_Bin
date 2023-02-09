#! /bin/bash

# epsilonarray=(0.1 0.05 0.01 0.005) #Computation of coarse grid and psi10.5
# fractionarray=(0.1 0.05 0.01 0.005)
# epsilonarray=(0.1) #Computation of coarse grid and psi10.5
# fractionarray=(0.1)

epsilonarray=(0.5) #Computation of coarse grid and psi10.5
fractionarray=(0.5)

# epsilonarray=(0.2 0.01) #Computation of coarse grid and psi10.5
# fractionarray=(0.2 0.01)

actiontime=1

python_name="singlecap_ex_newcab_newsigmaz.py"

maxiter=20000

# rhoarray=(0.667 0.750 1.00001 1.333 1.500)
# rhoarray=(0.800 0.850 0.900 0.950 1.00001 1.050 1.100 1.150 1.200 1.250 1.300)
# rhoarray=(0.800 0.850 0.900)
# rhoarray=(0.950 1.00001 1.050 1.100 1.150 1.200 1.250 1.300)
rhoarray=(1.00001 1.15 1.3)

for epsilon in ${epsilonarray[@]}; do
    for fraction in "${fractionarray[@]}"; do
        for rho in "${rhoarray[@]}"; do
            count=0
            # declare -n hXarr="$hXarri"

            # action_name="OneCapital_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_LR_${epsilon}"
            action_name="OneCapital_try_original"

            dataname="OneCapital_try_original_eps_${epsilon}_frac_${fraction}"

            # for delta in ${deltaarr[@]}; do
            #     for cearth in ${ceartharray[@]}; do
            #         for tauc in ${taucarray[@]}; do

            mkdir -p ./job-outs/${action_name}/eps_${epsilon}_frac_${fraction}/

            if [ -f ./bash/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}.sh ]; then
                rm ./bash/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}.sh
            fi

            mkdir -p ./bash/${action_name}/eps_${epsilon}_frac_${fraction}/

            touch ./bash/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}.sh

            tee -a ./bash/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}.sh <<EOF
#! /bin/bash

######## login
#SBATCH --job-name=${rho}
#SBATCH --output=./job-outs/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}.out
#SBATCH --error=./job-outs/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=3
#SBATCH --mem=1G
#SBATCH --time=03:00:00

####### load modules
module load python  gcc

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)
# perform a task

python3 -u /home/hqin/Twist_Bin/$python_name  --rho ${rho} --epsilon ${epsilon}  --fraction ${fraction}   --maxiter ${maxiter} --dataname ${dataname} --figname ${dataname}
echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
            count=$(($count + 1))
            sbatch ./bash/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}.sh
            #                 done
            #             done
        done
    done
done
