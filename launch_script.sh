#!/bin/sh

#SBATCH --job-name=svm_biogen_short_covariates
#SBATCH --gres=gpu:0
#SBATCH --mem=50000
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=ADAPT-03
#SBATCH --partition=MEDIUM-G1

. /home/ICTDOMAIN/d18129068/dnam_biomarker_generator/dnam_biogen/bin/activate
#source /home/ICTDOMAIN/d18129068/dnam_biomarker_generator/myenv/bin/activate

if [ $? -ne 0 ]; then
    echo "Error activating virtual environment."
    exit 1
fi

# Check if the virtual environment is activated
python - <<EOF
import sys

# Check if sys.prefix points to a virtual environment directory
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("Virtual environment detected.")
else:
    print("Not running in a virtual environment.")
EOF

python3 main.py "/home/ICTDOMAIN/d18129068/dnam_biomarker_generator/static/ibd_pypi_dnam_age_sex_bcc_short.pkl" "/home/ICTDOMAIN/d18129068/dnam_biomarker_generator/results/dnam_age_sex_bcc/short/" "SVM"
