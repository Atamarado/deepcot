root_path="ROOT-PATH"
save_folder=${root_path}/exps/mat-sed/run
config_folder=${root_path}/config/mat-sed

CUDA_NUM=0,1,2,3
export CUDA_VISIBLE_DEVICES=$CUDA_NUM
dir1="${root_path}/pretrained_model"
dir2="${save_folder}/finetune1_base_urban"

# check GPU memory

cd ${root_path}/recipes/finetune

# finetune1
#source ${root_path}/scripts/mem_check.sh 10000
mkdir -p $dir2
cp "$dir1/passt-s-f128-p16-s10-ap.476-swa.pt" "$dir2/best_student.pt"
echo "Running python main.py --multigpu=True --random_seed=True --config_dir="${config_folder}/finetune1_urban.yaml" --save_folder=$dir2 --continual=False --dataset=urban"
python main.py --multigpu=True --random_seed=True --config_dir="${config_folder}/finetune1_urban.yaml" --save_folder=$dir2 --continual=False --dataset=urban
