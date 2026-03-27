root_path="ROOT-PATH"
save_folder=${root_path}/exps/mat-sed/run
config_folder=${root_path}/config/mat-sed

CUDA_NUM=0,1,2,3
export CUDA_VISIBLE_DEVICES=$CUDA_NUM
dir1="${root_path}/pretrained_model"
dir2="${save_folder}/finetune1_deepcot"
dir3="${save_folder}/finetune2_deepcot"

# check GPU memory

# pretrain
#cd ${root_path}/recipes/mlm
#mkdir -p $dir1
#python main.py --multigpu=True --random_seed=True --config_dir="${config_folder}/pretrain.yaml" --save_folder=$dir1
#sleep 60

cd ${root_path}/recipes/finetune

# finetune2
#source ${root_path}/scripts/mem_check.sh 20000
mkdir -p $dir3
cp "${save_folder}/finetune1_base/best_teacher.pt" "$dir3/best_student.pt"
cp "${save_folder}/finetune1_base/best_teacher.pt" "$dir3/best_teacher.pt"
echo "Running python main.py --multigpu=True --random_seed=True --config_dir="${config_folder}/finetune2.yaml" --save_folder=$dir3 --continual=True"
python main.py --multigpu=True --random_seed=True --config_dir="${config_folder}/finetune2.yaml" --save_folder=$dir3 --continual=True
