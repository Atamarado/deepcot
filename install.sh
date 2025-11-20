#!/bin/bash -i

echo "Preparing conda envionrment..."
conda env create -f env.yml -n deepcot -y
conda activate deepcot

echo "Downloading necessary files"
python download_files.py
unzip CoOadTR/data/thumos_anet/OadTR_THUMOS.zip -d CoOadTR/data/thumos_anet
rm CoOadTR/data/thumos_anet/OadTR_THUMOS.zip
unzip CoOadTR/data/thumos_kin/OadTR_THUMOS_Kinetics.zip -d CoOadTR/data/thumos_anet
rm CoOadTR/data/thumos_kin/OadTR_THUMOS_Kinetics.zip
unzip roformer_models.zip
rm roformer_models.zip
echo "Done"

