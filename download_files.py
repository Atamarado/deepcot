import os
import gdown

os.makedirs("CoOadTR/data/thumos_anet", exist_ok=True)
os.makedirs("CoOadTR/data/thumos_kin", exist_ok=True)

gdown.download("https://drive.google.com/file/d/11kIx5jU_FOewTZi63eWwo7cLU9dSOUD3/view?usp=sharing", "CoOadTR/data/thumos_anet/OadTR_THUMOS.zip", fuzzy=True)
gdown.download("https://drive.google.com/file/d/1DdL72X5ZV61BFoveJyWB7I2K1eW4YCVj/view?usp=sharing", "CoOadTR/data/thumos_kin/OadTR_THUMOS_Kinetics.zip", fuzzy=True)

gdown.download("https://drive.google.com/file/d/1-yzDy5y57UM21BSSfRQpznyMz4F47K9q/view?usp=sharing", "roformer_models.zip", fuzzy=True)