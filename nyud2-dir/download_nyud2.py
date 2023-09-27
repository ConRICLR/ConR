
#####################################################################################
# Code is based on the Balanced MSE (https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Balanced_MSE_for_Imbalanced_Visual_Regression_CVPR_2022_paper.html) implementation
# from https://github.com/jiawei-ren/BalancedMSE/tree/main/nyud2-dir by Jiawei Ren
####################################################################################
import os
import gdown
import zipfile

print("Downloading and extracting NYU v2 dataset to folder './data'...")
data_file = "./data.zip"
gdown.download("https://drive.google.com/uc?id=1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw")
print('Extracting...')
with zipfile.ZipFile(data_file) as zip_ref:
    zip_ref.extractall('.')
os.remove(data_file)
print("Completed!")