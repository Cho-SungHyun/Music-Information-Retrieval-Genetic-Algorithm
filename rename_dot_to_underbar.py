import os
import shutil

src_path = r"C:\Users\CAU_MI\Desktop\개인연구\music\music_data\MICM\all_rename\\"
dst_path = r"C:\Users\CAU_MI\Desktop\개인연구\music\music_data\MICM\wav\\"
file_list = os.listdir(src_path)


for folder in file_list:
    fname = folder.split('.wav')[0]
    # print(fname)
    fname2 = fname.replace('.','_')
    # print(fname2)
    rename = fname2 + '.wav'
    # print(rename)
    shutil.copy2(src_path+folder, dst_path+rename)
