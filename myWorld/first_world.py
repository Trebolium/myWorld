import os, pdb, pickle, yaml, time, sys, argparse
from re import I
import pyworld as pw
import soundfile as sf
import numpy as np
from tqdm import tqdm
from params_data import *
from utils import recursive_file_retrieval, substring_exclusion, substring_inclusion, balance_class_list, get_world_feats, separate_into_groups

"""Script for collecting wav files from directory,
and converting them using WORLD functions. See help(pw) in interactive iPython to know more about these functions"""

def str2bool(v):
    return v.lower() in ('true')

def path_list_by_rules(rootDir, exclude_list, class_list):
    fileList = recursive_file_retrieval(rootDir)
    classes_only_list = substring_exclusion(fileList, exclude_list) 
    classes_only_list = substring_inclusion(classes_only_list, class_list)
    classes_only_list  = [f for f in classes_only_list if f[-6] == '_'] # to ensure that the format of the file ends with '_' and a vowel
    files_by_singers = separate_into_groups(classes_only_list, singer_list)
    total_balanced_list = []
    for f_by_singer in files_by_singers:
        total_balanced_list.extend(balance_class_list(f_by_singer, class_list, 10))
    return total_balanced_list 

if __name__ == '__main__':

    #define audio directory
    rootDir = '/import/c4dm-datasets/VocalSet1-2/data_by_singer'
    singer_list = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10','m11','f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']
    class_list=['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
    exclude_list = ['caro','row','long','dona']
    original_audio_path_list = []
    parser = argparse.ArgumentParser(description='params for converting audio to spectral using world', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n','--use_npss', type=str2bool, default=False)
    parser.add_argument('-o','--overwrite_dst_files', type=str2bool, default=False) 
    parser.add_argument('-d','--dst_dir', type=str)
    config = parser.parse_args()
    dst_dir = '/homes/bdoc3/my_data/world_vocoder_data/' +config.dst_dir
    use_npss = config.use_npss

    original_audio_path_list = path_list_by_rules(rootDir, exclude_list, class_list)
    w_param = {"fmin":50, "fmax":1100, 'num_feats':40, 'frame_dur':mel_window_step} # pw.default_frame_period is 5ms. We've changed the frame_dur to 10 ms
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    file = open(os.path.join(dst_dir, 'w_params.yaml'), 'w')
    yaml.dump(w_param, file)
    file.close()

    for file_path in tqdm(original_audio_path_list):
        dst_path = dst_dir +'/' +os.path.basename(file_path)[:-4] +'.pkl'
        if os.path.exists(dst_path):
            if config.overwrite_dst_files == False:
                print(f'{dst_path} already exists. Skipping')
                continue
        time_start = time.time()
        refined_f0, comp_sp, aper_env, comp_ap = get_world_feats(file_path, sampling_rate, w_param, use_npss=use_npss)
        if np.isnan(comp_sp).any() == True:
            print(f'nan found is comp_sp for file: {file_path}')
            pdb.set_trace() 
        with open(dst_path, 'wb') as f:
            pickle.dump((refined_f0, comp_sp, aper_env, comp_ap), f)
        print(f'{os.path.basename(file_path)} has been converted in {time.time() - time_start} seconds and saved')