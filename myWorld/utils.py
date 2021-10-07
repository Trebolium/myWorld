from re import I
import librosa, pysptk, copy, os, pdb
import numpy as np
import soundfile as sf
import pyworld as pw

# WORLD MANAGEAMENT

gamma = 0
mcepInput = 3  # 0 for dB, 3 for magnitude
alpha = 0.45
en_floor = 10 ** (-80 / 20)

def code_harmonic(sp, order):
    """Taken from https://github.com/seaniezhao/torch_npss/blob/master/data/data_util.py"""
    #get mel cepstrum analysis
    # the first arugment (pysptk.mcep) in the apply_along_axis function is a function itself. The second is the axis upon which the input array is sliced. Third is input array. Rest are arguments for pysptk.mcep 
    mceps = np.apply_along_axis(pysptk.mcep, 1, sp, order - 1, alpha, itype=mcepInput, threshold=en_floor)
    #do fft and take real
    scale_mceps = copy.copy(mceps)
    scale_mceps[:, 0] *= 2
    scale_mceps[:, -1] *= 2
    mirror = np.hstack([scale_mceps[:, :-1], scale_mceps[:, -1:0:-1]])
    mfsc = np.fft.rfft(mirror).real
    return mfsc

def get_world_feats(file_path, desired_sr, w_params, use_npss):
    """Process used from https://github.com/seaniezhao/torch_npss/blob/master/data/preprocess.py"""
    x, org_sr = sf.read(file_path)
    if org_sr != desired_sr:
        y = librosa.resample(x, org_sr, desired_sr)
    sr = desired_sr
    f0, t_stamp = pw.harvest(y, sr, w_params['fmin'], w_params['fmax'], w_params['frame_dur'])
    refined_f0 = pw.stonemask(y, f0, t_stamp, sr)
    spec_env = pw.cheaptrick(y, refined_f0, t_stamp, sr, f0_floor=w_params['fmin'])
    if use_npss == True:
        spec_env = code_harmonic(spec_env, w_params['num_feats'])
    else:
        spec_env = pw.code_spectral_envelope(spec_env, sr, w_params['num_feats'])
    aper_env = pw.d4c(y, refined_f0, t_stamp, sr)
    ap_env_reduced = pw.code_aperiodicity(aper_env, sr)
    return refined_f0, spec_env, aper_env, ap_env_reduced


### OS MANAGEMENT

def recursive_file_retrieval(parent_path):
    file_list = []
    parent_paths = [parent_path]
    more_subdirs = True
    while more_subdirs == True:
        subdir_paths = [] 
        for i, parent_path in enumerate(parent_paths):
            r,dirs,files = next(os.walk(parent_path)) 
            for f in files:
                file_list.append(os.path.join(r,f))
            # if there are more subdirectories
            if len(dirs) != 0:
                for d in dirs:
                    subdir_paths.append(os.path.join(r,d))
                # if we've finished going through subdirectories (each parent_path), stop that loop
            if i == len(parent_paths)-1:
                # if loop about to finish, change parent_paths content and restart loop
                if len(subdir_paths) != 0:
                    parent_paths = subdir_paths
                else:
                    more_subdirs = False
    return file_list

# takes a list of substrings and removes any entry the main list that contains these substrings
def substring_exclusion(main_list, exclude_list):
    filtered_list = [] 
    for f_path in main_list:
        exclusion_found = False
        for exclusion in exclude_list:
            if exclusion in f_path:
                exclusion_found = True
        if exclusion_found == False: 
            filtered_list.append(f_path)
    return filtered_list

def substring_inclusion(main_list, include_list):
    filtered_list = [] 
    for f_path in main_list:
        inclusion_found = False
        for inclusion in include_list:
            if inclusion in f_path:
                inclusion_found = True
        if inclusion_found == True: 
            filtered_list.append(f_path)
    return filtered_list

# ensures the total class number in a list is no more than the max allowance

def balance_class_list(main_list, class_list, max_occurances):
    class_counter_list = np.zeros(len(class_list))
    balanced_file_list = []
    for i, file in enumerate(main_list):
        for class_idx, class_name in enumerate(class_list):
            if class_name in file:
                if class_counter_list[class_idx] >= max_occurances:
                    break 
                class_counter_list[class_idx] += 1
                balanced_file_list.append(file)
    return balanced_file_list

def separate_into_groups(full_list, group_list):
    list_of_lists = [[] for i in range(len(group_list))]
    for s_idx, group in enumerate(group_list):
        this_group_list = []
        for file in full_list:
            if os.path.basename(file).startswith(group):
                this_group_list.append(file)
        list_of_lists[s_idx].extend(this_group_list)
    return list_of_lists