import os
import pdb, pickle, yaml
import pyworld as pw
import soundfile as sf
from utils import recursive_file_retrieval, substring_exclusion, substring_inclusion, get_world_feats

# def code_harmonic(sp, order):
#     """Taken from data_util.py in repo https://github.com/seaniezhao/torch_npss"""
#     #get mcep
#     mceps = np.apply_along_axis(pysptk.mcep, 1, sp, order - 1, alpha, itype=mcepInput, threshold=en_floor)
#     #do fft and take real
#     scale_mceps = copy.copy(mceps)
#     scale_mceps[:, 0] *= 2
#     scale_mceps[:, -1] *= 2
#     mirror = np.hstack([scale_mceps[:, :-1], scale_mceps[:, -1:0:-1]])
#     mfsc = np.fft.rfft(mirror).real
#     return mfsc

#define audio directory
rootDir = '/import/c4dm-datasets/VocalSet1-2/data_by_singer'
dst_dir = '/homes/bdoc3/my_data/world_vocoder_data/vocalSet_no_songs'
class_list=['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
exclude_list = ['caro','row','long','dona']
original_audio_path_list = []

if __name__ == "__main__":
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    w_param = {"fmin":50, "fmax":1100, 'num_feats':60, 'frame_dur':pw.default_frame_period}
    file = open(os.path.join(dst_dir, 'w_params.yaml'), 'w')
    yaml.dump(w_param, file)
    file.close()

    all_file_paths = recursive_file_retrieval(rootDir)
    filtered_file_paths = substring_exclusion(all_file_paths, exclude_list)
    filtered_file_paths = substring_inclusion(filtered_file_paths, class_list)
    filtered_file_paths = [f for f in filtered_file_paths if not os.path.basename(f).startswith('.')]
    filtered_file_paths = [f for f in filtered_file_paths if os.path.basename(f).endswith('wav')]
    for file_path in filtered_file_paths:
        x, sr = sf.read(file_path)
        refined_f0, comp_sp, comp_ap = get_world_feats(x, sr, w_param) 
        dst_path = dst_dir +'/' +os.path.basename(file_path)[:-4] +'.pkl'
        with open(dst_path, 'wb') as f:
            pickle.dump((refined_f0, comp_sp, comp_ap), f)
        print(f'{file_path} has been converted and saved')
