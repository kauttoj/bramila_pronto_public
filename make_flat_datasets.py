# This script creates flattened fMRI datasets starting from NIFTI 4D data, data mask and spatial parcellations.
# After this script, you may proceed with "make_aligned_datasets" script that creates additional RSRM aligned datasets.
#
# 15.5.2019 Janne K.

import numpy as np
import nibabel as nib
import pickle
import time
from joblib import Parallel, delayed
import os
from scipy.stats import zscore
import deepdish as dd
import itertools

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def process_single_file(args):
    # make zscored numpy data for one subject
    if not os.path.isfile(args['target']) or overwrite_old==1:
        #print('loading nifti')
        proxy_img = nib.load(args['source'])
        data = proxy_img.get_fdata(dtype=np.float32)
        #print('reshaping')
        assert data.shape[0:3] == mask_size,'data and mask size mismatch for file %s' % args['source']
        data = np.reshape(data, (n_total_voxels, data.shape[3]),order='F')
        data = data[mask_img_flat_ind, :]
        #print('zscoring')
        data = zscore(data, axis=1)
        #print('saving npy')
        payload = {'data':data,'ID':args['ID'],**common_dict}
        dd.io.save(args['target'],payload, compression=None)
    assert os.path.isfile(args['target']), 'File %s was not produced!' % args['target']

# Auto Run
if __name__ == "__main__":

    print('--- STARTING DATA PROCESSING ---', flush=True)

    # enter required parameters below
    allsubs = ['sub-08','sub-09','sub-10','sub-11','sub-12','sub-13','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20','sub-21','sub-22','sub-23','sub-24','sub-25','sub-26','sub-27','sub-28','sub-29','sub-30','sub-31','sub-32','sub-33','sub-35','sub-36','sub-37','sub-38','sub-39','sub-40','sub-41','sub-42','sub-43','sub-44','sub-45','sub-46','sub-47','sub-48','sub-49','sub-50','sub-51','sub-52','sub-53','sub-54','sub-55','sub-56','sub-57','sub-58']
    allruns = [1, 2, 3, 4, 5]
    overwrite_old = 1
    n_workers = 10

    alltemplates = [
        '3mm_Shirer2012_n14.nii',
        '3mm_BN_Atlas_246.nii',
        '3mm_aal.nii',
        '3mm_HarvardOxford_2mm_th25_TOTAL.nii',
        '3mm_ICA_n70.nii',
        '3mm_Gordon_Parcels_MNI_333.nii',
        '3mm_MNI-maxprob-thr25-1mm.nii',
        '3mm_Schaefer2018_100Parcels.nii',
        '3mm_bm20_grey30.nii',
        '3mm_cambridge_basc_multiscale_sym_scale064.nii',
        '3mm_combined_networks_mask.nii',
        '3mm_iCAP_20.nii',
        '3mm_Gordon_Parcels_MNI_333.nii']
    alltemplates_label = [x[4:-4] for x in alltemplates]

    ### DEBUGGING WITH SMALLER DATA ##############
    #allsubs = allsubs[0:3]
    #allruns = allruns[0:2]
    ############################
    if 0:
        alltemplates = [r'D:/WORK/emotion_project/3mm_parcellations/'+x for x in alltemplates]
        save_folder = r'D:/WORK/emotion_project/data/'
        data_root = r'D:/WORK/emotion_project/'
         # mask of brain, no analyzing beyond this, boolean
        brain_mask_file = r'D:/WORK/emotion_project/3mm_parcellations/3mm_mni_icbm152_t1_tal_nlin_asym_09a_mask.nii'
        # grey mask file, float
        grey_mask_file = r'D:/WORK/emotion_project/3mm_parcellations/3mm_mni_icbm152_gm_tal_nlin_asym_09a.nii'
        # data and subject-specific analysis mask, boolean
        analysis_mask_file = r'D:/WORK/emotion_project/grand_analysis_mask.nii'
    else:
        alltemplates = ['/m/nbe/scratch/empathy_and_emotions/Janne_analysis/3mm_parcellations/'+x for x in alltemplates]
        save_folder = r'/m/nbe/scratch/empathy_and_emotions/Janne_analysis/new_analysis_Feb2019/hyperalign/data/'
        data_root = r'/m/nbe/scratch/empathy_and_emotions/Janne_analysis/new_analysis_Feb2019/bramila_processed_final/'
         # mask of brain, no analyzing beyond this, boolean
        brain_mask_file = r'/m/nbe/scratch/empathy_and_emotions/Janne_analysis/3mm_parcellations/3mm_mni_icbm152_t1_tal_nlin_asym_09a_mask.nii'
        # grey mask file, float
        grey_mask_file = r'/m/nbe/scratch/empathy_and_emotions/Janne_analysis/3mm_parcellations/3mm_mni_icbm152_gm_tal_nlin_asym_09a.nii'
        # data and subject-specific analysis mask, boolean
        analysis_mask_file = r'/m/nbe/scratch/empathy_and_emotions/Janne_analysis/new_analysis_Feb2019/bramila_processed_final/grand_analysis_mask.nii'

    ## computations start, do not modify below
    print('making mask indices')

    # global anatomical brain mask, not data outside this!
    mask_img = (nib.load(brain_mask_file).get_fdata()) > 0
    n_mask_voxels = np.count_nonzero(mask_img)
    mask_size = mask_img.shape
    n_total_voxels = np.prod(mask_size)

    mask_img_flat = np.reshape(mask_img,n_total_voxels, order='F')
    mask_img_flat = mask_img_flat>0
    mask_img_flat_ind = np.where(mask_img_flat)[0]

    assert len(mask_img_flat_ind) == n_mask_voxels,'mask voxel count mismatch'
    assert 0.20 < n_mask_voxels/n_total_voxels < 0.40,'grand mask voxel ratio (%.2f) is weird!' % (n_mask_voxels/n_total_voxels)

    arr = tuple(range(1,len(mask_img_flat_ind)+1))
    mask_img_numbered = np.zeros(n_total_voxels,dtype=np.int32)
    mask_img_numbered[mask_img_flat_ind] = arr
    mask_img_numbered = np.reshape(mask_img_numbered,mask_size,order='F')

    assert np.count_nonzero(np.logical_and(mask_img_numbered>0,mask_img))==n_mask_voxels

    print('making mask coordinate matrix')

    mask_img_coord = np.zeros((len(arr),3),dtype=np.int32)
    for k in range(len(arr)):
        mask_img_coord[k,:] = np.where(mask_img_numbered==arr[k])

    alltemplates = [grey_mask_file,analysis_mask_file] + alltemplates
    alltemplates_label = ['grey_mask','analysis_mask'] + alltemplates_label

    template_mask_data = np.zeros((n_mask_voxels,len(alltemplates)),dtype=np.float32)

    n_rois = []
    for template_k in range(len(alltemplates)):

        templatemask_file = alltemplates[template_k]
        templatemask_name = alltemplates_label[template_k] # keep unique for each template

        atlas_img = nib.load(templatemask_file).get_fdata()

        assert mask_size == atlas_img.shape,'atlas size mismatch!'

        mask_img[mask_img<0]=0
        mask_img[np.isnan(mask_img)]=0
        if template_k>0:
            atlas_img = np.round(atlas_img)

        atlas_img_flat = np.reshape(atlas_img,n_total_voxels, order='F')
        template_mask_data[:,template_k] = atlas_img_flat[mask_img_flat_ind]

        arr = np.unique(template_mask_data[:,template_k])
        n_rois.append(len(arr[arr>0]))

    common_dict = {'parcellation_data':template_mask_data,'parcellation_label':alltemplates_label,'mask_img_flat_ind':mask_img_flat_ind,'mask_img_numbered':mask_img_numbered,'mask_img_coord':mask_img_coord,'mask_size':mask_size,'parcellation_roicount':n_rois}

    n_sub = len(allsubs)
    n_runs = len(allruns)

    print('making argument lists')
    arg_instances=[]
    for run in allruns:
        for sub in allsubs:
            for smooth in ['','_smoothed']:
                target = '%sRUN%i_%s_data%s.h5' % (save_folder,run, sub,smooth)
                source = '%sRUN%i/bramila/%s_mask_detrend_fullreg_filtered%s.nii' % (data_root,run, sub,smooth)
                assert os.path.exists(source),'file not found: %s' % source
                ID = sub + '_run' + str(run) + smooth
                args = {'ID':ID,'source':source,'target':target}
                arg_instances.append(args)

    print('Total %i tasks created, starting file conversion with %i workers' % (len(arg_instances),n_workers))

    start_time = time.time()
    if n_workers>1:
        # parallel
        Parallel(n_jobs=n_workers)(map(delayed(process_single_file), arg_instances))
    else:
        # serial
        for i,args in enumerate(arg_instances):
            print('...file %i or %i' % (i+1,len(arg_instances)))
            process_single_file(args)

    print('all done (took %.1fs)' % (time.time() - start_time), flush=True)