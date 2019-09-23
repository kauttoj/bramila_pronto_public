% script to run "bramila_pronto_run.m" regression/classification code
% Before running this, you need to have fMRI data in flat & parcellated .h5 files
% using "bramila_RSRM_aligner.py" Python code.

clearvars;
close all;
clc;
warning('on','all') % turn all warnings
%addpath('/m/nbe/scratch/empathy_and_emotions/shared_codes');
addpath('D:\JanneK\Documents\git_repos\bramila_pronto\shared_codes');

params = struct(); % create empty parameter set
load('bramila_pronto_defaultparams.mat'); % load default params (to be overwritten) OPTIONAL

%% input data X (neural, structural, etc)
% all 50 subjects
%params.allsubs = {'sub-08', 'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17','sub-18', 'sub-19', 'sub-20', 'sub-21', 'sub-22', 'sub-23', 'sub-24', 'sub-25', 'sub-26', 'sub-27','sub-28', 'sub-29', 'sub-30', 'sub-31', 'sub-32', 'sub-33', 'sub-35', 'sub-36', 'sub-37', 'sub-38','sub-39', 'sub-40', 'sub-41', 'sub-42', 'sub-43', 'sub-44', 'sub-45', 'sub-46', 'sub-47', 'sub-48','sub-49', 'sub-50', 'sub-51', 'sub-52', 'sub-53', 'sub-54', 'sub-55', 'sub-56', 'sub-57', 'sub-58'};
% dropped 3 heavy movers, 47 remaining              
params.allsubs = {'sub-08','sub-09','sub-10','sub-11','sub-12','sub-13','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20','sub-21','sub-22','sub-23','sub-24','sub-25','sub-26','sub-28','sub-29','sub-30','sub-32','sub-33','sub-35','sub-36','sub-37','sub-38','sub-39','sub-40','sub-42','sub-43','sub-44','sub-45','sub-46','sub-47','sub-48','sub-49','sub-50','sub-51','sub-52','sub-53','sub-54','sub-55','sub-56','sub-57','sub-58'};

%%--------- Define data and target -----------------------
params.emotion = 'sadness'; % source data type ("X")
params.target = 'FFNI_LackofEmpathy'; % response label ("Y")
%%--------------------------------------------------------

%% input data Y (behavioral response)
params.Y = get_scores(params.allsubs,params.target);
params.problem_type = 'regression';
params.max_memory_per_cpu = 2000; % how large dataset to keep in memory vs. loading from disk (in megabytes)

%params.output_folder = ['/m/nbe/scratch/empathy_and_emotions/Janne_analysis/toolboxes/bramila_pronto/TEMP/',params.emotion,'_',params.target];
params.output_folder = ['D:\JanneK\Documents\git_repos\bramila_pronto\TEMP\',params.emotion,'_',params.target];

%% files and frames for functional data, choose which type of input data (precomputed .h5 or .nii)
if 0 
    %% content of precomputed .h5 files:
    % /parcellation_roicount = vector of ROIs per parcellation
    % /parcellation_label = label of parcellation
    % /mask_size = 3D size of original mask
    % /parcellation_data = 2D matrix with voxels x #parcellations
    % /mask_img_numbered = ?
    % /mask_img_flat_ind = ?
    % /mask_img_coord = x, y, z coordinate of each voxel        
    fprintf('Input data type is precomputed H5\n');
    params.data_root_folder = '/m/nbe/scratch/empathy_and_emotions/Janne_analysis/new_analysis_Feb2019/hyperalign/data/';    
    params.allruns = 1:5;
    % Recommended for faster and more flexible analysis
    for sub = 1:length(params.allsubs)
        for run = 1:length(params.allruns)
            if sub==1 && run==1
                params.X.functional.file=[];
                params.X.functional.frames=[];
            end
            params.X.functional.files{sub}{run} = sprintf('%sRUN%i_%s_data_smoothed.h5',params.data_root_folder,params.allruns(run),params.allsubs{sub});
            assert(exist(params.X.functional.files{sub}{run},'file')>0,'file not found: %s',params.X.functional.files{sub}{run});
            if sub==1
                params.X.functional.frames{sub}{run} = get_fmri_frames(params.allruns(run),params.emotion);
            else
                params.X.functional.frames{sub}{run} = params.X.functional.frames{1}{run};
            end
        end
    end
    params.X.functional.do_windowed_zscoring = 1; % do within-window z-scoring after obtaining frames (kills baseline, only dynamics remain)
    params.X.functional.do_temporal_averaging = 0;
    params.X.functional.source_type = 'h5'; % precomputed h5 files    
    %% parcellation mask (assume subjects in same space)
    % NEED TO SPECIFY LABELS MANUALLY, CANNOT READ PYTHON LIST DIRECTLY (ugly!)
    % Note: No effect when using NIFTI source
    params.parcellation_label = {'grey_mask','analysis_mask','Shirer2012_n14','BN_Atlas_246','aal','HarvardOxford_2mm_th25_TOTAL','ICA_n70','Gordon_Parcels_MNI_333','MNI-maxprob-thr25-1mm','Schaefer2018_100Parcels','bm20_grey30','cambridge_basc_multiscale_sym_scale064','combined_networks_mask','iCAP_20','Gordon_Parcels_MNI_333'};
    params.parcellation_function = '(mask(1,:)>0.10).*(mask(2,:)>0).*(mask(3,:))'; % customize parcellation via logical operations
    params.min_voxel_per_ROI = 3*3*3; % for 3mm voxels 9x9x9=729mm^3    
    params.X.functional.preprocessor = 'None'; % is None, using precomputed kernels (fast!)
else
    % If prefer Nifti files, does not allow functional alignment analysis
    % Need to specify frames manually for the input files!
    fprintf('Input data type is Nifti\n');
    params.allruns = 1; % already concatenated
    for sub = 1:length(params.allsubs)
        for run = 1:length(params.allruns)
            if sub==1 && run==1
                params.X.functional.file=[];
                params.X.functional.frames=[];
            end                
            % MEAN:
            %params.X.functional.files{sub}{run} = sprintf('D:/JanneK/Documents/git_repos/bramila_pronto/data/nifti/%s_%s_mean.nii',params.allsubs{sub},params.emotion);
            % CONCATENATED:
            params.X.functional.files{sub}{run} = sprintf('D:/JanneK/Documents/git_repos/bramila_pronto/data/nifti/%s_%s_concat.nii',params.allsubs{sub},params.emotion);            
            assert(exist(params.X.functional.files{sub}{run},'file')>0,'file not found: %s',params.X.functional.files{sub}{run});            
            % NOTE: Assume same frames for all subjects!
            if sub==1
                params.X.functional.frames{sub}{run} = 1:250;
            else
                % same as for the first subject
                params.X.functional.frames{sub}{run} = params.X.functional.frames{1}{run};
            end
        end
    end
    params.X.functional.do_windowed_zscoring = 1; % do within-window z-scoring after obtaining frames (removes baseline)
    params.X.functional.do_temporal_averaging = 0; % average data temporally (from 4D to 3D). Cannot combine with z-scoring, baseline must remain non-zero!
    params.X.functional.parcellation_mask = 'D:\JanneK\Documents\git_repos\bramila_pronto\data\nifti\combined_mask_PRONTO.nii';        
    params.X.functional.data_mask = 'D:\JanneK\Documents\git_repos\bramila_pronto\data\grand_analysis_mask.nii';   
    params.X.functional.source_type = 'nifti'; % precomputed 4D files    
    params.X.functional.preprocessor = 'StandardScaler'; % is None, using precomputed kernels (fast!)
end

% Next we add additional modalities. Each must contain only one data-array (samples x features), we also need to save total number of elements in the data.
% If more complex data is needed, e.g., with multiple ROIs, one needs to add special case for that into "bramila_pronto_run" data loader.

if 1
    %% structural data with FreeSurfer estimated volume sizes
    data = single(get_structural(params.allsubs));
    elements = numel(data);
    file = sprintf('%sstructural_data.mat',params.output_folder);
    save(file,'data','elements','-v7.3');
    params.X.structural.files = {file}; % enclose in cell
    params.X.structural.preprocessor = 'StandardScaler'; % is none, skipping all transformations (very  fast computations)
end
if 0
    %% CHEAT by giving the responses! Only for sanity checking, never for any real analysis!
    warning('!!! Adding responses as input, this should give almost perfect accuracy. ONLY FOR TESTING, NEVER FOR REAL ANALYSIS !!!');
    data = single(params.Y + 0.001*rand(size(params.Y))); % add little noise
    elements = numel(data);
    file = sprintf('%sCHEAT_data.mat',params.output_folder);
    save(file,'data','elements','-v7.3');
    params.X.cheatkernel.files = {file}; % enclose in cell
    params.X.cheatkernel.preprocessor = 'StandardScaler'; % is none, skipping all transformations (very fast computations)
end

%% resting state data
%data = get_resting_state(params.allsubs);
%file = sprintf('%srestingstate_data.mat',params.output_folder);
%save(file,'data');
%params.X.restingstate = file;

%% physiological data, eyetracking, etc.
% .....

%% learner algorithm and params
if 1
    predictor_params.model = 'simpleMKL';% currenly available 'bayesMKL' and 'simpleMKL';
    predictor_params.args.C = [0.01,0.1,1,10,100,1000,10000]; % for simpleMKL
    %predictor_params.opts.svmreg_epsilon = 0.1; % default 0.01
else    
    predictor_params.model = 'bayesMKL';%
    predictor_params.args = []; % no tunable hyperparameters for bayesMKL!
end
params.predictor = predictor_params;

%% kernel type or types (pooled together)
% parameters that follow string after '_'
%  order of polynomial (polynomial kernel)
%  gamma scaling factor (radial kernel) in addition to standard 1/#features -scaling
params.kernels = {'linear','polynomial_2','radial_1'};
params.kernel_normalization = 'CenterAndNormalize'; % 'CenterAndNormalize' (recommended! Pronto-style), 'CenterAndTrace', 'Center' or 'None'

%% cross-validation of the model (main loop)
params.cv = 'LOSO'; % integer or string 'LOSO' (Leave-One-Subject-Out)

%% internal cross-validation within the main loop
params.internal_cv = 10; % NOTE: Model will be trained total params.cv*params.internal_cv times, keep this at 5-15
params.internal_cv_skipFeatureTraining = 1; % omit feature transformer re-training. Faster, but can be suboptimal! Disable, if possible.

%% how many workers to use (divided over folds), ideally equal or multiple of params.cv
params.workers = 4; % use 1 for debugging

%% local tempdata folder
params.tempdata_root = 'C:\Users\JanneK\AppData\Local\Temp\04_24_2019_11_34_03_bramila_pronto\';%'/tmp/04_08_2019_15_04_29_bramila_pronto/';
params.tempdata_root_overwrite = 0; % overwrite old files, if any (otherwise skip)

%% run analysis
results = bramila_pronto_run(params);