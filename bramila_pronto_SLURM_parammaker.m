function all_paramsets = bramila_pronto_SLURM_parammaker()
% make parameter sets (file "bramila_pronto_SLURM_parameters.mat") to be run in SLURM, does not submit jobs (use other function for that)!
clc;
warning('off','all'); % don't need this
rng(666);

% path to codes that compute scores, frames and structural data
addpath('/m/nbe/scratch/empathy_and_emotions/shared_codes');
addpath('D:\JanneK\Documents\git_repos\bramila_pronto\shared_codes');

%% --------------------------------
RESULTS_PATH = '/m/nbe/scratch/empathy_and_emotions/Janne_analysis/bramila_pronto_analysis/results_mkl/'; % all jobs, logs and results are saved here
%% --------------------------------

all_subjects = {'sub-08','sub-09','sub-10','sub-11','sub-12','sub-13','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20','sub-21','sub-22','sub-23','sub-24','sub-25','sub-26','sub-27','sub-28','sub-29','sub-30','sub-31','sub-32','sub-33','sub-35','sub-36','sub-37','sub-38','sub-39','sub-40','sub-41','sub-42','sub-43','sub-44','sub-45','sub-46','sub-47','sub-48','sub-49','sub-50','sub-51','sub-52','sub-53','sub-54','sub-55','sub-56','sub-57','sub-58'};
bad = cellfun(@(x) any(ismember(x,{'sub-31','sub-41','sub-27'})),all_subjects);
all_subjects(bad)=[];

all_emotions = {'fear', 'disgust', 'pain', 'sadness', 'anger', 'laughter','confusion','all'};
all_targets = {'FFNI_LackofEmpathy','AQ_total','FFNI_vulnerable','FFNI_grandiose','SRPS_PP','SRPS_SP','EQ_total'};

% parameters defined in "params_grid" structs are varied, while other are kept fixed
% kernels to test
params_grid.kernels = {...
    {'linear'},...
    {'radial_1'},...
    {'radial_100'},...    
    {'radial_0.01'},...
    {'correlation_pearson'},...
    {'polynomial_2'},...
    {'linear','polynomial_2','polynomial_3'},...
    {'radial_1','polynomial_2'},...
    {'correlation_pearson','polynomial_2'}...
    };
params_grid.smoothed_data = {0,1};
params_grid.use_mean_data = {0,1}; % if 0, we use spatio-temporal data
params_grid.parcellation = {...
    'Shirer2012_n14',...
    'aal',...
    'HarvardOxford_2mm_th25_TOTAL',...
    'ICA_n70',...
    'MNI-maxprob-thr25-1mm',...
    'Schaefer2018_100Parcels',...
    'bm20_grey30',...
    'cambridge_basc_multiscale_sym_scale064',...
    'combined_networks_mask'};

%%-----------------------
mkdir(RESULTS_PATH);
% create all combinations
params_grid = generate_params(params_grid);
% randomized grid search
params_grid = params_grid(randperm(length(params_grid)));
params_grid = params_grid';

TOTAL_JOBS = length(params_grid)*length(all_emotions)*length(all_targets);

if TOTAL_JOBS>1e+5
    error('Too much jobs, please reduce!')
end

% collect all individual runs into this cell vector
all_paramsets = cell(1,TOTAL_JOBS);
job_counter = 0;
for emotion_loop = 1:length(all_emotions)
    emotion = all_emotions{emotion_loop};
    for target_loop = 1:length(all_targets)
        target = all_targets{target_loop};
        for loop_params = params_grid
            loop_params = loop_params{1};
            
            % skip combination that do no make sense
            if strcmp(emotion,'all') && loop_params.use_mean_data==1
                % no use to average over majority of data!
                continue;
            end
            
            % job is valid, continue processing
            job_counter=job_counter+1;
            if 0%job_counter>20
                all_paramsets = all_paramsets(1:(job_counter-1));
                save('bramila_pronto_SLURM_parameters.mat','all_paramsets','params_grid','all_emotions','all_targets','all_subjects','-v7.3');
                error('Terminated as requested!');
            end
            
            params = []; % create empty parameter set
            %load('bramila_pronto_defaultparams.mat'); % load default params (to be overwritten) OPTIONAL
            
            %% input data X (neural, structural, etc)
            % all 50 subjects
            %params.allsubs = {'sub-08', 'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17','sub-18', 'sub-19', 'sub-20', 'sub-21', 'sub-22', 'sub-23', 'sub-24', 'sub-25', 'sub-26', 'sub-27','sub-28', 'sub-29', 'sub-30', 'sub-31', 'sub-32', 'sub-33', 'sub-35', 'sub-36', 'sub-37', 'sub-38','sub-39', 'sub-40', 'sub-41', 'sub-42', 'sub-43', 'sub-44', 'sub-45', 'sub-46', 'sub-47', 'sub-48','sub-49', 'sub-50', 'sub-51', 'sub-52', 'sub-53', 'sub-54', 'sub-55', 'sub-56', 'sub-57', 'sub-58'};
            % dropped 3 heavy movers, 47 remaining
            params.allsubs = all_subjects;
            
            %%--------- Define data and target -----------------------
            params.emotion = emotion; % source data type ("X")
            params.target = target; % response label ("Y")
            %%--------------------------------------------------------
            
            %% input data Y (behavioral response)
            params.Y = get_scores(params.allsubs,params.target);
            params.problem_type = 'regression';
            params.max_memory_per_cpu = 2100; % how large dataset to keep in memory vs. loading from disk (in megabytes)

            params.output_folder = RESULTS_PATH;
            mkdir(params.output_folder);
            
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
                params.data_root_folder = '/m/nbe/scratch/empathy_and_emotions/Janne_analysis/new_analysis_Feb2019/hyperalign/data/';
                %params.data_root_folder = 'D:\JanneK\Documents\git_repos\bramila_pronto\data\h5\';
                
                params.allruns = 1:5;
                for sub = 1:length(params.allsubs)
                    for run = 1:length(params.allruns)
                        if sub==1 && run==1
                            params.X.functional.files=[];
                            params.X.functional.frames=[];
                        end
                        if loop_params.smoothed_data
                            params.X.functional.files{sub}{run} = sprintf('%sRUN%i_%s_data_smoothed.h5',params.data_root_folder,params.allruns(run),params.allsubs{sub});
                        else
                            params.X.functional.files{sub}{run} = sprintf('%sRUN%i_%s_data.h5',params.data_root_folder,params.allruns(run),params.allsubs{sub});
                        end
                        if job_counter<20
                            assert(exist(params.X.functional.files{sub}{run},'file')>0,'file not found: %s',params.X.functional.files{sub}{run});
                        end
                        if sub==1
                            params.X.functional.frames{sub}{run} = get_fmri_frames(params.allruns(run),params.emotion);
                        else
                            params.X.functional.frames{sub}{run} = params.X.functional.frames{1}{run};
                        end
                    end
                end
                params.X.functional.do_windowed_zscoring = 1; % do within-window z-scoring after obtaining frames (kills baseline, only dynamics remain)
                params.X.functional.do_temporal_averaging = 0;
                if loop_params.use_mean_data                    
                    params.X.functional.do_windowed_zscoring = 0;
                    params.X.functional.do_temporal_averaging = 1;
                end
                params.X.functional.source_type = 'h5'; % precomputed h5 files
                %% parcellation mask (assume subjects in same space)
                % NEED TO SPECIFY LABELS MANUALLY, CANNOT READ PYTHON LIST DIRECTLY (ugly!)
                % Note: No effect when using NIFTI source
                params.parcellation_label = {'grey_mask','analysis_mask','Shirer2012_n14','BN_Atlas_246','aal','HarvardOxford_2mm_th25_TOTAL','ICA_n70','Gordon_Parcels_MNI_333','MNI-maxprob-thr25-1mm','Schaefer2018_100Parcels','bm20_grey30','cambridge_basc_multiscale_sym_scale064','combined_networks_mask','iCAP_20','Gordon_Parcels_MNI_333'};                
                parcellation_ind = find(ismember(params.parcellation_label,loop_params.parcellation));
                assert(parcellation_ind>2,'Bad parcellation index!');
                params.parcellation_function = sprintf('(mask(1,:)>0.10).*(mask(2,:)>0).*(mask(%i,:))',parcellation_ind); % customize parcellation via logical operations
                params.parcellation_label_chosen = params.parcellation_label{parcellation_ind};
                params.min_voxel_per_ROI = 3*3*3; % for 3mm voxels 9x9x9=729mm^3
                params.X.functional.preprocessor = 'None';%'StandardScaler'; % is None, using precomputed kernels (fast!)
            else
                % If prefer Nifti files, does not allow functional alignment analysis
                % Need to specify frames manually for the input files!
                params.allruns = 1:5;
                for sub = 1:length(params.allsubs)
                    for run = 1:length(params.allruns)
                        if sub==1 && run==1
                            params.X.functional.files=[];
                            params.X.functional.frames=[];
                        end
                        if loop_params.smoothed_data
                            params.X.functional.files{sub}{run} = sprintf('/m/nbe/scratch/empathy_and_emotions/Janne_analysis/new_analysis_Feb2019/bramila_processed_final/RUN%i/bramila/%s_mask_detrend_fullreg_filtered_smoothed_zscored.nii',run,params.allsubs{sub});
                        else
                            params.X.functional.files{sub}{run} = sprintf('/m/nbe/scratch/empathy_and_emotions/Janne_analysis/new_analysis_Feb2019/bramila_processed_final/RUN%i/bramila/%s_mask_detrend_fullreg_filtered_zscored.nii',run,params.allsubs{sub});
                        end
                        if job_counter<10
                            assert(exist(params.X.functional.files{sub}{run},'file')>0,'file not found: %s',params.X.functional.files{sub}{run});
                        end
                        % NOTE: Assume same frames for all subjects!
                        if sub==1
                            params.X.functional.frames{sub}{run} = get_fmri_frames(params.allruns(run),params.emotion);
                        else
                            % same as for the first subject!
                            params.X.functional.frames{sub}{run} = params.X.functional.frames{1}{run};
                        end
                    end
                end
                params.X.functional.do_windowed_zscoring = 1; % do within-window z-scoring after obtaining frames (kills baseline, only dynamics remain)
                params.X.functional.do_temporal_averaging = 0;
                if loop_params.use_mean_data                    
                    params.X.functional.do_windowed_zscoring = 0;
                    params.X.functional.do_temporal_averaging = 1;
                end
                params.X.functional.parcellation_mask = ['/m/nbe/scratch/empathy_and_emotions/Janne_analysis/3mm_parcellations/3mm_',loop_params.parcellation,'.nii'];
                if job_counter<50
                    assert(exist(params.X.functional.parcellation_mask,'file')>0,'Parcellation mask not found!');
                end
                params.X.functional.data_mask = '/m/nbe/scratch/empathy_and_emotions/Janne_analysis/new_analysis_Feb2019/bramila_processed_final/grand_analysis_mask.nii';
                params.parcellation_label_chosen = loop_params.parcellation;
                params.X.functional.source_type = 'nifti'; % precomputed 4D files
                params.X.functional.preprocessor = 'None';%'StandardScaler'; % is None, using precomputed kernels (fast!)
            end
            
            % Next we add additional modalities. Each must contain only one data-array (samples x features), we also need to save total number of elements in the data.
            % If more complex data is needed, e.g., with multiple ROIs, one needs to add special case for that into "bramila_pronto_run" data loader.            
            if 1
                %% structural data with FreeSurfer estimated volume sizes
                file = sprintf('%sstructural_data.mat',RESULTS_PATH);                
                if job_counter<5 && ~exist(file,'file')
                    data = single(get_structural(params.allsubs));
                    elements = numel(data);                    
                    save(file,'data','elements','-v7.3');
                end
                params.X.structural.files = {file}; % enclose in cell
                params.X.structural.preprocessor = 'StandardScaler'; % is none, skipping all transformations (very  fast computations)
            end
            
            %% resting state data
            if 1
                %% structural data with FreeSurfer estimated volume sizes
                file = sprintf('%srestingstate_data.mat',RESULTS_PATH);                
                if job_counter<5 && ~exist(file,'file')
                    data = single(get_restingstate(params.allsubs));
                    elements = numel(data);                    
                    save(file,'data','elements','-v7.3');
                end
                params.X.restingstate.files = {file}; % enclose in cell
                params.X.restingstate.preprocessor = 'None'; % is none, skipping all transformations (very  fast computations)
            end
            
            %% physiological data, eyetracking, etc.
            % .....
            
            %% learner algorithm and params
            if 1
                predictor_params.model = 'simpleMKL';% currenly available 'bayesMKL' and 'simpleMKL';
                predictor_params.args.C = [0.01,0.1,1,10,100,1000]; % for simpleMKL
            else
                predictor_params.model = 'bayesMKL';%
                predictor_params.args = []; % no tunable hyperparameters for bayesMKL!
            end
            params.predictor = predictor_params;
            
            %% kernel type or types (pooled together)
            % parameters that follow string after '_'
            %  order of polynomial (polynomial kernel)
            %  gamma scaling factor (radial kernel) in addition to standard 1/#features -scaling
            params.kernels = loop_params.kernels;
            params.kernel_normalization = 'CenterAndNormalize'; % 'CenterAndNormalize' (recommended! Pronto-style), 'CenterAndTrace', 'Center' or 'None'
            
            %% cross-validation of the model (main loop)
            params.cv = 'LOSO'; % integer or string 'LOSO' (Leave-One-Subject-Out)
            
            %% internal cross-validation within the main loop
            params.internal_cv = 10; % NOTE: Model will be trained total params.cv*params.internal_cv times, keep this at 5-15
            params.internal_cv_skipFeatureTraining = 1; % omit feature transformer re-training. Faster, but can be suboptimal! Disable, if possible.
            
            %% how many workers to use (divided over folds), ideally equal or multiple of params.cv
            params.workers = 12; % use 1 for debugging
            
            %% local tempdata folder
            params.tempdata_root = '';
            params.tempdata_root_overwrite = 0; % overwrite old files, if any (otherwise skip)
            params.delete_temp_data=1; % delete temp data after analysis
                                   
            % name of the results file
            params.result_file_name = sprintf('%sresults_set%i_%s_%s.mat',params.output_folder,job_counter,params.emotion,params.target);
            
            % analysis label
            params.analysis_name = num2str(job_counter);
            
            % name of the parameters file
            params.param_file = sprintf('%sparams_set%i.mat',params.output_folder,job_counter);
            save(params.param_file,'params');
            
            all_paramsets{job_counter} = params.param_file;
            if mod(job_counter+1,100)==0
                fprintf('..set %i of %i\n',job_counter,TOTAL_JOBS);
            end
        end
    end
end
all_paramsets = all_paramsets(1:job_counter);
fprintf('Total parameter sets: %i\n',length(all_paramsets));

save('bramila_pronto_SLURM_parameters.mat','all_paramsets','params_grid','all_emotions','all_targets','all_subjects','-v7.3');

end

function out_params = generate_params(in_params)

allfields = fields(in_params);
n=1;
for k = 1:length(allfields)
    n=n*length(in_params.(allfields{k}));
end
fprintf('Total %i combinations\n',n)

out_params = cell(5e+6,1);
params=struct();
[out_params,curlen] = add_params(out_params,params,in_params,allfields,1,0);
out_params = out_params(1:curlen);

end

function [out_params,curlen] = add_params(out_params,params,in_params,allfields,current_field,curlen)

if current_field>length(allfields)
    curlen=curlen+1;
    out_params{curlen}=params;    
else
    for k = 1:length(in_params.(allfields{current_field}))
        params.(allfields{current_field})=in_params.(allfields{current_field}){k};
        [out_params,curlen] = add_params(out_params,params,in_params,allfields,current_field+1,curlen);
        if curlen==length(out_params)
            if current_field==1
                fprintf('\n!!!! Maximum parameter combination count reached !!!!\n');
            end
            return;
        end
    end
end

end

