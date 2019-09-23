% predictive modeling using multiple kernel learning support-vector regression and classification.
% This is basically an easy-to-modify, minimal, self-sufficient and optimized implementation of Pronto Toolbox regression/classification with multiple kernels.
% The pipeline in short:
% (1) make temporary fast-to-access local files
% (2) split samples in folds (external)
% (3) split samples in sub-folds (internal)
% (4) optimize hyperparameters using internal folds
% (5) train feature extractor and MKL model
% (6) predict and estimate error over all folds

% Initial version April 2019, Janne Kauttonen, Aalto University (NBE)
%
% current version: 19.6.2019 added permutations to compute p-values

function results = bramila_pronto_run(params,index)

%% initializations
addpath(genpath([pwd,filesep,'machines'])); % make sure to add machines
warning('on','all'); % turn on all warnings
rng('default');

% some Pronto default parameters
%--------------------------------------------------
prt_def=[];

% for simpleMKL
prt_def.model.svmargs     = 1;
prt_def.model.libsvmargs  = '-q -s 0 -t 4 -c ';
prt_def.model.rtargs      = 601;
prt_def.model.l1MKLargs   = 1; % how many arguments
prt_def.model.l1MKLmaxitr = 250; % default 250
% for bayesMKL
prt_def.model.alpha_gamma = 1;
prt_def.model.beta_gamma = 1;
prt_def.model.alpha_omega = 1e-6; % force some sparsity
prt_def.model.beta_omega = 1e+6; % force some sparsity

% are parameters given as struct or file?
if ischar(params)
    fprintf('Loading parameters... ');
    A = matfile(params);
    if isfield(A,'all_paramsets')
        params=A.all_paramsets{index};
    else    
        if ~ismember('params',fields(A))
            error('Parameter file ''%s'' does not contain params structure!',params);
        end
        params = A.params;
    end
    clear A;
    fprintf(' done\n');
end

% !!!!!! DEBUGGING AREA, manually customize params %%%%%%%%%%%%%%
if 0
    warning('!!!!!! Debugging area activated, SHOULD ONLY USE FOR DEBUGGING !!!!!!');
    params.workers = 1;
    for ses=1:5
        for sub=1:46
            %params.X.functional.files{sub}{ses}=replace(params.X.functional.files{sub}{ses},'/m/nbe/scratch/empathy_and_emotions/Janne_analysis/new_analysis_Feb2019/hyperalign/data/','D:\JanneK\Documents\git_repos\bramila_pronto\data\h5\');
            sub
            ses
            params.X.functional.files{sub}{ses}=replace(params.X.functional.files{sub}{ses},'/m/nbe/scratch/empathy_and_emotions/Janne_analysis/new_analysis_Feb2019/bramila_processed_final/RUN1/bramila/','D:\JanneK\Documents\git_repos\bramila_pronto\data\nifti\RUN1\');           
        end
    end
    params.X.functional.parcellation_mask = replace(params.X.functional.parcellation_mask,'/m/nbe/scratch/empathy_and_emotions/Janne_analysis/3mm_parcellations/','D:\JanneK\Documents\git_repos\bramila_pronto\3mm_parcellations\');
    params.X.functional.data_mask = replace(params.X.functional.data_mask,'/m/nbe/scratch/empathy_and_emotions/Janne_analysis/new_analysis_Feb2019/bramila_processed_final/','D:\JanneK\Documents\git_repos\bramila_pronto\data\');
    
    params.tempdata_root= 'C:\Users\JanneK\AppData\Local\Temp\05_07_2019_17_48_50_bramila_pronto\';
    params.X.structural.files{1} = 'D:\JanneK\Documents\git_repos\bramila_pronto\TEMP\structural_data.mat';
    params.X.restingstate.files{1} = 'D:\JanneK\Documents\git_repos\bramila_pronto\TEMP\restingstate_data.mat';
end
% !!!!!!! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

params.prt_def = prt_def;
%--------------------------------------------------
if ~isfield(params.predictor,'opts')
    params.predictor.opts=[];
end
if ~isfield(params,'analysis_name')
    params.analysis_name = '';
end
if ~isfield(params,'permutations')
    params.permutations = 0;
end

% add some basic fields
params.N_modalities = length(fields(params.X));
params.N_samples = length(params.allsubs);
params.allsubs_ind = 1:length(params.allsubs);
params.allsubs_ind=params.allsubs_ind(:);

if isfield(params.X,'functional')
    assert(not(params.X.functional.do_windowed_zscoring && params.X.functional.do_temporal_averaging),'Cannot do both z-scoring and averaging, gives all zero data!!');
end

% main cross-validation loop
if ischar(params.cv) && strcmp('LOSO',params.cv)
    params.cv = params.N_samples;
elseif isnumeric(params.cv)
    params.cv = round(min(params.N_samples,params.cv));
else
    error('CV must be ''LOSO'' or integer');
end

params=start_pool(params);

if isfield(params.X,'functional')
    % create local tempdata
    params=create_tempdata(params);
end

% load data into memory, if small enough
params=preload_data(params);

% make sure problem type makes sense
if params.N_samples>10 && strcmp(params.problem_type,'regression')
    assert(length(unique(params.Y))>5,'should have >5 unique values for regression problem (now %i)',length(unique(params.Y)));
end

% if doing permutations, expand response by adding shuffled responses
% NOTE: first column is the original data!
if params.permutations>0
    params.Y = [params.Y,nan(params.N_samples,params.permutations)];
    for perm = 1:params.permutations
        params.Y(:,perm+1) = params.Y(randperm(params.N_samples));
    end
    assert(nnz(isnan(params.Y))==0,'params.Y contains NaN''s!!');
end

% create folds
rng('default');
[train_indices,test_indices] = KFold(params.cv,params.allsubs_ind);

% precompute kernels if possible, will save lots of time
params = get_precomputed_kernels(params);

% restart pool just in case it was closed
params=start_pool(params);

% process all folds, parallel highly recommended!
fprintf('\nStarting computing with %i workers (%i permutations)\n',params.workers,params.permutations);
params.start_time = tic();
if params.workers>1
    parfor (fold = 1:length(train_indices),params.workers)
        temp_params = params;
        temp_params.current_fold = fold;
        results_fold{fold} = process_fold(temp_params,train_indices{fold},test_indices{fold});
    end
else
    % no parallel, do one-by-one
    results_fold=cell(1,length(train_indices));
    for fold = 1:length(train_indices)
        fprintf('Starting fold %i of %i\n',fold,length(train_indices));
        params.current_fold = fold;
        results_fold{fold} = process_fold(params,train_indices{fold},test_indices{fold});
    end
end

% remove fields that take much memory, prepare for saving
params = clear_params(params);

% process results over all folds
kernel_weights = 0;
kernel_weights_mat = [];
Y_real = [];
Y_null = [];
Y_predicted = [];
for fold = 1:length(results_fold)
    Y_real=[Y_real;results_fold{fold}.Y_test_real];
    Y_null=[Y_null;results_fold{fold}.Y_test_predicted_null];
    Y_predicted=[Y_predicted;results_fold{fold}.Y_test_predicted];
    kernel_weights=kernel_weights+results_fold{fold}.kernel_weights*length(results_fold{fold}.Y_test_real);
    kernel_weights_mat = [kernel_weights_mat;results_fold{fold}.kernel_weights];
end
kernel_weights = kernel_weights/params.N_samples;

t = toc(params.start_time);
if t<120
    fprintf('\nAll folds done in %.1fsec !!\n',t);
else
    fprintf('\nAll folds done in %.1fmin !!\n',t/60);    
end

% populate the result structure
results = [];

results.error = get_prediction_error(Y_real(:,1),Y_predicted(:,1),params.problem_type);
results.error_null = get_prediction_error(Y_real(:,1),Y_null(:,1),params.problem_type);
results.error_ratio = results.error(1)./results.error_null(1); % RMSE ratio
results.error_ratio_mse = (results.error_ratio(1)).^2; % MSE ratio (variance reduction)
results.Y_real = Y_real(:,1);
results.Y_null = Y_null(:,1);
results.Y_predicted = Y_predicted(:,1);
results.R = corr(Y_real(:,1),Y_predicted(:,1),'type','pearson');
results.R2 = (results.R(1)).^2;

% if we have permutations, compute results
if params.permutations>0
    results.permutations.Y_real = Y_real(:,2:end);
    results.permutations.Y_null = Y_null(:,2:end);
    results.permutations.Y_predicted = Y_predicted(:,2:end);    
    results.permutations.error = get_prediction_error(results.permutations.Y_real,results.permutations.Y_predicted,params.problem_type);
    results.permutations.error_null = get_prediction_error(results.permutations.Y_real,results.permutations.Y_null,params.problem_type);
    results.permutations.error_ratio = results.permutations.error./results.permutations.error_null; % RMSE ratio
    results.permutations.error_ratio_mse = (results.permutations.error_ratio).^2; % MSE ratio (variance reduction)
    results.permutations.R=nan(1,length(results.permutations.error));
    for i=1:length(results.permutations.error)
        results.permutations.R(i) = corr(results.permutations.Y_real(:,i),results.permutations.Y_predicted(:,i),'type','pearson');
    end
    results.permutations.R2 = (results.permutations.R).^2;
    
    % now compute simple empirical p-values based on real and permuted data
    results.pvals.error = 1 - sum(results.error<results.permutations.error)/length(results.permutations.error); % left tail
    results.pvals.R = 1 - sum(results.R>results.permutations.R)/length(results.permutations.error); % right tail
    results.pvals.R2 = 1 - sum(results.R2>results.permutations.R2)/length(results.permutations.error); % right tail    
    try
        % try to compute Pareto tail approximates
        results.pvals.error_Pareto=nan;
        results.pvals.R_Pareto=nan;
        results.pvals.R2_Pareto=nan;        
        results.pvals.error_Pareto = compute_Pareto(results.error,results.permutations.error,'left');
        results.pvals.R_Pareto = compute_Pareto(atanh(results.R),atanh(results.permutations.R),'right'); % Pareto is unlimited
        results.pvals.R2_Pareto = compute_Pareto((atanh(results.R)).^2,(atanh(results.permutations.R)).^2,'right'); % Pareto is unlimited
    catch ERR
        warning('Failed to compute Pareto p-val tail estimates! Reason: %s',ERR.message);
    end
else
    % no permutations
    results.permutations=[];
    results.pvals=[];   
end

results.kernel_labels = params.kernel_labels;
results.mean_kernel_weights = kernel_weights;
results.kernel_weights_matrix = kernel_weights_mat;
results.params = params;
results.results_for_folds=results_fold;

if ~isfield(params,'result_file_name') || isempty(params.result_file_name)
    params.result_file_name = [params.output_folder,filesep,'bramila_pronto_RESULTS.mat'];
end
% try to save results on disk (just in case)
try
    if ~exist(params.output_folder,'dir')
        mkdir(params.output_folder);
    end
    assert(exist(params.output_folder,'dir')>0,'result folder could not be created!');
    save(params.result_file_name,'results_fold','params','results','-v7.3');
    fprintf('Results saved in ''%s''\n',params.result_file_name);
catch err
    warning('Failed to write results files: %s',err.message);
end

if ~isfield(params,'delete_temp_data')
    params.delete_temp_data=1;
end

try
    if params.delete_temp_data
        delete_temp_data(params);
    end
catch err
    warning('Failed to delete temp data: %s',err.message);
end

% print final results
fprintf('\n---- Final results over %i folds: Error %f (dummy %f), R %f, R2 %f, Error ratio %f ----\n',params.cv,results.error,results.error_null,results.R,results.R2,results.error_ratio);
if results.error_ratio>1
    warning('!! You model did WORSE than a dummy model (RMSE ratio over 1), there appears to be no connection between input and output :''( !!');
end

end

function delete_temp_data(params)
    % delete tempfiles
    if isfield(params.X,'functional')
        for sub = 1:length(params.X.functional.files)
            % make sure the file is tempfile!
            if contains(params.X.functional.files{sub},'_roidata_PRONTO_TEMPFILE.mat')
                delete(params.X.functional.files{sub});
            end
        end
    end
end

function params = clear_params(params)
% clear all unneccesary and large fields before saving 
fprintf('Cleaning parameter structure\n');
all_modalities = fields(params.X)';
for curr_modality = all_modalities
    params.X.(curr_modality{1}).preloaded_data = [];
    params.X.(curr_modality{1}).precomputed_X_kernels=[];
    params.roi_inds=[];
end
end

function params=preload_data(params)
% make precomputed kernels if possible
fprintf('Loading data into memory if less than %iMB in size\n',params.max_memory_per_cpu);
all_modalities = fields(params.X)';
for curr_modality = all_modalities
    params.X.(curr_modality{1}).preloaded_data = [];
    N_datafiles = length(params.X.(curr_modality{1}).files);    
    if isfield(params.X.(curr_modality{1}),'datasize_MB')
        mem = params.X.(curr_modality{1}).datasize_MB;
    else
        total_elements = 0;
        for sub=1:N_datafiles
            load(params.X.(curr_modality{1}).files{sub},'elements');
            total_elements = total_elements + elements;
        end
        mem = 4*total_elements/1e+6;
    end
    if mem<params.max_memory_per_cpu
        fprintf('..loading dataset for ''%s'' (size ~%0.2fMB)\n',curr_modality{1},mem);        
        N_subs = length(params.allsubs);
        assert(N_datafiles==1 || N_datafiles == N_subs,'number of files must be 1 or number of subjects!');
        alldata=cell(1,N_datafiles);
        for sub=1:N_datafiles
            load(params.X.(curr_modality{1}).files{sub},'data');
            if N_datafiles>1
                alldata{sub}=data;               
            else
                assert(size(data,1)==N_subs,'data row count must match subject count (%i is not %i)!',size(data,1),N_subs);
                alldata = data;                
            end
        end
        params.X.(curr_modality{1}).preloaded_data = alldata;
    else
        fprintf('..data for ''%s'' is too large to preload (size %iMB)\n',curr_modality{1},mem);
    end
end

end

function params = start_pool(params)
% start local pool with specific number of workers
if params.workers>1
    mycluster = gcp('nocreate');
    skip=0;
    if ~isempty(mycluster)
        if mycluster.NumWorkers~=params.workers
            delete(mycluster);
        else
            skip=1;
        end
    end
    if skip==0
        mycluster=parcluster('local');
        mycluster.NumWorkers = max(1,params.workers);        
        if ~isfield(params,'JobStorageLocation') || ( isfield(params,'JobStorageLocation') && isempty(params.JobStorageLocation) )
            t=tempname();
            mkdir(t);
            params.JobStorageLocation = t;
        end
        mycluster.JobStorageLocation=params.JobStorageLocation;
        parpool(mycluster);
        params.workers = mycluster.NumWorkers;
    end
end

end

function params = clear_precomputed_kernels(params,modalities)
% clear specified precomputed kernels
for curr_modality = modalities
    params.X.(curr_modality{1}).precomputed_X_kernels=[];
end
end

function [params,added_modalities] = get_precomputed_kernels(params,train_inds)
added_modalities=[];
if nargin==1
    % try to make kernels for all samples
    fprintf('Making precomputed kernels if no feature training was requested\n')
    params.kernel_labels = get_kernel_labels(params);
    all_modalities = fields(params.X)';
    for curr_modality = all_modalities
        params.X.(curr_modality{1}).precomputed_X_kernels = [];
        if strcmp(params.X.(curr_modality{1}).preprocessor,'None')
            % no preprocessing required - can precompute kernel!
            kernels = get_data(params,params.allsubs_ind,[],curr_modality{1},1);
            params.X.(curr_modality{1}).precomputed_X_kernels = kernels.(curr_modality{1});
            added_modalities=[added_modalities,{curr_modality{1}}];
        else
            % requires preprocessing - CANNOT precompute kernel
            fprintf('..skipping ''%s'' with preprocessing ''%s'' \n',curr_modality{1},params.X.(curr_modality{1}).preprocessor);
        end
    end    
else    
    % make kernels for selected samples
    all_modalities = fields(params.X)';
    for curr_modality = all_modalities
        if isempty(params.X.(curr_modality{1}).precomputed_X_kernels)
            % make kernel            
            kernels = get_data(params,train_inds,[],curr_modality{1},0); % do not print info
            full_kernels = nan(params.N_samples,params.N_samples,size(kernels.(curr_modality{1}),3));
            full_kernels(train_inds,train_inds,:)=kernels.(curr_modality{1});                        
            params.X.(curr_modality{1}).precomputed_X_kernels = full_kernels;
            added_modalities=[added_modalities,{curr_modality{1}}];
        end
    end    
end
end

function savedata(file,data,parcellation_function,parcellation_mask,timeindices,filenames,do_zscoring,do_averaging,elements)
% save data, function needed in parfor loops
save(file,'data','parcellation_function','parcellation_mask','timeindices','filenames','elements','do_zscoring','do_averaging','-v7.3');
end

function params = create_tempdata(params)
% create temporary local dataset for fast fMRI data access during regression
% input data is either .h5's or niftis

% create temp directory
temp_folder_id = [datestr(datetime('now'),'mm_dd_yyyy_HH_MM_SS'),params.analysis_name,'_bramila_pronto',filesep];
if isempty(params.tempdata_root)
    params.tempdata_root = [tempdir(),temp_folder_id];
end
if ~exist(params.tempdata_root,'dir')
    mkdir(params.tempdata_root);
    assert(exist(params.tempdata_root,'dir')>0,'failed to create tempdir!');
end
fprintf('Using tempdir ''%s''\n',params.tempdata_root);

total_elements = 0;

do_zscoring = params.X.functional.do_windowed_zscoring;
do_averaging = params.X.functional.do_temporal_averaging;

params.X.functional.files_original = params.X.functional.files; % store original filenames

if strcmp(params.X.functional.source_type,'h5_aligned')    
    parcellation_mask = params.parcellation_label;
    roi_inds=[];
    roi = 0;
    n=0;
    while 1
        try
            roi_data = h5read(params.X.functional.files{1}{1},sprintf('/R_common/i%i',roi));
            roi=roi+1;
            roi_inds{roi} = n+(1:size(roi_data,2));
            n=n+length(roi_inds{roi});
        catch
            break;
        end
    end
    N_rois = roi;
    params.total_voxels = n; % not really voxels anymore
    new_roi_inds=roi_inds;
    new_parcellation_coord=nan;
    
    for sub = 1:length(params.allsubs)
        file = [params.tempdata_root,sprintf('sub%i_roidata_PRONTO_TEMPFILE.mat',sub)];
        new_files{sub}=file;
        if exist(file,'file')>0 && params.tempdata_root_overwrite==0 && isSameData(params,sub,parcellation_label,file)
            fprintf('...subject ''%s'': %i of %i (%s) !! using existing file!\n',params.allsubs{sub},sub,length(params.allsubs),file);
            load(file,'elements');
            total_elements = total_elements+elements;
            continue;
        else
            fprintf('...subject ''%s'': %i of %i (%s)\n',params.allsubs{sub},sub,length(params.allsubs),file);
        end
        new_data = [];
        timeindices = cell(1,length(params.allruns));
        filenames = cell(1,length(params.allruns));
        for run = 1:length(params.allruns)            
            time_ind = params.X.functional.frames{sub}{run};
            timeindices{run} = time_ind;
            filenames{run} = params.X.functional.files{sub}{run};
            if ~isempty(time_ind)
                rundata = nan(length(time_ind),params.total_voxels,'single');
                for roi = 1:length(roi_inds)
                    data = h5read(params.X.functional.files{sub}{run},sprintf('/R_common/i%i',roi-1));
                    data = data(time_ind,:);
                    rundata(:,roi_inds{roi}) = data;
                end
                if params.X.functional.do_windowed_zscoring
                    rundata = zscore(rundata,[],1);
                end
                new_data = [new_data;rundata];
            end
        end
        assert(size(new_data,2)==params.total_voxels,sprintf('Voxel count does not match! Data size was %s',mat2str(size(new_data))));
        if params.X.functional.do_temporal_averaging
            new_data = mean(new_data,1);
        end
        % new_data = timepoints x voxels
        data=new_data;
        elements = numel(data);
        total_elements = total_elements+elements;
        save(file,'data','parcellation_mask','timeindices','filenames','do_zscoring','do_averaging','elements','-v7.3');
    end

elseif strcmp(params.X.functional.source_type,'h5')
    % fMRI data is already organized into .h5 files
    % compute parcellation mask
    mask = h5read(params.X.functional.files{1}{1},'/parcellation_data');
    parcellation_mask = eval(params.parcellation_function);
    parcellation_mask(isnan(parcellation_mask))=0;
    parcellation_coord = h5read(params.X.functional.files{1}{1},'/mask_img_coord');
    
    % get ROIs
    IDs = unique(parcellation_mask);
    IDs(IDs==0)=[];
    IDs(isnan(IDs))=[];
    n=0;
    counts = nan(1,length(IDs));
    roi_inds = [];
    
    % get roi indices and voxel counts, nullify too small parcels
    n_voxels = 0;
    for k = 1:length(IDs)
        id = IDs(k);
        counts(k)=sum(parcellation_mask==id);
        if counts(k)<params.min_voxel_per_ROI
            n=n+1;
            counts(k)=0;
            parcellation_mask(parcellation_mask==id)=0;
        else
            roi_inds{end+1}=find(parcellation_mask==id);
            n_voxels = n_voxels + length(roi_inds{end});
        end
    end
    assert(n_voxels == sum(counts));
    if n>0
        warning('!!!! Removed %i ROIs each with less than %i voxels !!!!',n,params.min_voxel_per_ROI);
    end
    params.total_voxels = sum(counts);
    
    fprintf('final parcellation contains %i parts and total %i voxels\n',length(roi_inds),params.total_voxels);
    
    % get new indices and coordinates
    new_roi_inds=[];
    new_parcellation_coord=zeros(3,params.total_voxels,'int32');
    n=0;
    for roi = 1:length(roi_inds)
        new_roi_inds{roi} = n+(1:length(roi_inds{roi}));
        new_parcellation_coord(:,new_roi_inds{roi}) = parcellation_coord(:,roi_inds{roi});
        n=n+length(roi_inds{roi});
    end
    
    parcellation_function=params.parcellation_function;
    
    % new files and frames for functional data
    new_files=cell(1,length(params.allsubs));
    if params.workers>1
        elements = zeros(1,length(params.allsubs));
        parfor (sub = 1:length(params.allsubs),params.workers)
            file = [params.tempdata_root,sprintf('sub%i_roidata_PRONTO_TEMPFILE.mat',sub)];
            new_files{sub}=file;
            timeindices = cell(1,length(params.allruns));
            filenames = cell(1,length(params.allruns));                  
            if exist(file,'file')>0 && params.tempdata_root_overwrite==0 && isSameData(params,sub,parcellation_mask,file)
                fprintf('...subject ''%s'': %i of %i (%s) !! using existing file!\n',params.allsubs{sub},sub,length(params.allsubs),file);
                A = load(file,'elements');
                elements(sub) = A.elements;                
                continue;
            else
                fprintf('...subject ''%s'': %i of %i (%s)\n',params.allsubs{sub},sub,length(params.allsubs),file);
            end
            new_data = [];
            for run = 1:length(params.allruns)
                data = h5read(params.X.functional.files{sub}{run},'/data');
                time_ind = params.X.functional.frames{sub}{run};
                timeindices{run} = time_ind;
                filenames{run} = params.X.functional.files{sub}{run};                  
                if ~isempty(time_ind)
                    rundata = zeros(length(time_ind),params.total_voxels,'single');
                    data = data(params.X.functional.frames{sub}{run},:);
                    for roi = 1:length(roi_inds)
                        rundata(:,new_roi_inds{roi}) = data(:,roi_inds{roi});
                    end
                    if params.X.functional.do_windowed_zscoring
                        rundata = zscore(rundata,[],1);
                    end                     
                    new_data = [new_data;rundata];
                end
            end
            assert(size(new_data,2)==params.total_voxels,sprintf('Voxel count does not match! Data size was %s',mat2str(size(new_data))));
            if params.X.functional.do_temporal_averaging
                new_data = mean(new_data,1);
            end            
            data=new_data;
            elements(sub)=numel(data);
            savedata(file,data,parcellation_function,parcellation_mask,timeindices,filenames,do_zscoring,do_averaging,elements(sub));                        
        end
        total_elements = sum(elements);
    else
        for sub = 1:length(params.allsubs)
            file = [params.tempdata_root,sprintf('sub%i_roidata_PRONTO_TEMPFILE.mat',sub)];
            new_files{sub}=file;
            if exist(file,'file')>0 && params.tempdata_root_overwrite==0 && isSameData(params,sub,parcellation_mask,file)
                fprintf('...subject ''%s'': %i of %i (%s) !! using existing file!\n',params.allsubs{sub},sub,length(params.allsubs),file);
                load(file,'elements');
                total_elements = total_elements+elements;                
                continue;
            else
                fprintf('...subject ''%s'': %i of %i (%s)\n',params.allsubs{sub},sub,length(params.allsubs),file);
            end
            new_data = [];
            timeindices = cell(1,length(params.allruns));
            filenames = cell(1,length(params.allruns));            
            for run = 1:length(params.allruns)                                
                data = h5read(params.X.functional.files{sub}{run},'/data');
                time_ind = params.X.functional.frames{sub}{run};                
                timeindices{run} = time_ind;
                filenames{run} = params.X.functional.files{sub}{run};                
                if ~isempty(time_ind)
                    rundata = zeros(length(time_ind),params.total_voxels,'single');
                    data = data(params.X.functional.frames{sub}{run},:);
                    for roi = 1:length(roi_inds)
                        rundata(:,new_roi_inds{roi}) = data(:,roi_inds{roi});
                    end
                    if params.X.functional.do_windowed_zscoring
                        rundata = zscore(rundata,[],1);
                    end                 
                    new_data = [new_data;rundata];
                end
            end
            assert(size(new_data,2)==params.total_voxels,sprintf('Voxel count does not match! Data size was %s',mat2str(size(new_data))));
            if params.X.functional.do_temporal_averaging
                new_data = mean(new_data,1);
            end
            % new_data = timepoints x voxels
            data=new_data;
            elements = numel(data);
            total_elements = total_elements+elements;            
            save(file,'data','parcellation_function','parcellation_mask','timeindices','filenames','do_zscoring','do_averaging','elements','-v7.3');
        end
    end

elseif strcmp(params.X.functional.source_type,'nifti')    
    % fMRI data is nifti
    assert(~isempty(which('load_nii.m')),'NIFTI tools not found, please add path to Matlab for load_nii function!');    
    % prepare parcellation mask
    nii = load_nii(params.X.functional.parcellation_mask,1);
    parcellation_mask = double(nii.img);
    nii = load_nii(params.X.functional.data_mask,1);
    data_mask = double(nii.img);    
    % cut out voxels outside data mask
    parcellation_mask = parcellation_mask.*(data_mask>0);
    parcellation_mask(isnan(parcellation_mask))=0;
    params.total_voxels = nnz(parcellation_mask);
    roi_num = unique(parcellation_mask(:));
    roi_num(roi_num==0)=[];     
    N_roi = length(roi_num);
    fprintf('..found %i ROIs in nifti mask\n',N_roi);
        
    % get new indices and coordinates
    roi_inds=[];
    new_parcellation_coord=zeros(3,params.total_voxels,'int32');
    parcellation_masks = cell(1,N_roi);
    n=0;
    for roi = 1:length(roi_num)
        roi_inds{roi} = n+(1:nnz(parcellation_mask==roi_num(roi)));
        [x,y,z] = ind2sub(size(parcellation_mask),find(parcellation_mask==roi_num(roi)));        
        new_parcellation_coord(:,roi_inds{roi}) = [x,y,z]';
        parcellation_masks{roi} = parcellation_mask==roi_num(roi);
        n=n+length(roi_inds{roi});
    end            
    assert(params.total_voxels==n,'voxel count does not match!');
    
    new_roi_inds=roi_inds;
    new_files=[];
    
    for sub = 1:length(params.allsubs)
        file = [params.tempdata_root,sprintf('sub%i_roidata_PRONTO_TEMPFILE.mat',sub)];        
        new_files{sub}=file;
        if exist(file,'file')>0 && params.tempdata_root_overwrite==0 && isSameData(params,sub,parcellation_mask,file)
            fprintf('...subject ''%s'': %i of %i (%s) !! using existing file!\n',params.allsubs{sub},sub,length(params.allsubs),file);
            load(file,'elements');
            total_elements = total_elements+elements;
            continue;
        else
            fprintf('...subject ''%s'': %i of %i (%s)\n',params.allsubs{sub},sub,length(params.allsubs),file);
        end
        new_data = [];
        timeindices = cell(1,length(params.allruns));
        filenames = cell(1,length(params.allruns));
        for run = 1:length(params.allruns)
            time_ind = params.X.functional.frames{sub}{run};
            timeindices{run} = time_ind;
            filenames{run} = params.X.functional.files{sub}{run};
            if ~isempty(time_ind)
                % data = timepoints x voxels
                if length(time_ind)==1 && time_ind==0
                    data = load_nii_mask(params.X.functional.files{sub}{run},parcellation_masks,400);
                    time_ind = size(data{1},1);
                else                                
                    data = load_nii_mask(params.X.functional.files{sub}{run},parcellation_masks,400,time_ind);
                end
                rundata = zeros(length(time_ind),params.total_voxels,'single');                
                for roi = 1:N_roi
                    rundata(:,roi_inds{roi}) = data{roi};
                end
                if params.X.functional.do_windowed_zscoring
                    rundata = zscore(rundata,[],1);
                end
                new_data = [new_data;rundata];
            end
        end
        assert(size(new_data,2)==params.total_voxels,sprintf('Voxel count does not match! Data size was %s',mat2str(size(new_data))));
        if params.X.functional.do_temporal_averaging
            new_data = mean(new_data,1);
        end
        data=new_data;
        elements = numel(data);
        total_elements = total_elements+elements;
        assert(nnz(isnan(data))==0,'NaN value found in ROI data! Subject %s',params.allsubs{sub});
        save(file,'data','parcellation_mask','timeindices','filenames','do_zscoring','do_averaging','elements','-v7.3');        
    end
else
   error('Unknown source data type, only ''h5'' or ''nifti'' are allowed!');
end
params.X.functional.files = new_files;
params.roi_inds=new_roi_inds;
params.parcellation_coord = new_parcellation_coord;
params.X.functional.datasize_MB = total_elements*4/1e+6;
fprintf('loaded fMRI data size ~%.3gGB\n',params.X.functional.datasize_MB/1000);

end

function is_same = isSameData(params,sub,parcellation_mask,file)
% check if parcellation, frames and filenames match with stored ones
is_same=0;
try
    if ischar(parcellation_mask)
        A = load(file,'parcellation_mask','timeindices','filenames','do_zscoring','do_averaging');
        assert(strcmp(parcellation_mask,A.parcellation_mask));
    else
        A = load(file,'parcellation_mask','timeindices','filenames','do_zscoring','do_averaging');
        assert(nnz(parcellation_mask-A.parcellation_mask)==0);
    end
    for run=1:length(A.timeindices)
        assert(nnz(A.timeindices{run}-params.X.functional.frames{sub}{run})==0 && strcmp(A.filenames{run},params.X.functional.files{sub}{run})==1)
    end
    assert(params.X.functional.do_windowed_zscoring == A.do_zscoring && params.X.functional.do_temporal_averaging ==  A.do_averaging);
    is_same = 1;
catch
end   
end
            
function [train_indices,test_indices] = KFold(folds,samples)
% get k-fold train/test split indices
train_indices=[];
test_indices=[];    
N_samples = length(samples);
if folds==N_samples
    for i=1:folds
        a=samples;
        a(i)=[];
        train_indices{end+1} = sort(a(:),'ascend');
        test_indices{end+1} = samples(i);
    end
else
    CVO = cvpartition(N_samples,'k',folds);
    for i=1:folds
        a = samples(find(CVO.training(i)));
        train_indices{end+1} = sort(a(:),'ascend');
        a = samples(find(CVO.test(i)));
        test_indices{end+1} = sort(a(:),'ascend');
    end
end

end

function kernel = compute_kernel(X,kerneltype)
% compute requested kernel, default scaling parameters according to sklearn
% for all kernels, larger values equals to higher sample similarity!
X = double(X); % convert to double, no need to save space for kernels here!
L = size(X,2); % how many features
ind = strfind(kerneltype,'_');
if ~isempty(ind)
    % kernel contains a parameter, extract it
    param_str = kerneltype(ind+1:end);
    kerneltype=kerneltype(1:(ind-1));
end
if strcmp(kerneltype,'linear') %  k(x,y) = x*y'
    kernel = X*X';
elseif strcmp(kerneltype,'radial') % k(x,y) = exp(-param*(norm(x-y)^2)/L)
    XXh1 = sum(X.^2,2)*ones(1,size(X,1));
    XXh2 = sum(X.^2,2)*ones(1,size(X,1));
    omega = XXh1+XXh2' - 2*(X*X');
    kernel = exp(-omega*str2double(param_str)/L); %
elseif strcmp(kerneltype,'polynomial') %  k(x,y) = (1+x*y'/L)^param
    kernel = (1.0 + (X*X')/L).^str2double(param_str);
elseif strcmp(kerneltype,'correlation') %  k(x,y) = corr(x,y,'type',param_str)
    kernel = corr(X',X','type',param_str);
else
    error('Unknown kernel! Allowed ones: ''linear'', ''radial_X'', ''correlation'', ''polynomial_X''');
end
end

function data = feature_transformer(preprocessor,data,test_samples)
% Feature transformation, learn from training data and apply to test data
train_samples = 1:size(data,1);
train_samples(test_samples)=[];
% make sure ok, 
assert(length(train_samples)+length(test_samples)==size(data,1));

% transform features
if strcmp(preprocessor,'StandardScaler')
    [data(train_samples,:),mu,sigma] = zscore(data(train_samples,:)); % learn params, applyt to training data
    ind = sigma>0;
    data(test_samples,:) = data(test_samples,:)-mu; % apply to testing data   
    data(test_samples,ind) = data(test_samples,ind)./sigma(ind); % apply to testing data   
elseif strcmp(preprocessor,'MaxAbsScaler')
    mu = mean(data(train_samples,:));
    data = data - mu;
    sigma = max(abs(data(train_samples,:)));
    ind = sigma>0;
    data(test_samples,ind) = data(test_samples,ind)./sigma(ind);
elseif strcmp(preprocessor,'None')    
    % no preprocessing, use data as is (not generally recommended)
elseif strcmp(preprocessor,'Dummy')    
    % no preprocessing, FOR DEBUGGING PURPOSES (modify indexing, but not data)
else        
    error('Unknown preprocessing type (%s), only ''None'', ''StandardScaler'' and ''MaxAbsScaler'' allowed!',preprocessor);
end

end

function [X_train,X_test] = center_and_normalize_kernel(params,X,tr_idx,te_idx)
% kernel normalization and splitting into test and train sets
% References:
%   vis.caltech.edu/~graf/my_papers/proceedings/GraBor01.pdf
%   Kernel Methods for Pattern Analysis, Shawe-Taylor & Cristianini, Campbridge University Press, 2004

% Centering and normalizing (Pronto-style, recommended!)
if strcmp(params.kernel_normalization,'CenterAndNormalize')
    % See Pronto function "prt_apply_operations.m"
    X_train = X(tr_idx,tr_idx,:);
    X_test = X(te_idx, tr_idx,:);
    X_tt = X(te_idx, te_idx,:);
    tr = 1:length(tr_idx);
    te = (1:length(te_idx))+max(tr);
    for d = 1:size(X,3)
        % Centering: to make the dataset zero mean; kernel will capture the co-variance of data items around the mean        
        [X_train(:,:,d),X_test(:,:,d),X_tt(:,:,d)] = prt_centre_kernel(X_train(:,:,d),X_test(:,:,d),X_tt(:,:,d));
        Phi = [X_train(:,:,d), X_test(:,:,d)'; X_test(:,:,d), X_tt(:,:,d)];
        % Normalization: projects the data on the unit sphere, removes the effect of the length of data vectors
        Phi = prt_normalise_kernel(Phi);
        X_train(:,:,d) = Phi(tr,tr);
        X_test(:,:,d) = Phi(te,tr);
    end
% only centering
elseif strcmp(params.kernel_normalization,'Center')
    X_train = X(tr_idx,tr_idx,:);
    X_test = X(te_idx, tr_idx,:);
    for d = 1:size(X,3)
        [X_train(:,:,d),X_test(:,:,d)] = prt_centre_kernel(X_train(:,:,d),X_test(:,:,d));
    end
% simple centering and trace normalization
elseif strcmp(params.kernel_normalization,'CenterAndTrace')
    n = size(X,1);
    G = (eye(n) - ones(n)/n);
    for d = 1:size(X,3)
        X(:,:,d) = G*X(:,:,d)*G; % centering, not respecting train/test split!
        % divide by trace
        X(:,:,d) = n*X(:,:,d)/trace(X(:,:,d));
    end
    X_train = X(tr_idx,tr_idx,:);
    X_test = X(te_idx, tr_idx,:);
% do nothing, just split
elseif strcmp(params.kernel_normalization,'None')
    % no scaling, just splitting (NOT recommended!)
    X_train = X(tr_idx,tr_idx,:);
    X_test = X(te_idx, tr_idx,:);
else
    error('Unknown kernel normalization type (%s) !!!!',params.kernel_normalization);
end

end

function [kernels,X_kernels_total,internal_train_samples,internal_test_samples] = get_data(params,train_samples,test_samples,modality,print_info)
% get kernels
if nargin<3
    test_samples=[];
    print_info=0;
    modality=[];
end
if nargin<4
    print_info=0;
    modality=[];
end
if nargin<5
    print_info=0; 
end

n_train = length(train_samples);
n_test = length(test_samples);

% all samples to extract, these must match with the subject IDs (
samples = [train_samples;test_samples];
% new internal indices for training and testing
% these must be used when working with result kernels!
internal_train_samples = (1:length(train_samples))';
internal_test_samples = (length(internal_train_samples)+(1:length(test_samples)))';

N_samples = length(samples);

if isempty(modality)
    all_modalities = fields(params.X)';
else
    all_modalities = {modality};
end

kernels=struct();
total_kernel_count = 0;
for curr_modality = all_modalities      
    if strcmp(curr_modality{1},'functional')
        if ~isempty(params.X.functional.precomputed_X_kernels)
            kernels.functional = params.X.functional.precomputed_X_kernels(samples,samples,:);
            kernel_count=size(kernels.functional,3);            
        else        
            if print_info
                fprintf('...making kernels for fMRI with %i ROIs (%i+%i samples)\n',length(params.roi_inds),n_train,n_test);
            end
            params.X.functional.n_kernels = length(params.roi_inds)*length(params.kernels);
            X_kernels = zeros(N_samples,N_samples,params.X.functional.n_kernels);
            X_labels = cell(1,params.X.functional.n_kernels);
            kernel_count=0;
            % load data for rois and subjects
            for roi = 1:length(params.roi_inds)
                for sub = 1:length(samples)
                    % get data from files or preloaded variable
                    if isempty(params.X.(curr_modality{1}).preloaded_data)
                        % not in memory, need to load from disk
                        A = matfile(params.X.functional.files{samples(sub)});
                    else
                        % data already in memory
                        A.data = params.X.(curr_modality{1}).preloaded_data{samples(sub)};
                    end                    
                    ts = A.data(:,params.roi_inds{roi}); % timeries of all ROI voxels
                    ts = ts(:); % flatten into form [timeslice1,timeslice2,...] where each frame follow each other                    
                    if sub==1
                        fmri_data = zeros(length(samples),length(ts),'single');
                    end
                    fmri_data(sub,:) = ts;
                end
                % transform data
                fmri_data = feature_transformer(params.X.functional.preprocessor,fmri_data,internal_test_samples);
                for kernel_nr = 1:length(params.kernels)
                    kernel_count=kernel_count+1;
                    X_kernels(:,:,kernel_count) = compute_kernel(fmri_data,params.kernels{kernel_nr});
                    X_labels{kernel_count} = sprintf('fMRI_ROI%i-%s',roi,params.kernels{kernel_nr});
                end
            end
            clear fmri_data;
            kernels.functional = X_kernels;
        end        
    else % all else, we assume only one "ROI" per these modalities
        if ~isempty(params.X.(curr_modality{1}).precomputed_X_kernels)
            kernels.(curr_modality{1}) = params.X.(curr_modality{1}).precomputed_X_kernels(samples,samples,:);
            kernel_count=size(kernels.(curr_modality{1}),3);
        else
            if print_info
                fprintf('...making kernels for modality ''%s'' (%i+%i samples)\n',curr_modality{1},n_train,n_test);
            end
            params.X.(curr_modality{1}).n_kernels = length(params.kernels);
            X_kernels = zeros(N_samples,N_samples,params.X.(curr_modality{1}).n_kernels);
            X_labels = cell(1,params.X.(curr_modality{1}).n_kernels);
            kernel_count=0;
            
            % get data from files or preloaded variable
            if isempty(params.X.(curr_modality{1}).preloaded_data)
                assert(length(params.X.(curr_modality{1}).files)==1,'Assuming only one file for non-fMRI input data!');
                A = load(params.X.(curr_modality{1}).files{1});     
                data = A.data(samples,:);
                clear A;
            else
                data = params.X.(curr_modality{1}).preloaded_data(samples,:);
            end            
            
            % transform data
            data = feature_transformer(params.X.(curr_modality{1}).preprocessor,data,internal_test_samples);
            
            for kernel_nr = 1:length(params.kernels)
                kernel_count=kernel_count+1;
                X_kernels(:,:,kernel_count) = compute_kernel(data,params.kernels{kernel_nr});
                X_labels{kernel_count} = sprintf('nonfMRI_%s',params.kernels{kernel_nr});
            end
            clear data;
            kernels.(curr_modality{1}) = X_kernels;
        end
    end    
    total_kernel_count=total_kernel_count+kernel_count;
end
% requested all kernels
if nargout>1
    X_kernels_total = nan(N_samples,N_samples,total_kernel_count);
    k=0;
    for curr_modality = all_modalities   
        n = size(kernels.(curr_modality{1}),3);
        X_kernels_total(:,:,(1:n)+k) = kernels.(curr_modality{1});
        k=k+n;
    end
    assert(nnz(isnan(X_kernels_total))==0,sprintf('NaN''s in pooled kernel data (modality %s), BUG!'));    
end

end

function X_labels_total = get_kernel_labels(params)
kernels=[];
all_modalities = fields(params.X)';
for curr_modality = all_modalities
    if strcmp(curr_modality{1},'functional')
        X_labels = [];
        for roi = 1:length(params.roi_inds)
            for kernel_nr = 1:length(params.kernels)
                X_labels{end+1} = sprintf('%s_ROI%i_%s',curr_modality{1},roi,params.kernels{kernel_nr});
            end
        end
        kernels.functional = X_labels;
    else % all else, we assume only one "ROI" per these modalities
        X_labels = [];
        for kernel_nr = 1:length(params.kernels)
            X_labels{end+1} = sprintf('%s_%s',curr_modality{1},params.kernels{kernel_nr});
        end
        kernels.(curr_modality{1}) = X_labels;
    end
end
X_labels_total = [];
for curr_modality = all_modalities
    X_labels_total = [X_labels_total,kernels.(curr_modality{1})];
end

end

function res = get_prediction_error(y1,y2,type)
% compute error, either for regression or classification
% smaller is better
if strcmp(type,'regression')
    res = sqrt(mean((y1-y2).^2)); % root-mean-squared-error (RMSE)
else
    res = sum(y1~=y2)/size(y1,1); % misclassification error rate, should use F1
end
end

function res = make_prediction_dummy(Y_train,type)
if strcmp(type,'regression')
    res = mean(Y_train);
else
    res = mode(Y_train);
end
end

function model_paramsets = get_model_parameters(predictor_params)
% generate combinations of parameters
root_struct.model = predictor_params.model;
root_struct.opts = predictor_params.opts;
if isempty(predictor_params.args)
    root_struct.args = [];
    model_paramsets{1} = root_struct;
else
    model_paramsets = field_iterator({},root_struct,predictor_params.args,fields(predictor_params.args),1);
end
% randomize parameter order to remove possible order biases (e.g., in equal performance case or early termination)
model_paramsets = model_paramsets(randperm(length(model_paramsets)));
end

function stored_params = field_iterator(stored_params,incompete_params,all_params,all_fields,current_field)
% recursive field iterator function
if current_field>length(all_fields)
    stored_params{end+1} = incompete_params;
    return
else
    for n=1:length(all_params.(all_fields{current_field}))
        incompete_params.args.(all_fields{current_field})=all_params.(all_fields{current_field})(n);
        stored_params = field_iterator(stored_params,incompete_params,all_params,all_fields,current_field+1);
    end
end
end

function [Y_test_predicted,kernel_weights] = make_prediction(params,model_paramsets_in,X_train_kernels,Y_train,X_test_kernels)
% given parameters, kernels and targets, train and test the model
% learning functions are from Pronto Toolbox

if iscell(model_paramsets_in)
   % parameters given as cell array, one set for each target
   is_cell_matrix = 1;
   assert(length(model_paramsets_in)==size(Y_train,2),'must have as many parameter sets as targets!');
else
   % one set applied to all targets
   is_cell_matrix = 0;
   model_paramsets = model_paramsets_in;
end

d = [];
d.train  = X_train_kernels;
d.test = X_test_kernels;
d.use_kernel=1;
d.te_targets = nan(size(X_test_kernels,1),1);

% run prediction for all permutations (columns of Y_train)
Y_test_predicted = nan(size(X_test_kernels,1),size(Y_train,2));
for col = 1:size(Y_train,2)
    
    if is_cell_matrix
        model_paramsets = model_paramsets_in{col};
    end
    
    d.tr_targets = Y_train(:,col);    
    
    if strcmp(model_paramsets.model,'simpleMKL')
        if strcmp(params.problem_type,'regression')
            output = prt_machine_sMKL_reg(d,params.prt_def,model_paramsets.args.C,model_paramsets.opts);
        else
            output = prt_machine_sMKL_cla(d,params.prt_def,model_paramsets.args.C,model_paramsets.opts);
        end
    elseif strcmp(model_paramsets.model,'bayesMKL')
        if strcmp(params.problem_type,'regression')
            output = prt_machine_bayesMKL_reg(d,params.prt_def,[],model_paramsets.opts);
        else
            output = prt_machine_bayesMKL_cla(d,params.prt_def,[],model_paramsets.opts);
        end
    elseif strcmp(model_paramsets.model,'GMKL')
        output = GMKL_learner(d,params.prt_def,model_paramsets.args.C,model_paramsets.opts,params.problem_type);
    elseif strcmp(model_paramsets.model,'spicyMKL')
        output = SPICY_learner(d,params.prt_def,[model_paramsets.args.alpha,model_paramsets.args.C],model_paramsets.opts,params.problem_type);
    else
        error('requested model not implemented (%s)',model_paramsets.model);
    end
    
    if col==1
        % only interested in kernel weights for unpermuted data!
        kernel_weights = output.beta;
    end
    Y_test_predicted(:,col) = output.predictions;
    
end
% sanity check that no NaNs remain
assert(nnz(isnan(Y_test_predicted))==0,'Predicted values contains NaN''s!');

end

function ind = argmin(arr)
[~,ind]=min(arr);
end

function print_best_params(fold,predictor_params)
% print parameter info
fprintf('..Fold %i: Best parameters (total %i) for ''%s'': ',fold,length(predictor_params.args),predictor_params.model);
for f = fields(predictor_params.args)'
    fprintf('%s=%d, ',f{1},predictor_params.args.(f{1}));
end
fprintf('\n');
end

function [results,X_train_kernels,X_test_kernels] = process_fold(params,train_inds,test_inds,model_paramsets,X_train_kernels,X_test_kernels,Y_train,Y_test)
% function to process single fold. Handles both external and internal folds depending on the input parameters.
% NOTE: input train_inds & test_inds always in external index space!

start_time = tic();

is_internal = 1; % internal or external fold type
if nargin<4
    model_paramsets = get_model_parameters(params.predictor);
    is_internal=0;
end

% if more than one parameterset given, starting internal parameter fine-tuning
if length(model_paramsets)>1
    internal_tuning = struct();

    % these are only response available, will be split between train and test
    internal_tuning.Y_train_all = params.Y(train_inds,:);
    
    added_modalities=[];
    if params.internal_cv_skipFeatureTraining
        % precompute kernels
        [params,added_modalities] = get_precomputed_kernels(params,train_inds);
    end
    
    % get train/test split for train_inds in internal space
    [internal_tuning.int_train_inds,internal_tuning.int_test_inds] = KFold(params.internal_cv,1:length(train_inds));
    
    fprintf('...Fold %i: Tuning hyperparameters with %i folds, %i parameter combinations and %i permutations (total %i evaluations)\n',...
        params.current_fold,length(internal_tuning.int_train_inds),length(model_paramsets),size(internal_tuning.Y_train_all,2)-1,size(internal_tuning.Y_train_all,2)*length(internal_tuning.int_train_inds)*length(model_paramsets));
        
    % initialize error matrix, dimension SPLITS x PARAMETERS x TARGETS
    internal_tuning.errors = nan(length(internal_tuning.int_train_inds),length(model_paramsets),size(internal_tuning.Y_train_all,2));
    
    % loop over tuning folds
    for internal_fold = 1:length(internal_tuning.int_train_inds)
        % indices in full space, needed to get correct data
        internal_tuning.ext_int_train_inds = train_inds(internal_tuning.int_train_inds{internal_fold});
        internal_tuning.ext_int_test_inds = train_inds(internal_tuning.int_test_inds{internal_fold});
        % sanity check that no overlap with original test data!
        assert(sum(ismember(internal_tuning.ext_int_train_inds,test_inds))==0 && sum(ismember(internal_tuning.ext_int_test_inds,test_inds))==0,'test and train overlap!');
        
        % get responses
        internal_tuning.Y_train = internal_tuning.Y_train_all(internal_tuning.int_train_inds{internal_fold},:);
        internal_tuning.Y_test = internal_tuning.Y_train_all(internal_tuning.int_test_inds{internal_fold},:);
        
        for internal_loop = 1:length(model_paramsets)            
            if internal_loop==1
                % only compute data for first loop, then only change model parameters
                [fold_results,internal_tuning.X_train_kernels,internal_tuning.X_test_kernels] = process_fold(params,internal_tuning.ext_int_train_inds,internal_tuning.ext_int_test_inds,model_paramsets(internal_loop));
            else
                % we have the data, just change model parameters and re-run
                fold_results = process_fold(params,[],[],model_paramsets(internal_loop),internal_tuning.X_train_kernels,internal_tuning.X_test_kernels,internal_tuning.Y_train,internal_tuning.Y_test);
            end
            internal_tuning.errors(internal_fold,internal_loop,:)=fold_results.error;
        end
    end
    errors_sum = squeeze(sum(internal_tuning.errors)); % sum or errors over folds
    model_paramsets = model_paramsets(argmin(errors_sum)); % get parameters with smallest total error
    print_best_params(params.current_fold,model_paramsets{1}); % print on screen
    params=clear_precomputed_kernels(params,added_modalities); % clear any precomputed kernels (Important!)
    clear internal_tuning added_modalities errors_sum
end

if is_internal==1
    assert(length(model_paramsets)==1,sprintf('For internal runs only one model_paramsets allowed (given %i)!',length(model_paramsets)));
    model_paramsets = model_paramsets{1};   
else
    assert(length(model_paramsets) == params.permutations+1,'must have length(model_paramsets) == params.permutations+1 !');
end

if nargin<8 % if not given, need to compute kernels
    % get kernels and internal indices (test data always last rows/cols)
    [~,X_kernels,internal_train_inds,internal_test_inds] = get_data(params,train_inds,test_inds,[],is_internal==0);
    
    % get responses, these are in external coordinates!
    Y_train = params.Y(train_inds,:);
    Y_test = params.Y(test_inds,:);
   
    % preprocess kernels (fast, no need to precompute)
    [X_train_kernels,X_test_kernels] = center_and_normalize_kernel(params,X_kernels,internal_train_inds,internal_test_inds);       
end

% first make dummy prediction to compare against
Y_test_predicted_null = repmat(make_prediction_dummy(Y_train,params.problem_type),[size(Y_test,1),1]);%ones(size(Y_test))*make_prediction_dummy(Y_train,params.problem_type);

% get real MKL prediction
assert(nnz(isnan(X_train_kernels))==0,'Training kernels contain NaNs! Check you data!');
assert(nnz(isnan(X_test_kernels))==0,'Test kernels contain NaNs! Check you data!');

% run MKL algorithm, return predictions and kernel weights
[Y_test_predicted,kernel_weights] = make_prediction(params,model_paramsets,X_train_kernels,Y_train,X_test_kernels);

% weights must sum in to 1!
assert(abs( sum(kernel_weights) - 1.0 )<0.0001,'Failed kernel sum check (not equal to 1)! Should never happen!');

% get errors and their ratio
errors = get_prediction_error(Y_test,Y_test_predicted,params.problem_type);
error_nulls = get_prediction_error(Y_test,Y_test_predicted_null,params.problem_type);
error_ratios = errors./error_nulls; % this is for RMSE

% pool results
results.error = errors;
results.Y_test_real = Y_test;
results.Y_test_predicted=Y_test_predicted;
results.Y_test_predicted_null=Y_test_predicted_null;
results.error_ratio=error_ratios;
results.kernel_weights=kernel_weights;
results.model_paramsets = model_paramsets;
results.test_samples = size(Y_test,1);

if is_internal==0
    % only print for external fold
    fprintf('>>>> Fold %i: Final result (%i samples), error %f (null %f), error ratio %f (took %.1fmin)\n',params.current_fold,results.test_samples,errors(1),error_nulls(1),error_ratios(1),toc(start_time)/60);
end

end

%%%%%%%
% ADDITONAL HELPER FUNCTIONS, NOT NEEDED IN LEARNING
%%%%%%%

% ROI-wrapper of load_nii to load data from mask, useful for ROI analysis
function [ts_data,PATTERN_mask_ind] = load_nii_mask(niifile,mask,N_max_volumes,target_volumes)
nii=load_nii_hdr(niifile);
data_siz = nii.dime.dim(2:4);
if ~iscell(mask)
    a=mask;
    clear mask;
    mask{1}=a;
end
% prepare mask indices
N_masks = length(mask);
str = '[';
for i=1:N_masks    
    mask_siz{i} = size(mask{i});        
    if length(mask_siz{i})==3
        PATTERN_mask_ind{i} = find(mask{i});
    else
        PATTERN_mask_ind{i} = mask{i};
    end    
    str = [str,num2str(length(PATTERN_mask_ind{i})),', '];        
    if length(mask_siz{i})==3
        if nnz(data_siz-mask_siz{i})>0
            error('Mask %i has different dimension',i);
        end
    end
end
str = [str(1:end-2),']'];
if length(str)>75
    str = [str(1:70),'...]'];
end
N_vols=get_nii_frame(niifile);
if nargin<4
    target_volumes=1:N_vols;
end
assert(N_vols>=target_volumes(end));
N_vols = min(N_vols,length(target_volumes));

% segments to read
N_read_parts = ceil(N_vols/N_max_volumes);
segments = round(linspace(1,N_vols,N_read_parts+1));
segments(end)=segments(end)+1;
%fprintf('Mask size %s, reading data (in %i parts)... ',str,N_read_parts)
allvols=[];
% initialize timeseires
for i=1:N_masks
    ts_data{i}=zeros(N_vols,length(PATTERN_mask_ind{i}));
end
% read timeseries
for k=1:N_read_parts    
    vols = segments(k):(segments(k+1)-1); % indices to target_volumes
    allvols = [allvols,target_volumes(vols)];
    try
        nii=load_nii(niifile,target_volumes(vols)); % get actual volumes
    catch err        
        nii=load_untouch_nii(niifile,target_volumes(vols)); % get actual volumes
        warning('!!!!!!!!!!!!! Raw data matrix used. Left-right flipping of the matrix possible! %s !!!!!!!!!!!!!!',err.message);
    end
    data = nii.img;
    siz=size(data);
    if length(siz)<4 % is 3D, set volumes = 1
        siz(4)=1;
    end
    data=reshape(data,[],siz(4))';
    
    for i=1:N_masks
        ts_data{i}(vols,:)=data(:,PATTERN_mask_ind{i});
    end
    clear data nii;    
end
if length(allvols)~=N_vols || nnz(target_volumes-allvols)>0
    error('Volume sanity check failed!')
end
%fprintf(' done!\n')
end

%% ALL REMAINING FUNCTIONS USED FOR TAIL APPROXIMATIONS OF P-VALS

function P = compute_Pareto(G,Gdist,TAIL)
% G = real values
% Gdist = null values
% TAIL = which tail of interest (left or right)
if strcmp(TAIL,'left')
    [P,apar,kpar,upar] = palm_pareto(G,Gdist(:),true,0.10,false);
elseif strcmp(TAIL,'right')
    [P,apar,kpar,upar] = palm_pareto(G,Gdist(:),false,0.10,false);
else
    error('Wrong TAIL input, must be left or right');
end

end

function [P,apar,kpar,upar] = palm_pareto(G,Gdist,rev,Pthr,G1out)
% Compute the p-values for a set of statistics G, taking
% as reference a set of observed values for G, from which
% the empirical cumulative distribution function (cdf) is
% generated. If the p-values are below Pthr, these are
% refined further using a tail approximation from the
% Generalised Pareto Distribution (GPD).
%
% Usage:
% P = palm_pareto(G,Gdist,rev,Pthr)
%
% Inputs:
% G      : Vector of Nx1 statistics to be converted to p-values
% Gdist  : A Mx1 vector of observed values for the same statistic
%          from which the empirical cdf is build and p-values
%          obtained. It doesn't have to be sorted.
% rev    : If true, indicates that the smallest values in G and
%          Gvals, rather than the largest, are the most significant.
% Pthr   : P-values below this will be refined using GPD tail.
% G1out  : Boolean indicating whether G1 should be removed from the null
%          distribution.
%
% Output:
% P      : P-values.
% apar   : Scale parameter of the GPD.
% kpar   : Shape parameter of the GPD.
% upar   : Location parameter of the GPD.
%
% For a complete description, see:
% * Winkler AM, Ridgway GR, Douaud G, Nichols TE, Smith SM.
%   Faster permutation inference in brain imaging.
%   Neuroimage. 2016 Jun 7;141:502-516.
%   http://dx.doi.org/10.1016/j.neuroimage.2016.05.068
% 
% This function is based on the following papers:
% * Knijnenburg TA, Wessels LFA, Reinders MJT, Shmulevich I. Fewer
%   permutations, more accurate P-values. Bioinformatics.
%   2009;25(12):i161-8.
% * Hosking JRM, Wallis JR. Parameter and quantile estimation for
%   the generalized Pareto distribution. Technometrics.
%   1987;29:339-349.
% * Grimshaw SD. Computing Maximum Likelihood Estimates for the
%   Generalized Pareto Distribution. Technometrics. 1993;35:185-191.
% * Choulakian V, Stephens MA. Goodness-of-Fit Tests for the
%   Generalized Pareto Distribution. Technometrics. 2001;43(4):478-484.
% 
% Also based on the tool released by Theo Knijnenburg, available at:
% https://sites.google.com/site/shmulevichgroup/people/theo-knijnenburg
% 
% _____________________________________
% Anderson M. Winkler
% FMRIB / University of Oxford
% Mar/2015
% http://brainder.org

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% PALM -- Permutation Analysis of Linear Models
% Copyright (C) 2015 Anderson M. Winkler
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

% Compute the usual permutation p-values.
if G1out,
    Gdist = Gdist(2:end,:);
end
P    = palm_datapval(G,Gdist,rev);
Pidx = P < Pthr; % don't replace this "<" for "<=".

% If some of these are small (as specified by the user), these
% will be approximated via the GPD tail.
if any(Pidx),
    
    % Number of permutations & distribution CDF
    nP   = size(Gdist,1);
    if rev,
        [~,Gdist,Gcdf] = palm_competitive(Gdist,'descend',true);
    else
        [~,Gdist,Gcdf] = palm_competitive(Gdist,'ascend',true);
    end
    Gcdf = Gcdf/nP;
    
    % Keep adjusting until the fit is good. Change the step to 10 to get
    % the same result as Knijnenburg et al.
    Q  = (751:10:999)/1000;
    nQ = numel(Q);
    q  = 1;
    Ptail = NaN;
    while any(isnan(Ptail)) && q < nQ-1,

        % Get the tail
        qidx  = Gcdf >= Q(q);
        Gtail = Gdist(qidx);
        qi    = find(qidx,1);
        if qi == 1,
            upar = Gdist(qi) - mean(Gdist(qi:qi+1));
        else
            upar = mean(Gdist(qi-1:qi));
        end
        if rev,
            ytail = upar - Gtail;
            y     = upar - G((G < upar) & Pidx);
        else
            ytail = Gtail - upar;
            y     = G((G > upar) & Pidx) - upar;
        end
        
        % Estimate the distribution parameters. See §3.2 of Hosking &
        % Wallis (1987). Compared to the usual GPD parameterisation, 
        % here k = shape (xi), and a = scale.
        x    = mean(ytail);
        s2   = var(ytail);
        apar = x*(x^2/s2 + 1)/2;
        kpar =   (x^2/s2 - 1)/2;
        
        % Check if the fitness is good
        A2pval = andersondarling(gpdpvals(ytail,apar,kpar),kpar);
            
        % If yes, keep. If not, try again with the next quantile.
        if A2pval > .05;
            cte = numel(Gtail)/nP;
            Ptail = cte*gpdpvals(y,apar,kpar);
        else
            q = q + 1;
        end
    end
    
    % Replace the permutation p-value for the approximated
    % p-value
    if ~ isnan(Ptail),
        if rev,
            P((G < upar) & Pidx) = Ptail;
        else
            P((G > upar) & Pidx) = Ptail;
        end
    end
end
end

% ==============================================================
function p = gpdpvals(x,a,k)
% Compute the p-values for a GPD with parameters a (scale)
% and k (shape).
if abs(k) < eps;
    p = exp(-x/a);
else
    p = (1 - k*x/a).^(1/k);
end
if k > 0;
    p(x > a/k) = 0;
end
end

% ==============================================================
function A2pval = andersondarling(z,k)
% Compute the Anderson-Darling statistic and return an
% approximated p-value based on the tables provided in:
% * Choulakian V, Stephens M A. Goodness-of-Fit Tests
%   for the Generalized Pareto Distribution. Technometrics.
%   2001;43(4):478-484.

% This is Table 2 of the paper (for Case 3, in which 
% a and k are unknown, bold values only)
ktable = [0.9 0.5 0.2 0.1 0 -0.1 -0.2 -0.3 -0.4 -0.5]';
ptable = [0.5 0.25 0.1 0.05 0.025 0.01 0.005 0.001];
A2table = [ ...
    0.3390 0.4710 0.6410 0.7710 0.9050 1.0860 1.2260 1.5590
    0.3560 0.4990 0.6850 0.8300 0.9780 1.1800 1.3360 1.7070
    0.3760 0.5340 0.7410 0.9030 1.0690 1.2960 1.4710 1.8930
    0.3860 0.5500 0.7660 0.9350 1.1100 1.3480 1.5320 1.9660
    0.3970 0.5690 0.7960 0.9740 1.1580 1.4090 1.6030 2.0640
    0.4100 0.5910 0.8310 1.0200 1.2150 1.4810 1.6870 2.1760
    0.4260 0.6170 0.8730 1.0740 1.2830 1.5670 1.7880 2.3140
    0.4450 0.6490 0.9240 1.1400 1.3650 1.6720 1.9090 2.4750
    0.4680 0.6880 0.9850 1.2210 1.4650 1.7990 2.0580 2.6740
    0.4960 0.7350 1.0610 1.3210 1.5900 1.9580 2.2430 2.9220];

% The p-values are already sorted
k  = max(0.5,k);
z  = flipud(z)';
n  = numel(z);
j  = 1:n;

% Anderson-Darling statistic and p-value:
A2 = -n -(1/n)*((2*j-1)*(log(z) + log(1-z(n+1-j)))');
i1 = interp1(ktable,A2table,k,'linear','extrap');
i2 = interp1(i1,ptable,A2,'linear','extrap');
A2pval = max(min(i2,1),0);

end

function pvals = palm_datapval(G,Gvals,rev)
% Compute the p-values for a set of statistics G, taking
% as reference a set of observed values for G, from which
% the empirical cumulative distribution function (cdf) is
% generated, or using a custom cdf.
%
% Usage:
% pvals = palm_datapval(G,Gvals,rev)
%
% Inputs:
% G     : Vector of Nx1 statistics to be converted to p-values
% Gvals : A Mx1 vector of observed values for the same statistic
%         from which the empirical cdf is build and p-values
%         obtained. It doesn't have to be sorted.
% rev   : If true, indicates that the smallest values in G and
%         Gvals, rather than the largest, are the most significant.
%
% Output:
% pvals : P-values.
%
% This function is a simplification of the much more generic
% 'cdftool.m' so that only the 'data' option is retained.
% To increase speed, there is no argument checking.
%
% _____________________________________
% Anderson M. Winkler
% FMRIB / Univ. of Oxford
% Jul/2012 (1st version)
% Jan/2014 (this version)
% http://brainder.org

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% PALM -- Permutation Analysis of Linear Models
% Copyright (C) 2015 Anderson M. Winkler
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if rev, % if small G are significant
    
    % Sort the data and compute the empirical distribution
    [~,cdfG,distp] = palm_competitive(Gvals(:),'ascend',true);
    cdfG  = unique(cdfG);
    distp = unique(distp)./numel(Gvals);
    
    % Convert the data to p-values
    pvals = zeros(size(G));
    for g = 1:numel(cdfG),
        pvals(G >= cdfG(g)) = distp(g);
    end
    pvals(G > cdfG(end)) = 1;
    
else % if large G are significant (typical case)
    
    % Sort the data and compute the empirical distribution
    [~,cdfG,distp] = palm_competitive(Gvals(:),'descend',true);
    cdfG  = unique(cdfG);
    distp = flipud(unique(distp))./numel(Gvals);
    
    % Convert the data to p-values
    pvals = zeros(size(G));
    for g = numel(cdfG):-1:1,
        pvals(G <= cdfG(g)) = distp(g);
    end
    pvals(G > cdfG(end)) = 0;
end

end

function [unsrtR,S,srtR] = palm_competitive(X,ord,mod)
% Sort a set of values and return their competition
% ranks, i.e., 1224, or the modified competition ranks,
% i.e. 1334. This makes difference only when there are
% ties in the data. The function returns the ranks in
% their original order as well as sorted.
% 
% Usage:
% [unsrtR,S,srtR] = palm_competitive(X,ord,mod)
% 
% Inputs:
% - X      : 2D array with the original data. The
%            function operates on columns. To operate
%            on rows or other dimensions, use transpose
%            or permute the array's higher dimensions.
% - ord    : Sort as 'ascend' (default) or 'descend'.
% - mod    : If true, returns the modified competition
%            ranks, i.e., 1334. This is the
%            correct for p-values and cdf. Otherwise
%            returns standard competition ranks.
% 
% Outputs:
% - unsrtR : Competitive ranks in the original order.
% - S      : Sorted values, just as in 'sort'.
% - srtR   : Competitive ranks sorted as in S.
%
% Examples:
% - To obtain the empirical cdf of a dataset in X, use:
%   cdf   = palm_competitive(X,'ascend',true)/size(X,1);
% - To obtain the empirical p-values for each value in X, use:
%   pvals = palm_competitive(X,'descend',true)/size(X,1);
% 
% _____________________________________
% Anderson M. Winkler
% FMRIB / University of Oxford
% Nov/2012
% http://brainder.org

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% PALM -- Permutation Analysis of Linear Models
% Copyright (C) 2015 Anderson M. Winkler
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

% Check inputs
if nargin < 1 || nargin > 3,
    error('Incorrect number of arguments.');
elseif nargin == 1,
    ord = 'ascend';
    mod = false;
elseif nargin == 2,
    mod = false;
end

% Important: The function starts by computing the
% unmodified competition ranking. This can be
% ascending or descending. However, note that the
% modified ranking for the ascending uses the
% unmodified descending, whereas the modified
% descending uses the modified ascending, hence
% the need to "reverse" the inputs below.
if mod,
    if strcmpi(ord,'ascend'),
        ord = 'descend';
    elseif strcmpi(ord,'descend'),
        ord = 'ascend';
    end
end

% Unmodified competition ranking
[nR,nC] = size(X);
unsrtR  = single(zeros(size(X)));
[S,tmp] = sort(X,ord);
[~,rev] = sort(tmp);
srtR = repmat((1:nR)',[1 nC]);
for c = 1:nC, % loop over columns

    % Check for +Inf and -Inf and replace them
    % for a value just higher or smaller than
    % the max or min, respectively.
    infpos = isinf(S(:,c)) & S(:,c) > 0;
    infneg = isinf(S(:,c)) & S(:,c) < 0;
    if any(infpos),
        S(infpos,c) = max(S(~infpos,c)) + 1;
    end
    if any(infneg),
        S(infneg,c) = min(S(~infneg,c)) - 1;
    end
    
    % Do the actual sorting, checking for obnoxious NaNs
    dd = diff(S(:,c));
    if any(isnan(dd)),
        error(['Data cannot be sorted. Check for NaNs that might be present,\n', ...
            'or precision issues that may cause over/underflow.\n', ...
            'If you are using "-approx tail", consider adding "-nouncorrected".%s'],'');
    end
    f = find([false; ~logical(dd)]);
    for pos = 1:numel(f),
        srtR(f(pos),c) = srtR(f(pos)-1,c);
    end
    unsrtR(:,c) = single(srtR(rev(:,c),c)); % original order as the data
    
    % Put the infinities back
    if any(infpos),
        S(infpos,c) = +Inf;
    end
    if any(infneg),
        S(infneg,c) = -Inf;
    end
end

% Prepare the outputs for the modified rankings, i.e.,
% flip the sorted values and ranks
if mod,
    
    % Do the actual modification
    unsrtR = nR - unsrtR + 1;
    
    % Flip outputs
    if nargout >= 2,
        S = flipud(S);
    end
    if nargout == 3,
        srtR = flipud(nR - srtR + 1);
    end
end

end
