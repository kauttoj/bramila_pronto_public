% load parameter sets and extract best results
function best_results = bramila_pronto_SLURM_resultparser(RESULT_FOLDER,best_results_in)
% input
%  RESULT_FOLDER = folder with result .mat files
%  best_results_in = existing best_results structure (skip extraction)

% make this global so it can be accessed afterwards
global best_results

N_workers = 4; % if more than one, will use parfor

if nargin==0
    RESULT_FOLDER = '/m/nbe/scratch/empathy_and_emotions/Janne_analysis/bramila_pronto_analysis/WITH_RATINGS/final_stats/';
end

clc;
if nargin<2
    %%---- read result files ----------------------------
    
    result_files = dir([RESULT_FOLDER,'results_*.mat']);
    
    N_files = length(result_files);
    
    assert(N_files>30,'Less than 30 files found! Check you folder.');
    
    result_files_temp = cell(1,N_files);
    for file_num = 1:N_files
        result_files_temp{file_num} = [result_files(file_num).folder,filesep,result_files(file_num).name];
    end
    result_files=result_files_temp;
    result_files = result_files(randperm(N_files)); % just for fun, not needed
    
    fprintf('Found %i files, finding best results\n',N_files);
    
    if N_workers>1
        %% USE MULTIPLE WORKERS
        % start pool and get real number of workers
        N_workers = start_pool(N_workers);
        
        % divide files for workers
        subset_ind = round(linspace(1,N_files+1,N_workers+1));
        result_files_sets = cell(1,N_workers);
        for subset = 1:N_workers
            result_files_sets{subset} = result_files(subset_ind(subset):(subset_ind(subset+1)-1));
        end               
        
        % find best results for subsets
        fprintf('Starting top results search with %i subsets (local pooling)\n',N_workers);
        best_results_subset = cell(1,N_workers);
        parfor subset = 1:N_workers      
            best_results_subset{subset}=[];
            for file_num = 1:length(result_files_sets{subset})
                filename = result_files_sets{subset}{file_num};
                A=[];
                try
                    A = load(filename,'params','results');
                catch err
                    warning(sprintf('Failed to read file %s (%s)',filename,err.message));
                    continue;
                end
                if isfield(A,'results')                                    
                    %
                    % ADD CONDITIONS HERE (e.g., fixed parameters)
                    %       
                    if not(A.results.params.smoothed_data == 1) || not(A.results.params.use_mean_data == 1),continue,end;
					
                    best_results_subset{subset} = update_results(best_results_subset{subset},A);
                else
                    warning(sprintf('Failed to read file %s (%s)',filename,'no results field'));
                end
            end            
        end
        
        fprintf('Starting top results search over subsets (global pooling)\n');
        % find best results over subsets
        best_results = best_results_subset{1}; % set first set as the best and try to improve that
        for subset = 2:N_workers      
            all_targets = fields(best_results_subset{subset});
            for target = all_targets'
                target = target{1};
                all_emotions = fields(best_results_subset{subset}.(target));
                for emotion = all_emotions'
                    emotion = emotion{1};
                    try
                        % if exists and is better, update
                        if best_results.(target).(emotion).error_ratio_mse>best_results_subset{subset}.(target).(emotion).error_ratio_mse
                            best_results.(target).(emotion) = best_results_subset{subset}.(target).(emotion);
                        end
                    catch
                        % does not exist yet, add it
                        best_results.(target).(emotion) = best_results_subset{subset}.(target).(emotion);
                    end                    
                end
            end
        end                                            
    else
        %% USE SINGLE WORKER
        fprintf('Starting top results search (global pooling)\n');
        best_results=[];
        for file_num = 1:N_files
            if mod(file_num+1,100)==0
                fprintf('...reading file %i of %i\n',file_num,length(result_files));
            end
            filename = result_files{file_num};
            try
                A = load(filename,'params','results');
            catch err
                warning(sprintf('Failed to read file %s (%s)',filename,err.message));
                continue;
            end
            if isfield(A,'results')                           
                %
                % ADD CONDITIONS HERE (e.g., fixed parameters)
                %                                       
                if 1
                    best_results = update_results(best_results,A);
                end
            else
                warning(sprintf('Failed to read file %s (%s)',filename,'no results field'));
            end
        end
    end
    fprintf('All done! Listing best results:\n\n');    
else
    best_results = best_results_in;
end

%%---- print results -----------

PLOT_TARGET = 'correlation';my_ylim = [0,0.85];
%PLOT_TARGET = 'MSE_ratio';my_ylim = [0,0.65];

assert(~isempty(best_results),'best_results is empty! Nothing to print.');

addpath('/m/nbe/scratch/empathy_and_emotions/shared_codes');
get_colors;

all_targets = sort(fields(best_results));
N_grid = ceil(sqrt(length(all_targets))); % how many subplots we need

close all;
fig = figure('position',[55          67        1256         841]);
my_axes = tight_subplot(fig,N_grid,N_grid,[0.15,0.05],[0.15,0.05],[0.05,0.02]);
for target_num = 1:length(all_targets)
    all_emotions = sort(fields(best_results.(all_targets{target_num})));
    fprintf('\nTarget = ''%s''\n',all_targets{target_num});
    
    y_data = [];
    x_labels = [];
    pval_data=[];
    for emotion_num = 1:length(all_emotions)
        x = best_results.(all_targets{target_num}).(all_emotions{emotion_num}).error_ratio_mse;
        s = '....';
        if x<0.80
            s='--->';
        end
        
        try
            lab = best_results.(all_targets{target_num}).(all_emotions{emotion_num}).params.parcellation_label_chosen;
        catch
            lab = best_results.(all_targets{target_num}).(all_emotions{emotion_num}).params.parcellation_label;
        end
        
        % collect data
        if strcmp(PLOT_TARGET,'MSE_ratio')
            y_data(end+1)=1-x;
            pval_data(end+1)=best_results.(all_targets{target_num}).(all_emotions{emotion_num}).pvals.error_Pareto;
        elseif strcmp(PLOT_TARGET,'correlation')
            y_data(end+1)=best_results.(all_targets{target_num}).(all_emotions{emotion_num}).R;
            pval_data(end+1)=best_results.(all_targets{target_num}).(all_emotions{emotion_num}).pvals.R_Pareto;
        else
           error('Unknown performance metrics!');
        end        
        x_labels{end+1}=all_emotions{emotion_num};
        
        fprintf('%s ''%s'': mse_ratio = %f, R = %f, R2 = %f [smoothed=%i, averaged=%i (%i frames), parcellation=%s, kernel={%s}]\n',...
            s,...
            all_emotions{emotion_num},...
            x,...
            best_results.(all_targets{target_num}).(all_emotions{emotion_num}).R,...
            best_results.(all_targets{target_num}).(all_emotions{emotion_num}).R2,...
            best_results.(all_targets{target_num}).(all_emotions{emotion_num}).params.smoothed_data,...
            best_results.(all_targets{target_num}).(all_emotions{emotion_num}).params.use_mean_data,...
            best_results.(all_targets{target_num}).(all_emotions{emotion_num}).params.top_frame_count,...
            lab,...
            cell2str(best_results.(all_targets{target_num}).(all_emotions{emotion_num}).params.kernels)...
            );
    end    
    % plot this target
%     subplot(N_grid,N_grid,target_num);
%     bar(1:length(y_data),y_data);
%     set(gca,'xtick',1:length(y_data),'xticklabels',x_labels,'xticklabelrotation',-50);
%     ylabel('1-MSE_{ratio}','interpreter','tex');    
%     title(sprintf('target=%s',all_targets{target_num}),'interpreter','none');    

    %subplot(N_grid,N_grid,target_num);
    pval_data_fdr = mafdr(pval_data,'bhfdr',true);
    for k=1:length(y_data)        
        h=bar(k,y_data(k),'parent',my_axes(target_num));hold on;
        h.set('facecolor',COLORS.(x_labels{k})/255);
        p = pval_data_fdr(k);
        te = '';
        if p<0.005
            te = '**';
        elseif p<0.05
            te = '*';
        end
        text(k,y_data(k)+0.020,te,'horizontalalignment','center','fontsize',12,'parent',my_axes(target_num));
    end
    set(my_axes(target_num),'ylim',my_ylim);
    set(my_axes(target_num),'xtick',1:length(y_data),'xticklabels',x_labels,'xticklabelrotation',-50,'ticklabelinterpreter','none','fontsize',11);
    title(my_axes(target_num),sprintf('target=%s',all_targets{target_num}),'interpreter','none');    
    if strcmp(PLOT_TARGET,'MSE_ratio')
        ylabel(my_axes(target_num),'1-MSE_{ratio}','interpreter','tex','fontsize',14);
    elseif strcmp(PLOT_TARGET,'correlation')
        ylabel(my_axes(target_num),'Correlation','interpreter','tex','fontsize',14);
    else
        error('Unknown performance metrics!');
    end

end

fprintf('All done! To access ''best_results'', type ''global best_results''\n');

end

function best_results = update_results(best_results,A)
% update top results
target = A.params.target;
emotion = A.params.emotion;
try
    if best_results.(target).(emotion).error_ratio_mse>A.results.error_ratio_mse
        % new value is better, update
        best_results.(target).(emotion).error_ratio_mse = A.results.error_ratio_mse;
        best_results.(target).(emotion).R2 = A.results.R2;
        best_results.(target).(emotion).R = A.results.R;
        best_results.(target).(emotion).params = A.params;
        if isfield(A.results,'pvals') && ~isempty(A.results.pvals)
            best_results.(target).(emotion).pvals  = A.results.pvals;
        end
    end
catch err
    % if here, there was not existing value so we add the current value
    best_results.(target).(emotion).error_ratio_mse = A.results.error_ratio_mse;
    best_results.(target).(emotion).R2 = A.results.R2;
    best_results.(target).(emotion).R = A.results.R;
    best_results.(target).(emotion).params = A.params;
    if isfield(A.results,'pvals') && ~isempty(A.results.pvals)
        best_results.(target).(emotion).pvals  = A.results.pvals;
    end
end

end

function str = cell2str(cellstr)
%CELL2STR Summary of this function goes here
%   Detailed explanation goes here
str = [];
for i=1:length(cellstr)
    if ~ischar(cellstr{i})
        error('Input cell includes non-string data!')
    end
    if i>1
        str = [str,' ',cellstr{i}];
    else
        str = cellstr{i};
    end
end
end

function workers=start_pool(workers)
% start local pool with specific number of workers
if workers>1
    mycluster = gcp('nocreate');
    skip=0;
    if ~isempty(mycluster)
        if mycluster.NumWorkers~=workers
            delete(mycluster);
        else
            skip=1;
        end
    end
    if skip==0
        mycluster=parcluster('local');
        mycluster.NumWorkers = max(1,workers);        
        parpool(mycluster);
    end
    workers = mycluster.NumWorkers;
end

end
