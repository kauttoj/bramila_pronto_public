% load parameter sets and submit as individual jobs
function jobname = bramila_pronto_SLURM_jobsubmitter(param_file)

LOCAL_TEST = 0; % 1 for debugging, run code locally without really submitting (FOR DEBUGGING)
MAX_JOBS_SUBMIT = 40000; % how many jobs to submit at once
% location of bramila_pronto_run.m
code_path = '/m/nbe/scratch/empathy_and_emotions/Janne_analysis/bramila_pronto_analysis';

warning('on','all');
assert(exist([code_path,filesep,'bramila_pronto_run.m'],'file')>0,'Function ''bramila_pronto_run.m'' not found!');

root_path = fileparts(which(param_file));
param_file = [root_path,filesep,param_file];
fprintf('loading data %s\n',param_file);
load(param_file);

assert(exist('all_paramsets','var')>0,'Parameters not loaded!');

n_jobs = length(all_paramsets);

% randomize job order!
rng(666);
all_paramsets = all_paramsets(randperm(n_jobs));

fprintf('found total %i parameter sets to submit\n',n_jobs);

fprintf('starting submitting loop\n');
% initialize arrays
jobname = cell(1,n_jobs);
logfile = cell(1,n_jobs);
count=0;
% prepare and send jobs
for kk = 1:n_jobs
    if count>MAX_JOBS_SUBMIT-1
        fprintf('Maximum number of submissions reached (%i), terminating loop\n',MAX_JOBS_SUBMIT);
        break;
    end
    params = all_paramsets{kk};
    resultfile = params.result_file_name;    
    if exist(resultfile,'file')>0
        continue;
    end
    jobfile = [resultfile(1:end-4),'_JOB.sh'];    
    MAX_mem = 20000; % default max memory
    if params.use_mean_data==0
        MAX_mem = 50000; % more memory when not averaging
    end
    MAX_time=10;
    count=count+1;
    
    if ~isfield(params,'workers')
        params.workers=12;
    end
    
    function_command = sprintf('bramila_pronto_run(''%s'')',params.param_file);
    [jobname{kk},logfile{kk}] = sendjob(jobfile,function_command,code_path,LOCAL_TEST,MAX_mem,MAX_time,params.workers);    
    if mod(count+1,50)==0
        fprintf('...submitted job %i of %i (index %i)\n',count,n_jobs,kk);
    end    
end    
fprintf('Finished! Submitted %i jobs in total\n',count);

end

function [jobname,logfile] = sendjob(filename,function_command,codepath,doLocal,maxmem,maxtime,maxcores)
% write and send job (or run locally)

if nargin<5
    doLocal=0;
    maxcores=1;
    maxmem=25000;
    maxtime=4;
end

logfile = [filename(1:end-6),'LOG.log'];

maxmem = ceil(maxmem/maxcores); % cannot use one limit for all, need to split!

dlmwrite(filename, '#!/bin/sh', '');
dlmwrite(filename, '#SBATCH -p batch','-append','delimiter','');
dlmwrite(filename, sprintf('#SBATCH -t %.2d:00:00',maxtime),'-append','delimiter','');
dlmwrite(filename, '#SBATCH --nodes=1','-append','delimiter','');
dlmwrite(filename, '#SBATCH --ntasks=1','-append','delimiter','');
dlmwrite(filename,['#SBATCH --cpus-per-task=',num2str(maxcores)],'-append','delimiter','');
dlmwrite(filename, '#SBATCH --qos=normal','-append','delimiter','');
dlmwrite(filename, ['#SBATCH -o "' logfile '"'],'-append','delimiter','');
dlmwrite(filename, sprintf('#SBATCH --mem-per-cpu=%i',maxmem),'-append','delimiter','');
dlmwrite(filename, 'hostname; date;','-append','delimiter','');
dlmwrite(filename, 'module load matlab/r2018b','-append','delimiter','');
dlmwrite(filename,sprintf('srun matlab_multithread -nosplash -nodisplay -nodesktop -r "cd(''%s'');%s;exit;"',codepath,function_command),'-append','delimiter','');

jobname = 'unknown';

if doLocal==1
    %%%% FOR TESTING AND DEBUGGING ONLY - run locally in serial manner
    warning('RUNNING JOB LOCALLY AS PART OF DEBUGGING');
    command = sprintf('%s;',function_command);
    eval(command);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else
    [a,b]=unix(['sbatch ' filename]);
    s = 'Submitted batch job ';
    k = strfind(b,s);
    if ~isempty(k)
        jobname = strtrim(b(length(s):end));
    end
end

end

