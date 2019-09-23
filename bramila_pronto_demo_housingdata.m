% BOSTON housing dataset for testing and sanity checking

clearvars;
close all;
clc;
warning('on','all') % turn all warnings

params = struct(); % create empty parameter set
load('bramila_pronto_defaultparams.mat'); % load default params (to be overwritten) OPTIONAL

%% input data X (neural, structural, etc)

DATA = load([pwd,'\machines\SimpleMKL\housing_data.mat']);

%%--------- Define data and target for the problem Y = f(X) -----------------------
params.emotion = 'house_features'; % source data type ("X")
params.target = 'house_price'; % response label ("Y")
%%--------------------------------------------------------

%% input data Y (response, left hand side of equation)
params.Y = DATA.y;
params.problem_type = 'regression';
params.max_memory_per_cpu = 2000; % how large dataset to keep in memory vs. loading from disk (in megabytes)
% subjects are just samples
params.allsubs=[];
for sub=1:length(params.Y)
    params.allsubs{sub} = sprintf('sub-%i',sub); % samples
end
params.output_folder = [pwd,'\TEMP\'];

%% structural data with FreeSurfer estimated volume sizes
data = single(DATA.X);
elements = numel(data);
file = sprintf('%shouse_data.mat',params.output_folder);
save(file,'data','elements','-v7.3');
params.X.housedata.files = {file}; % enclose in cell
params.X.housedata.preprocessor = 'StandardScaler'; % how to process data across subject for each feature

%% learner algorithm and related params
if 0
    predictor_params.model = 'simpleMKL';
    predictor_params.args.C = [0.001,0.01,0.1,1,10,100,500,800];
elseif 1
    predictor_params.model = 'spicyMKL';
    predictor_params.args.alpha = [0.70,0.95]; % elastic-net ratio, between 0 and 1 (1=full L1)
    predictor_params.args.C = [0.005,0.01,0.1,0.5,0.80]; % between 0 and 1
else    
    predictor_params.model = 'bayesMKL';
    predictor_params.args = []; % no tunable hyperparameters!
end
params.predictor = predictor_params;

%% kernel type or types (pooled together)
% parameters that follow string after '_'
%  order of polynomial (polynomial kernel)
%  gamma scaling factor (radial kernel) in addition to standard 1/#features -scaling
params.kernels = {'correlation_spearman','linear','polynomial_2','polynomial_3','polynomial_4','radial_0.001','radial_0.01','radial_0.1','radial_0.5','radial_1','radial_5','radial_10','radial_100','radial_500','correlation_pearson'};
params.kernel_normalization = 'CenterAndNormalize'; % 'CenterAndNormalize' (recommended! Pronto-style), 'CenterAndTrace', 'Center' or 'None'

%% cross-validation of the model (main loop)
params.cv = 20; % integer or a string 'LOSO' (Leave-One-Subject-Out)

%% internal cross-validation within the main loop
params.internal_cv = 10; % NOTE: Model will be trained total params.cv*params.internal_cv times, keep this at 5-15
params.internal_cv_skipFeatureTraining = 1; % omit feature transformer re-training. Faster, but can be suboptimal! Disable, if possible.

%% how many workers to use (divided over folds), ideally equal or multiple of params.cv
params.workers = 1; % use 1 for debugging

%% local tempdata folder
params.tempdata_root = '/tmp/bramila_pronto_testing/';
params.tempdata_root_overwrite = 1; % overwrite old files, if any (otherwise skip)
params.delete_temp_data=1;

%% run analysis
results = bramila_pronto_run(params);

% example results:
% bayesMKL ---- Final results over 10 folds: Error 3.421237 (dummy 9.207467), R 0.929946, R2 0.864799, Error ratio 0.371572 ----
% simpleMKL ---- Final results over 10 folds: Error 3.664100 (dummy 9.201393), R 0.922618, R2 0.851224, Error ratio 0.398211 ----
% spicyMKL ---- Final results over 10 folds: Error 3.669807 (dummy 9.215265), R 0.918307, R2 0.843288, Error ratio 0.398231 ----

% for comparison, result for 10 folds using simpleMKL example script (not using bramila_pronto code):
% simpleMKL ---- Final results over 10 folds: Error 3.678736 (dummy 9.201910), R 0.921412, R2 0.849000, Error ratio 0.399780 ----