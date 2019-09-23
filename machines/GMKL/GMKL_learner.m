function output = GMKL_learner(d,def,args,args_optional,problem_type)
% Generalized Multiple Kernel Learning wrapper function.
%
% This function shows how to call COMPGDoptimize. The code is general
% and you can plug in your own kernel function and regularizer. To
% illustrate, I've included code for computing sums and products of
% RBF kernels across features based on both feature vectors and
% precomputed kernel matrices.
%
% MATLAB's quadprog will only work for small data sets. Also, I've
% only provide code for C-SVC. Note that you must have the
% optimization toolbox installed before you can use quadprog.
%
% If you would like to use LIBSVM (for SVR/SVC) instead, please
% set parms.TYPEsolver='LIB'. In this case, you must have the
% appropriate (modified) svmtrain.mexxxx in your path. Compiled 32 and
% 64 bit binaries for Windows and Linux are provided in
% ../LIBSVM/Binaries but you'll need to check whether they're compatible
% with your version of MATLAB. The modified source code is provided in
% ../LIBSVM/Code/libsvm-2.84-1 and you can compile that directly if you
% prefer.
%
% You can also incorporate your favourite SVM solver by editing
% COMPGDoptimize.m.
%
% Code written by Manik Varma.

global Kprecomp;

if nargin==0
    load('temp_params');
end
% The following code shows how to call COMPGDoptimize with
% precomputed matrices as input

ktrain = zeros(size(d.train,2),size(d.train,1),size(d.train,3));
ktest = zeros(size(d.train,2),size(d.test,1),size(d.train,3));
for k = 1:size(d.train,3)
    ktrain(:,:,k) =  (d.train(:,:,k))';
    ktest(:,:,k) =  (d.test(:,:,k))';
end

if strcmp(problem_type,'regression')
    m = mean(d.tr_targets);  % mean of the training data
    tr_targets = d.tr_targets - m; % mean centre targets
else
    % change targets from 1/2 to -1/1    
    tr_targets = d.tr_targets;
    c1PredIdx  = tr_targets  ==1; 
    tr_targets(c1PredIdx)  = 1; %positive values = 1 
    tr_targets(~c1PredIdx) = -1; %negative values = 2 
    m = mode(tr_targets);
end

Kprecomp=ktrain;
TYPEreg=0; % L1
TYPEker=2;
NUMkernels = size(ktrain,3);
NUMtrainpts = size(ktrain,1);
NUMtestpts = size(ktest,2);

parms=initparms(NUMkernels,TYPEker,TYPEreg);
parms.BOOLverbose=0;
parms.C = args;
%   features - NUMdims x NUMtrainpts matrix of training feature
%              vectors. If you are using precomputed kernels then
%              set features=[1:NUMtrainpts];
%   labels   - NUMtrainpts x 1 vector of training labels. For
%              classification, use +1 and -1 for the positive and
%              negative class respectively.
%   parms    - structure specifying algorithm parameters. Look at
%              PGDwrapper for details on setting these parameters.
try
    svm=COMPGDoptimize(1:NUMtrainpts,tr_targets,parms);
    
    % test data.
    Kprecomp=ktest;
    K=parms.fncK(svm.svind,1:NUMtestpts,parms,svm.d);
    
    func_val=(svm.b+svm.alphay'*K)';
    beta = abs(svm.d);
catch err
    warning('Failed to solve GMKL! Setting predictions to dummy!! (C=%f)',parms.C);
    beta = ones(1,size(d.train,3))/length(size(d.train,3));   
    func_val = zeros(size(d.te_targets))+m;
end

% 
if strcmp(problem_type,'regression')
    predictions = func_val + m;
else
    predictions = sign(func_val);
    % change predictions from 1/-1 to 1/2
    c1PredIdx               = predictions==1;
    predictions(c1PredIdx)  = 1; %positive values = 1
    predictions(~c1PredIdx) = 2; %negative values = 2    
end

%-------------------------------------------------------------------------
output.predictions = predictions;
output.func_val    = func_val;
output.type        = problem_type;
output.alpha       = svm.d;
output.b           = nan;%b;
output.totalSV     = nan;%length(alpha_sv);
output.beta        = beta(:)'/sum(beta); %kernel weights

clear -global 

end

function parms=initparms(NUMk,TYPEker,TYPEreg)
% Kernel parameters
switch (TYPEker),
    case 0, parms.KERname='Sum of RBF kernels across features';
        parms.gamma=ones(NUMk,1);
        parms.fncK=@KSumRBF;
        parms.fncKdash=@KdashSumRBF;
        parms.fncKdashint=@KdashSumRBFintermediate;
    case 1, parms.KERname='Product of RBF kernels across features';
        parms.gamma=1;
        parms.fncK=@KProdRBF;
        parms.fncKdash=@KdashProdRBF;
        parms.fncKdashint=@KdashProdRBFintermediate;
    case 2, parms.KERname='Sum of precomputed kernels';
        parms.fncK=@KSumPrecomp;
        parms.fncKdash=@KdashSumPrecomp;
        parms.fncKdashint=@KdashSumPrecompintermediate;
    case 3, parms.KERname='Product of exponential kernels of precomputed disance matrices';
        parms.gamma=1;
        parms.fncK=@KProdExpPrecomp;
        parms.fncKdash=@KdashProdExpPrecomp;
        parms.fncKdashint=@KdashProdExpPrecompintermediate;
    otherwise, fprintf('Unknown kernel type.\n'); keyboard;
end;

% Regularization parameters
switch (TYPEreg),
    case 0, parms.REGname='l1';          % L1 Regularization
        parms.sigma=ones(NUMk,1);
        parms.fncR=@Rl1;
        parms.fncRdash=@Rdashl1;
    case 1, parms.REGname='l2';          % L2 Regularization
        parms.mud=ones(NUMk,1);
        parms.covd=1e-1*eye(NUMk);
        parms.invcovd=inv(parms.covd);
        parms.fncR=@Rl2;
        parms.fncRdash=@Rdashl2;
    otherwise, fprintf('Unknown regularisation type.\n'); keyboard;
end;

% Standard SVM parameters
parms.TYPEprob=0;           % 0 = C-SVC, 3 = EPS-SVR (try others at your own risk)
parms.C=10;                 % Misclassification penalty for SVC/SVR ('-c' in libSVM)

% Gradient descent parameters
parms.initd=rand(NUMk,1);   % Starting point for gradient descent
parms.TYPEsolver='MAT';     % Use Matlab's quadprog as an SVM solver.
parms.TYPEstep=1;           % 0 = Armijo, 1 = Variant Armijo, 2 = Hessian (not yet implemented)
parms.MAXITER=40;           % Maximum number of gradient descent iterations
parms.MAXEVAL=200;          % Maximum number of SVM evaluations
parms.MAXSUBEVAL=20;        % Maximum number of SVM evaluations in any line search
parms.SIGMATOL=0.3;         % Needed by Armijo, variant Armijo for line search
parms.BETAUP=2.1;           % Needed by Armijo, variant Armijo for line search
parms.BETADN=0.3;           % Needed by variant Armijo for line search
parms.BOOLverbose=true;     % Print debug information at each iteration
parms.SQRBETAUP=parms.BETAUP*parms.BETAUP;
end

function r=Rl1(parms,d), r=parms.sigma'*d; end
function rdash=Rdashl1(parms,d), rdash=parms.sigma; end

function r=Rl2(parms,d), delta=parms.mud-d;r=0.5*delta'*parms.invcovd*delta; end
function rdash=Rdashl2(parms,d), rdash=parms.invcovd*(d-parms.mud); end

function K=KSumPrecomp(TRNfeatures,TSTfeatures,parms,d)
global Kprecomp;
if (size(Kprecomp,3)~=length(d)), fprintf('Number of kernels does not equal number of weights.\n');keyboard; end;
K=lincomb(Kprecomp(TRNfeatures,TSTfeatures,:),d);
end

function Kdashint=KdashSumPrecompintermediate(TRNfeatures,TSTfeatures,parms,d)
Kdashint=[];
end

function Kdash=KdashSumPrecomp(TRNfeatures,TSTfeatures,parms,d,k,Kdashint)
global Kprecomp;
Kdash=Kprecomp(TRNfeatures,TSTfeatures,k);
end

function K=lincomb(base,d)
nzind=find(d>1e-4);
K=zeros(size(base,1),size(base,2));
if (~isempty(nzind)), for k=1:length(nzind), K=K+d(nzind(k))*base(:,:,nzind(k)); end; end;
end
