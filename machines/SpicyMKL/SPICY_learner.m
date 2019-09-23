function output = SPICY_learner(d,def,args,args_optional,problem_type)

ktrain = zeros(size(d.train,2),size(d.train,1),size(d.train,3));
ktest = zeros(size(d.train,2),size(d.test,1),size(d.train,3));
for k = 1:size(d.train,3)
    ktrain(:,:,k) =  (d.train(:,:,k))';
    ktest(:,:,k) =  (d.test(:,:,k))';
end

options.stopdualitygap = 1;
options.stopIneqViolation = 0;
options.tolOuter = 0.01;
options.tolInner = 0.0001;
options.display=1;
options.regname = 'elasticnet';

if strcmp(problem_type,'regression')
    m = mean(d.tr_targets);  % mean of the training data
    tr_targets = d.tr_targets - m; % mean centre targets
    options.loss = 'square';
else
    % change targets from 1/2 to -1/1 
    tr_targets = d.tr_targets;
    c1PredIdx  = tr_targets  ==1; 
    tr_targets(c1PredIdx)  = 1; %positive values = 1 
    tr_targets(~c1PredIdx) = -1; %negative values = 2 
    m= mode(tr_targets);
    options.loss = 'logit';
end

try
    [alpha,beta,b,activeset,posind,params,story] = SpicyMKL(d.train,tr_targets,args(2)*[args(1),1-args(1)],options);        
    ktest_final = zeros(length(d.te_targets),length(d.tr_targets));
    for i = 1:size(d.train,3)
        ktest_final = ktest_final + beta(i)*d.test(:,:,i);
    end    
    func_val = ((ktest_final*alpha)+b); % add mean from the training set    
    beta = abs(beta);
catch err
    warning('Failed to solve spicyMKL! Setting predictions to dummy!! (C=%f)',args);
    beta = ones(1,size(d.train,3))/length(size(d.train,3));   
    func_val = zeros(size(d.te_targets));
end

% 
if strcmp(problem_type,'regression')
    predictions = func_val + m;
else
    predictions = sign(func_val)+m;
    % change predictions from 1/-1 to 1/2
    c1PredIdx               = predictions==1;
    predictions(c1PredIdx)  = 1; %positive values = 1
    predictions(~c1PredIdx) = 2; %negative values = 2    
end

%-------------------------------------------------------------------------
output.predictions = predictions;
output.func_val    = func_val;
output.type        = problem_type;
output.alpha       = alpha;
output.b           = nan;%b;
output.totalSV     = nan;%length(alpha_sv);
output.beta        = beta(:)'/sum(beta); %kernel weights

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
