# bramila_pronto

Matlab software for supervised learning (regression/classification) for multimodal fMRI data using multiple-kernel (e.g., simpleMKL) method. The core design follows closely to Pronto Toolbox by UCL with additional functionalities. The code is minimalistic and self-sufficient, which makes it easy to modify and understand how things work.

---How to use---

bramila_pronto_run.m: The main code for regression/classification, takes parameters as input and performs supervised learning, returns structure with model accuracy etc.

bramila_pronto_demo.m: Set up input data and parameters for single run of "bramila_pronto_run.m" function

bramila_pronto_SLURM_parammaker.m: Create multiple parameter sets as combinations of data and learning parameters, outputs .mat file "bramila_pronto_SLURM_parameters.mat"

bramila_pronto_SLURM_jobsubmitter.m: Reads file "bramila_pronto_SLURM_parameters.mat", makes a job file for each individual parameter set and submits the job to SLURM system

bramila_RSRM_aligner.py: Python code to prepare .h5 files of fMRI data for multiple parcellations and with RSRM alignment (if wanted), recommended over nifti input!


28.4.2019 Janne Kauttonen (initial version)
