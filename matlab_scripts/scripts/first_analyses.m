addpath("C:\Users\ponce\OneDrive\Desktop\exp_set\matlab_scripts\support-files") % adds the function sem
if ~exist('Stats','var')
    load("N:\Data-Ephys-MAT\Caos-26082025-002_Stats.mat",'data'); % loads the datamat
    Stats = data; % structure with fields related to the experiment
    disp(Stats) 
    clear data
    load("N:\Data-Ephys-MAT\Diablito-26082025-002_rasters.mat")
    rasters = data; % chan x time x images
    clear data
end