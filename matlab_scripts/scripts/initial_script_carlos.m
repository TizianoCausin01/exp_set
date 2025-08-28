addpath("C:\Users\ponce\OneDrive\Desktop\exp_set\matlab_scripts\support-files") % adds the function sem
if ~exist('Stats','var')
    load("N:\Data-Ephys-MAT\Diablito-11082025-002_Stats.mat",'data'); % loads the datamat
    Stats = data; % structure with fields related to the experiment
    disp(Stats) 
    clear data
    load("N:\Data-Ephys-MAT\Diablito-11082025-002_rasters.mat")
    rasters = data; % chan x time x images
    clear data
end
fnames = Stats.TunCurve_pics; % cell array with filenames of presented pictures
hasAlexNet = contains(fnames,'AlexNet'); % checking for the alexnet images
hasResNet = contains(fnames,'ResNet50'); % checking for the resnet images
hasBoth = hasAlexNet & hasResNet ;
hasAlexNetOnly = hasAlexNet & ~hasResNet;
hasResNetOnly = ~hasAlexNet & hasResNet;
isArray1 = ismember(Stats.spikeID,1:32); % checks for the channels of one array (ismember because it's not said it's in order I suppose)
isArray2 = ismember(Stats.spikeID,33:64); % checks for the other array
% from rasters, takes only the neurons related to Array1 or Array2 and the
% images selected by alexnet or resnet
resp_mean_alexnet_array1 = mean(mean(rasters(isArray1,:,hasAlexNetOnly),3),1); % average over the images (3) and then over the channels (1)
resp_sem_alexnet_array1 = sem(mean(rasters(isArray1,:,hasAlexNetOnly),3),1); % average over the images (3) and then sem over the channels (1)
resp_mean_alexnet_array2 = mean(mean(rasters(isArray2,:,hasAlexNetOnly),3),1); % average over the images (3) and then over the channels (1)
resp_sem_alexnet_array2 = sem(mean(rasters(isArray2,:,hasAlexNetOnly),3),1); % average over the images (3) and then sem over the channels (1)
resp_mean_resnet_array1 = mean(mean(rasters(isArray1,:,hasResNetOnly),3),1); % average over the images (3) and then over the channels (1)
resp_sem_resnet_array1 = sem(mean(rasters(isArray1,:,hasResNetOnly),3),1); % average over the images (3) and then sem over the channels (1) 
%%
if ~exist('myDS','var')
    myDS = imageDatastore(Stats.picLoc); % creates a data loader with the pics
end
[~,fnamesInDS] = fileparts(myDS.Files) ; % takes the filenames
hasAlexNetInDS = contains(fnamesInDS,'AlexNet'); 
hasResNetInDS = contains(fnamesInDS,'ResNet50');
hasAlexNetOnlyInDS = hasAlexNetInDS & ~hasResNetInDS;
hasResNetOnlyInDS = ~hasAlexNetInDS & hasResNetInDS;
alexnetOnlyDS = subset(myDS,find(hasAlexNetOnlyInDS));
resnetOnlyDS = subset(myDS,find(hasResNetOnlyInDS));

figure
nexttile
imagesc(mean(rasters,3)); colorbar
hc = colorbar; ylabel(hc,'response')
title('raw responses')
xlabel('time from image onset')
ylabel('channel')
nexttile
imagesc(zscore(mean(rasters,3),0,2));
title('z-scored responses')
hc = colorbar; ylabel(hc,'response')
xlabel('time from image onset')
ylabel('channel')

nexttile
h1 = plot(mean(mean(rasters(isArray1,:,:),3),1),'r')
hold on
h2 = plot(mean(mean(rasters(isArray2,:,:),3),1),'k')
title('raster plot')
xlabel('time from image onset')
ylabel('response')
%legend([h1 h2], {'Array1', 'Array2'}, 'Location', 'best')
nexttile
imagesc(Stats.TunCurve(:,:,1)); colorbar
title('tuning curve of neurons')
xlabel('images')
ylabel('channels')
h1 = colorbar; ylabel(hc,'response'); 
nexttile
h1 = shadedErrorBar(1:size(rasters,2),resp_mean_alexnet_array1,resp_sem_alexnet_array1,'r',1)
hold on
h2 = shadedErrorBar(1:size(rasters,2),resp_mean_alexnet_array2,resp_sem_alexnet_array2,'r',1);
set(h2.mainLine,'Lines','--')
xlabel('time from image onset')
ylabel('response')
h3 = shadedErrorBar(1:size(rasters,2),resp_mean_resnet_array1,resp_sem_resnet_array1,'m',1)
%legend([h1.mainLine, h2.mainLine, h3.mainLine], ...
       % {'Array1 (alexnet)', 'Array2 (alexnet)', 'Array1 (resnet)'}, ...
       % 'Location', 'best')
% nexttile
% montage(alexnetOnlyDS)
% title('AlexNet images')
% nexttile
% montage(resnetOnlyDS)
% title('resnet')
