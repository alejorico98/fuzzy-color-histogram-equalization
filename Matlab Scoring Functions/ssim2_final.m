clc
clear all
close all

%% Get images path from folders

dataSetDir = fullfile('');

%% Load images
RealDir = fullfile(dataSetDir,'clean');

ResultsDir = fullfile(dataSetDir,'results'); %compare results
%ResultsDir = fullfile(dataSetDir,'hazy'); %compare inputs

%%
realIma = imageDatastore(RealDir);
realIma = realIma.Files;
resultsIma = imageDatastore(ResultsDir);
resultsIma = resultsIma.Files;

%% Get ssim index

ssim_final=[];

for i=1:length(resultsIma) %only the 5 last images for testing
    
    imaReal = imread(realIma{i,1});
    imaResult = imread(resultsIma{i,1});
    
    [mssim, ssim_map] = NTIRE_SSIM_imgs(imaReal, imaResult, 6);
    
    ssim_final(i,1)=(mssim);
    
end

ssim_average=sum(ssim_final)/length(ssim_final) %compute the average









