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

psnr_final=[];

for i=1:length(resultsIma)
    
    imaReal = imread(realIma{i,1});
    imaResult = imread(resultsIma{i,1});
    
    res = NTIRE_PeakSNR_imgs(imaReal, imaResult, 6);
    
    psnr_final(i,1)=(res);
    
end

psnr_average=sum(psnr_final)/length(psnr_final) %average
