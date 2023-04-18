clc;
clear all;
close all;
warning off;
% Read in video file
vid = VideoReader('visiontraffic.avi');
v = VideoWriter('smallSizeVideo.avi');
open(v)
while hasFrame(vid)
    frame = readFrame(vid);
    frame = imresize(frame, 1/2);
    writeVideo(v, frame);
end
close(v)

% Read in video file
vid = VideoReader('smallSizeVideo.avi');
%frame = readFrame(vid);
[r, c, d] = size(frame);
N = 150;
n = 1;
allNFrames = zeros(N, r, c, d);
while hasFrame(vid) && n <= N
    % Read in the current frame
    frame = readFrame(vid);
    allNFrames(n, :, :,:) = frame;
    n = n + 1;
end

allHistograms = zeros(256, r, c, d);
for i = 1 : r
    for j = 1 : c
        for k = 1 : d
            hist = myhist(allNFrames(:,i, j, k));
            allHistograms(:, i, j, k) = double(hist);
        end
    end
end
delete hist;


% Fit 2 Gaussian components to each histogram
K = 2;
gmmMU = zeros(K, r, c, d);
gmmSigma = zeros(K, r, c, d);
gmmMC = zeros(K, r, c, d);
vec1 = zeros(1, N);
for i = 1 : r
     for j = 1 : c
         for k = 1:d
            % Fit Gaussian mixture model to histogram data
            vec = allHistograms(:, i, j, k);
            y = expandHist(vec);
            [mu1, sigma1, p1] = fitGMM(y', K,5);
            gmmMU(:, i, j, k)=mu1;
            gmmSigma(:, i, j, k)=sigma1;
            gmmMC(:, i, j, k)=p1;
        end
    end
end
v = VideoWriter('newfile.avi');
se = strel('square',3);   
   
blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
'AreaOutputPort', false, 'CentroidOutputPort', false, ...
'MinimumBlobArea', 400);

videoPlayer = vision.VideoPlayer('Name', 'Detected Cars');
videoPlayer.Position(3:4) = [650,400];  % window size: [width, height]
    
open(v)
    
while hasFrame(vid) % Read in the current frame
        frame = readFrame(vid);
        %detects the background and removes it to compute foreground
        [foreground, gmmMU, gmmSigma, gmmMC, allHistograms] = foregrounddetector(frame, allHistograms, gmmMU, gmmSigma, gmmMC, K);
        filteredForeground = imopen(foreground, se);
        

     
        % Detect the connected components with the specified minimum area, and
        % compute their bounding boxes
        bbox = step(blobAnalysis, logical(filteredForeground));
        result = insertShape(frame, 'Rectangle', bbox, 'Color', 'green');
        numCars = size(bbox, 1);
        result = insertText(result, [10 10], numCars, 'BoxOpacity', 1, ...
        'FontSize', 14);
        step(videoPlayer, result);  % display the results
        writeVideo(v, result)
end
close(v)
    
