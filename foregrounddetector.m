function [foreground, gmmMU, gmmSigma, gmmMC, allHistograms] = foregrounddetector(grayFrame, allHistograms, gmmMU, gmmSigma, gmmMC, K)
[r, c, d] = size(grayFrame);
background  = uint8(zeros(r, c));

for i = 1 : r
    for j = 1 : c
        for k = 1:d
            % Compute the distance between pixel value X and the means of the GMM
            X = double(grayFrame(i, j));
            mu = gmmMU(:, i, j);
            sigma = gmmSigma(:, i, j);
            dist=sqrt(sum((X - mu).*(X - mu))./sigma);
                    
            % Find the Gaussian component with the minimum distance
            [minDist, idx] = min(dist);
            s = sqrt(sigma(idx));

            % Check if the pixel value belongs to the selected Gaussian component
                
            if minDist < 2.5*sqrt(s)
                mc = gmmMC(idx, i, j, k);
                if mc/s > .001
                    background(i, j) = 1;      
                end
            end

            vec1 = allHistograms(:, i, j, k);
            vec2 = vec1;
            vec2(X + 1) = vec2(X + 1) + 1;
            if sqrt(sum((vec1/sum(vec1) - vec2/sum(vec2)) .^ 2)) > .009
                y = expandHist(vec2);
                [mu1, sigma1, p1] = fitGMM(y', K,200);
                gmmMU(:, i, j, k)=mu1';
                gmmSigma(:, i, j, k)=sigma1;
                gmmMC(:, i, j, k)=p1;
            end
        end          
    end
end
foreground = 1- background;
foreground = logical(foreground);

