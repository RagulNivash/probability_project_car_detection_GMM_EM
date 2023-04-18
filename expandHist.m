function out = expandHist(hist)
[x, id] = find(hist ~= 0);
out = [];
for i = 1 : length(x)
    out = [out, repmat((x(i)-1), 1, hist(x(i)))];
end

