function h = myhist(x)
h = zeros(1, 256);
for i=1:length(x)
    h(x(i)+1) = h(x(i)+1) + 1;
end