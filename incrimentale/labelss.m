function [ label ] = labelss( labels )
%LABELS Summary of this function goes here
%   Detailed explanation goes here
label=[];
for i = 1:length(labels)
    if(strcmp(labels(i),'t')>0)
        label(i)=1;
    else
        label(i)=0;
    end
end

end

