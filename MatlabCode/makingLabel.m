function label=makingLabel(list)
label=[];
temp=[];

for i=1:size(list,1)
        fileName=char(list(i));
        k=strfind(fileName,'down');
        if(k>=1)
            temp=1;
            label=[label;temp];
        end
        if(sum(k)==0)
            temp=0;
            label=[label;temp];
        end
    end
end