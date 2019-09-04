function subW=upgradeW(subW,sublearningRate,subEpsi,subX)

for level=1:size(subW,2)
    for perceptronId=1:size(subW{level},1)
        for eachDataAttId=1:size(subX{level},1)
            
            subW{level}(perceptronId,eachDataAttId)=subW{level}(perceptronId,eachDataAttId)-sublearningRate*subEpsi{level}(1,perceptronId)*subX{level}(eachDataAttId,1);
    
        end
    end
end

end
