function rbfTry()

    
    csvData = csvread('abalone_clean.csv');
    
    inputs = csvData(:, 1:8);
    outputs = csvData(:, 9:11);
    
    %inputs2 = zeros(size(inputs,1), size(inputs,2));
    outputs2 = zeros(size(outputs,1), 1);
   
    for i = 1:size(outputs,1)
        lne = outputs(i,:);
        
        if lne(1) == 0 && lne(2) ==0 && lne(3) == 1
            outputs2(i) = 1;
        elseif lne(1) == 0 && lne(2) == 1 && lne(3) == 0
            %outputs2(i) = [];
            %inputs(i) = [];
        elseif lne(1) == 1 && lne(2) == 0 && lne(3) == 0
            outputs2(i) = 3;
        end
    end
    
    
    t = cputime;
    svm = svmtrain(inputs, outputs2);
    
    testedGrps = svmclassify(svm, inputs);
    
    results = sum(abs(outputs2 - testedGrps)) / size(outputs2,1)
  
    cputime - t
    
end