function rbfTry()

    
   
    csvData = csvread('yeast_clean.csv');
    
    inputs = csvData(:, 1:8);
    outputs = csvData(:, 9:18);
    
    %inputs2 = zeros(size(inputs,1), size(inputs,2));
    %outputs2 = zeros(size(outputs,1), 1);
   
    inputs2 = zeros(1,8);
    outputs2 = [];
    count = 1;
 
    for i = 1:size(outputs,1)
        lne = outputs(i,:);
        
        if lne(1) == 1 
            %outputs2(i) = 1;
            inputs2(count,:) = inputs(i,:);
            outputs2(count) = 1;
            count = count + 1;
        elseif lne(2) == 1 
            %outputs2(i) = 2;
            inputs2(count,:) = inputs(i,:);
            outputs2(count) = 2;
            count = count + 1;
        end
    end
    
    t = cputime;
    svm = svmtrain(inputs2, outputs2');
    
    testedGrps = svmclassify(svm, inputs);
    
    results = sum(abs(outputs2' - testedGrps)) / size(outputs2',1)
  
    cputime - t
    
end