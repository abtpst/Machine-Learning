disp('===========================Results for Part A=========================')

NumberOfWords = numel (vocab)

First50 = vocab(1:50,1)

TrainSpamPercent = (size(find(trainLabels))/(size(trainLabels)))*100

TrainHamPercent = 100-TrainSpamPercent

ValSpamPercent = (size(find(valLabels))/(size(valLabels)))*100

ValHamPercent = 100-ValSpamPercent

TestSpamPercent = (size(find(testLabels))/(size(testLabels)))*100

TestHamPercent = 100-TestSpamPercent

NumberOfEmptyColumns = numel(find(sum(trainFeat)==0))

CountOfSuccessInSpams = det(sum(bsxfun(@times, trainLabels, trainFeat(:,find(strcmp(vocab,'success'))))))

FractionOfSpamsContainingSuccess = numel(find(bsxfun(@times, trainLabels, trainFeat(:,find(strcmp(vocab,'success'))))))/numel(find(trainLabels == 1))

disp('===========================Results for Part B=========================')

unsmoothedSpamLogProb = [];

unsmoothedHamLogProb = [];

smoothedSpamLogProb = [];

smoothedHamLogProb = [];

for i = 1:numel(vocab)
    
    totalsum = sum(trainFeat(:,i));
    
    spamsum = det(sum(bsxfun(@times, trainLabels, trainFeat(:,i))));
    
    hamsum = totalsum-spamsum;
    
    smoothedspamsum = spamsum+0.1;
    
    smoothedhamsum = hamsum+0.1;
    
    smoothedtotalsum = totalsum+15896.3;
    
    xspam=spamsum/totalsum;
   
    xham=hamsum/totalsum;
    
    if(xspam<eps)
        xspam=eps;
    end
    
    if(xham<eps)
        xham=eps;
    end
    
    yspam=smoothedspamsum/smoothedtotalsum;
    
    yham=smoothedhamsum/smoothedtotalsum;
    
    if(yspam<eps)
        yspam=eps;
    end
    
    if(yham<eps)
        yham=eps;
    end
    
    unsmoothedSpamLogProb = [unsmoothedSpamLogProb,(log(xspam)+log(0.5093))];

    smoothedSpamLogProb = [smoothedSpamLogProb,(log(yspam)+log(0.5093))];

    unsmoothedHamLogProb = [unsmoothedHamLogProb,(log(xham)+log(0.4907))];

    smoothedHamLogProb = [smoothedHamLogProb,(log(yham)+log(0.4907))];

end

 smoothedTrainingAccuracy = 1-(numel(find(((trainFeat*(smoothedSpamLogProb'))>(trainFeat*(smoothedHamLogProb')))~=trainLabels)))/numel(trainLabels)
 
 unsmoothedTrainingAccuracy = 1-(numel(find(((trainFeat*(unsmoothedSpamLogProb'))>(trainFeat*(unsmoothedHamLogProb')))~=trainLabels)))/numel(trainLabels)
 
 unsmoothedTestAccuracy = 1-(numel(find(((testFeat*(unsmoothedSpamLogProb'))>(testFeat*(unsmoothedHamLogProb')))~=testLabels)))/numel(testLabels)
 
 smoothedTestAccuracy = 1-(numel(find(((testFeat*(smoothedSpamLogProb'))>(testFeat*(smoothedHamLogProb')))~=testLabels)))/numel(testLabels)
 
disp('===========================Results for Part C=========================')

enbefore = entropy(trainLabels);

Successposits= find(trainFeat(:,14));
Successnegits= find(trainFeat(:,14)==0);
SuccessposSplit = trainLabels(Successposits);
SuccessnegSplit = trainLabels(Successnegits);

IgainOfSuccess = enbefore-((numel(Successposits)/numel(trainLabels))*entropy(SuccessposSplit)+(numel(Successnegits)/numel(trainLabels))*entropy(SuccessnegSplit))

Igain=[];

for i=1:size(vocab)

    posits= find(trainFeat(:,i));
    
    negits= find(trainFeat(:,i)==0);
    
    posSplit = trainLabels(posits);
    
    negSplit = trainLabels(negits);

    e1=0;e2=0;

if(~isempty(posSplit))
e1=entropy(posSplit);
end

if(~isempty(negSplit))
e2=entropy(negSplit);
end

Igain=[Igain, (enbefore-((numel(posits)/numel(trainLabels))*e1+(numel(negits)/numel(trainLabels))*e2))];

end

[sortedValues,sortIndex] = sort(Igain(:),'descend'); 

max10Indices = sortIndex(1:10);
max50Indices = sortIndex(1:50);
max100Indices = sortIndex(1:100);
max200Indices = sortIndex(1:200);

Top10Words = vocab(max10Indices)

smoothedSpam10LogProb = [];

smoothedHam10LogProb = [];

for i = 1:numel(max10Indices)
    
    totalsum = det(sum(trainFeat(:,max10Indices(i))));
    
    spamsum = det(sum(bsxfun(@times, trainLabels, trainFeat(:,max10Indices(i)))));
    
    hamsum = totalsum-spamsum;
    
    smoothedspamsum = spamsum+0.1;
    
    smoothedhamsum = hamsum+0.1;
    
    smoothedtotalsum = totalsum+1;
    
    yspam=smoothedspamsum/smoothedtotalsum;
    
    yham=smoothedhamsum/smoothedtotalsum;
    
    if(yspam<eps)
        yspam=eps;
    end
    
    if(yham<eps)
        yham=eps;
    end
    
    
    smoothedSpam10LogProb = [smoothedSpam10LogProb,(log(yspam)+log(0.5093))];

    smoothedHam10LogProb = [smoothedHam10LogProb,(log(yham)+log(0.4907))];

  end

smoothedSpam50LogProb = [];

smoothedHam50LogProb = [];


for i = 1:numel(max50Indices)
    
    totalsum = det(sum(trainFeat(:,max50Indices(i))));
    
    spamsum = det(sum(bsxfun(@times, trainLabels, trainFeat(:,max50Indices(i)))));
    
    hamsum = totalsum-spamsum;
    
    smoothedspamsum = spamsum+0.1;
    
    smoothedhamsum = hamsum+0.1;
    
    smoothedtotalsum = totalsum+5;
    
    yspam=smoothedspamsum/smoothedtotalsum;
    
    yham=smoothedhamsum/smoothedtotalsum;
    
    if(yspam<eps)
        yspam=eps;
    end
    
    if(yham<eps)
        yham=eps;
    end
    
    
    smoothedSpam50LogProb = [smoothedSpam50LogProb,(log(yspam)+log(0.5093))];

    smoothedHam50LogProb = [smoothedHam50LogProb,(log(yham)+log(0.4907))];

 end

smoothedSpam100LogProb = [];

smoothedHam100LogProb = [];

for i = 1:numel(max100Indices)
    
    totalsum = det(sum(trainFeat(:,max100Indices(i))));
    
    spamsum = det(sum(bsxfun(@times, trainLabels, trainFeat(:,max100Indices(i)))));
    
    hamsum = totalsum-spamsum;
    
    smoothedspamsum = spamsum+0.1;
    
    smoothedhamsum = hamsum+0.1;
    
    smoothedtotalsum = totalsum+10;
    
    yspam=smoothedspamsum/smoothedtotalsum;
    
    yham=smoothedhamsum/smoothedtotalsum;
    
    if(yspam<eps)
        yspam=eps;
    end
    
    if(yham<eps)
        yham=eps;
    end
    
    
    smoothedSpam100LogProb = [smoothedSpam100LogProb,(log(yspam)+log(0.5093))];

    smoothedHam100LogProb = [smoothedHam100LogProb,(log(yham)+log(0.4907))];

 end

smoothedSpam200LogProb = [];

smoothedHam200LogProb = [];

for i = 1:numel(max200Indices)
    
    totalsum = det(sum(trainFeat(:,max200Indices(i))));
    
    spamsum = det(sum(bsxfun(@times, trainLabels, trainFeat(:,max200Indices(i)))));
    
    hamsum = totalsum-spamsum;
    
    smoothedspamsum = spamsum+0.1;
    
    smoothedhamsum = hamsum+0.1;
    
    smoothedtotalsum = totalsum+20;
    
    yspam=smoothedspamsum/smoothedtotalsum;
    
    yham=smoothedhamsum/smoothedtotalsum;
    
    if(yspam<eps)
        yspam=eps;
    end
    
    if(yham<eps)
        yham=eps;
    end
    
    
    smoothedSpam200LogProb = [smoothedSpam200LogProb,(log(yspam)+log(0.5093))];

    smoothedHam200LogProb = [smoothedHam200LogProb,(log(yham)+log(0.4907))];

end

Accuracy10 = 1-(numel(find((((valFeat(:,max10Indices))*(smoothedSpam10LogProb'))>((valFeat(:,max10Indices))*(smoothedHam10LogProb')))~=valLabels)))/numel(valLabels)

Accuracy50 = 1-(numel(find((((valFeat(:,max50Indices))*(smoothedSpam50LogProb'))>((valFeat(:,max50Indices))*(smoothedHam50LogProb')))~=valLabels)))/numel(valLabels)

Accuracy100 = 1-(numel(find((((valFeat(:,max100Indices))*(smoothedSpam100LogProb'))>((valFeat(:,max100Indices))*(smoothedHam100LogProb')))~=valLabels)))/numel(valLabels)

Accuracy200 = 1-(numel(find((((valFeat(:,max200Indices))*(smoothedSpam200LogProb'))>((valFeat(:,max200Indices))*(smoothedHam200LogProb')))~=valLabels)))/numel(valLabels)

TestAccuracy200 = 1-(numel(find((((testFeat(:,max100Indices))*(smoothedSpam100LogProb'))>((testFeat(:,max100Indices))*(smoothedHam100LogProb')))~=testLabels)))/numel(testLabels)

