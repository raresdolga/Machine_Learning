function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure('Name',"Contor plot for 2 features Logistic-regression",'NumberTitle','off');
hold on;
p = find(y==1);
n =find(y==0);
plot(X(p,1),X(p,2),'k+','linewidth',2,'MarkerSize',7), plot(X(n,1),X(n,2),'yo','LineWidth',2,'MarkerSize',7);
% Put some labels 
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;







% =========================================================================



hold off;

end
