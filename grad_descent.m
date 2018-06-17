function grad_descent()
clear all
close all
clc
step=0.001;
gama=0.99;
threshold=10^(-4);

% Create grid
x=-1:0.05:1;
y=-1:0.05:1;
[X Y] = meshgrid(x,y);

% Define our function
myFun = @(a,b) 80*(a.^4)+0.01*(b.^6);
myFun_dx = @(a,b) 320*(a.^3);
myFun_dy = @(a,b) 0.06*(b.^5);
myFun_dxdx = @(a,b) 960*(a.^2);
myFun_dydy = @(a,b) 0.3*(b.^4);
myFun_dxdy = 0;

s=surf(X,Y,myFun(X,Y));
hold on;
%s.EdgeColor = 'none';
meshgrid off

coord = zeros(100000,2);

% Starting coordinates are (1,1)
coord(1,:) = [1 1];
plot3(coord(1,1),coord(1,2),myFun(coord(1,1),coord(1,2)),'m*','MarkerSize',20);

momentum_x = 0;
momentum_y = 0;

i=2;
while(1)
    % Gradient descent: calculate delta - Wt+1 = Wt + dWt ;
    % dWt=-step*grad(func)
    %delta_x = - step * myFun_dx(coord(i-1,1),coord(i-1,2));
    %delta_y = - step * myFun_dy(coord(i-1,1),coord(i-1,2));
    
    % Gradient descent: Newton func (Multiplying by Hassian)
    %delta_x = - 1/myFun_dxdx(coord(i-1,1),coord(i-1,2)) * myFun_dx(coord(i-1,1),coord(i-1,2));
    %delta_y = - 1/myFun_dydy(coord(i-1,1),coord(i-1,2)) * myFun_dy(coord(i-1,1),coord(i-1,2));
    
    % Gradient descent: Momentum
    momentum_x = gama * momentum_x + step * myFun_dx(coord(i-1,1),coord(i-1,2));
    momentum_y = gama * momentum_y + step * myFun_dy(coord(i-1,1),coord(i-1,2));
    delta_x = -momentum_x; 
    delta_y = -momentum_y;
    
    % Gradient descent: Update coordinates
    coord(i,:) = coord(i-1,:)+[delta_x,delta_y];
    
    % Plot the points on the graph.
    plot3(coord(i,1),coord(i,2),myFun(coord(i,1),coord(i,2)),'m*','MarkerSize',20)
    i=i+1  
    
    if sqrt(delta_x^2 + delta_y^2) < threshold
        break;
    end
end

title(['Gradient descent: ' num2str(i) ' Iterations']);
xlabel('X')
ylabel('Y')

end

