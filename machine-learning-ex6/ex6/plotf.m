function   plotf (X_axis, Y_axis,Z_axis)
figure;
title("Error function for cross value");
xlabel("C_val");
ylabel("sigma_val");
surf(X_axis,Y_axis,Z_axis);
end