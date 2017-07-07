function lambda = lambda_minim(lambda_vect,error_val)
    m = size(error_val,1);
    mini = error_val(1);
    index = 1;
    for i = 2 : m
        if(mini > error_val(i))
            mini = error_val(i);
            index = i;
        end
    end
    lambda = lambda_vect(index);
end