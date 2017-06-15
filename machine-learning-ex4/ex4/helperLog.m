function rez = helperLog (param,y)
rez = (y')*log(param) + ((1-y)')*log(1-param);
end