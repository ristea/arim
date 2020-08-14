function [lable, distance_label] = get_label(r_axis, distances, ampl)

lable = zeros(1, 2048);
distance_label = zeros(1, 2048);

for i=1:1:length(distances)
    [min_value, min_index] = min(abs(r_axis - distances(i)));
    lable(min_index) = ampl(i);
    distance_label(min_index) = distances(i);
end

end

