__kernel void fillDistanceMatrix(__global const float* xGroup,
                                 __global float* distanceMatrix,
                                 const int groupSize)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < groupSize && j < groupSize) {
        distanceMatrix[i * groupSize + j] = fabs(xGroup[i] - xGroup[j]);
    }
}