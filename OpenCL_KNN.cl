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

__kernel void kth_element(__global float* distanceMatrix,
                          __global double* result,
                                 const int groupSize,
                                 const int K)
{

    // Since distanceMatrix is 1D memory in OpenCL, i and groupSize determines the row starting position

    // Get the current thread number (iteration in the "loop")
    int i = get_global_id(0);

    // Clamp k to the max group size
    int kClamped = (K > groupSize - 1) ? (groupSize - 1) : K;
    // Adjusts k to the correct position in the array
    int kAdjusted = kClamped + i * groupSize;

    // ------------------------ START QUICKSELECT ------------------------ //

    // Create starting points
    int pivot = i * groupSize;
    int lowerBound = pivot;
    int upperBound = pivot + groupSize - 1;

    // Partition Counters
    int left;
    int right;

    // Iterate until pivot is equal to kClamped
    while ( lowerBound < upperBound ) {

        // Partitioning (Hoare's Algorithm)
        pivot = lowerBound;
        left = lowerBound - 1;
        right = upperBound + 1;

        // Partition the numbers
        while (left < right) {
            
            // Increment on the left until left >= pivot
            do {
                left++;
            } while (distanceMatrix[left] < distanceMatrix[pivot]);

            // Increment on the right until right <= pivot
            do {
                right--;
            } while (distanceMatrix[right] > distanceMatrix[pivot]);
            
            // Swap if left < right
            if (left < right) {
                float temp = distanceMatrix[left];
                distanceMatrix[left] = distanceMatrix[right];
                distanceMatrix[right] = temp;
            }
        }

        // Sets the new pivot to the right element
        pivot = right;

        // If the pivot is less than k, re-adjust to the upper-bound
        if (pivot < kAdjusted) {
            lowerBound = pivot+1;
        // If the pivot is less than k, re-adjust to the lower-bound
        } else if (pivot > kAdjusted) {
            upperBound = pivot;
        // If the pivot is the k, break the loop
        } else if (pivot == kAdjusted) {
            break;
        }
    }

    // ------------------------- END QUICKSELECT ------------------------- //


    // After QuickSelect, store the found element in Ri
    // Calculate row element
    float Ri = distanceMatrix[kAdjusted];
    // Calculate the result to be read back to the program
    result[i] = kClamped / (groupSize * 2.0f * Ri);

}




// This is the first part of the mean calculations
// Each element will go through and do xi / len(result)
__kernel void partial_means(__global float* result, int size){
    int global_id = get_global_id(0);

    if (global_id < size) {
        result[global_id] /= size;
    }
}


// this is the second part of the mean calculation
// We use reduction to calculate a partial sum of elements until we get small
// enough
__kernel void sum_partial_means(__global float *result, __local float *sharedMem, int size){
    int gid = get_global_id(0);
    int localId = get_local_id(0);
    int groupSize = get_local_size(0);

    // Load data into local (shared in cuda) memory
    if (gid < size){
        sharedMem[localId] = result[gid];
    }
    else {
        // Pad with zero if global id is out of bounds
        sharedMem[localId] = 0.0f; 
    }

    // __syncThreads()
    barrier(CLK_LOCAL_MEM_FENCE);

    // reduction within the workgroup
    for (int stride = 1; stride < groupSize; stride *= 2){

        int index = 2 * stride * localId;
        if (index < groupSize && index + stride < groupSize){
            sharedMem[index] += sharedMem[index + stride];
        }

        // __syncThreads
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this group to global mem
    if (localId == 0){
        result[get_group_id(0)] = sharedMem[0];
    }
}
