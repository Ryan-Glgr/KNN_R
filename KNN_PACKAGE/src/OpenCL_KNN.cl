
// Define type T, defaults to float
#ifndef T
#define T float
#endif

__kernel void fillDistanceMatrix(__global const T* xGroup,
                                 __global T* distanceMatrix,
                                 const int groupSize)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < groupSize && j < groupSize) {
        distanceMatrix[i * groupSize + j] = fabs(xGroup[i] - xGroup[j]);
        
    }
}

__kernel void kth_element(__global T* distanceMatrix,
                          __global T* result,
                                 const int groupSize,
                                 const int K)
{

    // Since distanceMatrix is 1D memory in OpenCL, i and groupSize determines the row starting position

    // Declare local variables
    __local int lowerBound;
    __local int upperBound;
    __local int size;
    __local int l;
    __local int j;
    __local int pivotIndex;
    __local T pivot;
    __local T Ri;

    // Get the current thread number (iteration in the "loop")
    const int i = get_global_id(0);

    // Clamp k to the max group size
    const int kClamped = (K > groupSize - 1) ? (groupSize - 1) : K;
    // Adjusts k to the correct position in the array
    const int kAdjusted = kClamped + i * groupSize;

    // ------------------------ START QUICKSELECT ------------------------ //

    // Gets the first and last indexes of the current grou
    lowerBound = groupSize * i;
    upperBound = groupSize * i + groupSize - 1;

    // Iterate until the lowerBound and upperBound are the same
    while ( lowerBound <= upperBound ) {
        
        // --------------- START PARTITION --------------- //
        
        size = upperBound - lowerBound + 1;
        pivot = distanceMatrix[upperBound];

        l = lowerBound - 1;
        // Partitioning (Lomuto's Algorithm)
        j = lowerBound;
        while (j < upperBound) {
        
            // If the current is less than the pivot
            // Swap i and the current element
            // And move i up to the next element
            // (Essentially move an element from the right to the left,
            // And then move to the element which will be swapped next)
            if (distanceMatrix[j] < pivot) {
                l++;
                T temp = distanceMatrix[l];
                distanceMatrix[l] = distanceMatrix[j];
                distanceMatrix[j] = temp;
            }

            j++;
        }

        // Swap the last non-swapped on the left and the pivot
        T temp = distanceMatrix[l+1];
        distanceMatrix[l+1] = distanceMatrix[upperBound];
        distanceMatrix[upperBound] = temp;

        // ---------------- END PARTITION ---------------- //

        // Get the current pivot index
        pivotIndex = l + 1;

        // Change the bounds if the pivot is not equal to k
        if (pivotIndex == kAdjusted) {
            break;    
        } else if (pivotIndex > kAdjusted) {
            upperBound = pivotIndex - 1;
        } else {
            lowerBound = pivotIndex + 1;
        }

    }

    // ------------------------- END QUICKSELECT ------------------------- //


    // After QuickSelect, store the found element in Ri
    // Calculate row element
    Ri = distanceMatrix[kAdjusted];
    // Calculate the result to be read back to the program
    result[i] = kClamped / (groupSize * 2.0f * Ri);

}