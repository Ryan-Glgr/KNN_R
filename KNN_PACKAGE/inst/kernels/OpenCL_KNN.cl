
// Define type T, defaults to float
#ifndef T
#define T float
#endif

__kernel void fillDistanceMatrix(__global const T* xGroup, __global T* distanceMatrix, const int groupSize)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < groupSize && j < groupSize) {
        distanceMatrix[i * groupSize + j] = fabs(xGroup[i] - xGroup[j]);
        
    }
}

__kernel void kth_element(__global T* distanceMatrix, __global T* result, const int groupSize, const int K)
{

    // Since distanceMatrix is 1D memory in OpenCL, i and groupSize determines the row starting position

    // Get the current thread number (iteration in the "loop")
    int i = get_global_id(0);

    // Clamp k to the max group size
    int kClamped = (K > groupSize - 1) ? (groupSize - 1) : K;
    // Adjusts k to the correct position in the array
    int kAdjusted = kClamped + i * groupSize;

    // ------------------------ START QUICKSELECT ------------------------ //

    // Gets the first and last indexes of the current group
    int lowerBound = groupSize * i;
    int upperBound = groupSize * i + groupSize - 1;

    // Iterate until the lowerBound and upperBound are the same
    while ( lowerBound <= upperBound ) {
        
        // --------------- START PARTITION --------------- //
        
        int size = upperBound - lowerBound + 1;
        T pivot = distanceMatrix[upperBound];

        int i = lowerBound - 1;
        // Partitioning (Lomuto's Algorithm)
        for (int j = lowerBound; j < upperBound; j++) {
        
            // If the current is less than the pivot
            // Swap i and the current element
            // And move i up to the next element
            // (Essentially move an element from the right to the left,
            // And then move to the element which will be swapped next)
            if (distanceMatrix[j] < pivot) {
                i++;
                T temp = distanceMatrix[i];
                distanceMatrix[i] = distanceMatrix[j];
                distanceMatrix[j] = temp;
            }

        }

        // Swap the last non-swapped on the left and the pivot
        T temp = distanceMatrix[i+1];
        distanceMatrix[i+1] = distanceMatrix[upperBound];
        distanceMatrix[upperBound] = temp;

        // ---------------- END PARTITION ---------------- //

        // Get the current pivot index
        int pivotIndex = i + 1;

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
    float Ri = distanceMatrix[kAdjusted];
    // Calculate the result to be read back to the program
    result[i] = kClamped / (groupSize * 2.0f * Ri);
}