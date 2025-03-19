/**
* @file: OpenCL_KNN.cl
 * @brief: Kernel functions which are called by OpenCL_KNN.cpp.
 * @authors: Ryan Gallagher, Matt Hansen, Austin Snyder
 * @date: 3.18.2025
 */
// Define type T, defaults to float
#ifndef T
#define T float
#endif

/**!
 * @brief Parallelization the distance calculation algorithm.
 *
 * @param xGroup the group of x to compare to itself
 * @param distanceMatrix the array of distances between elements in xGroup
 * @param groupSize the size of the xGroup to be handled by the current kernel
 *
 * @details Calculates the distance of each element X in in a group to
 * each of its other elements
 */
__kernel void fillDistanceMatrix(__global const T* xGroup,
                                 __global T* distanceMatrix,
                                 const int groupSize)
{
    int i = get_global_id(0); // Group to be considered
    int j = get_global_id(1); // Position in considered group

    // Check for valid case
    if (i < groupSize && j < groupSize) {
        // Perform distance calculation
        distanceMatrix[i * groupSize + j] = fabs(xGroup[i] - xGroup[j]);
        
    }
}


/**!
 * @brief Parallelization of the kth_element function using Lomuto's partitioning with QuickSelect.
 *
 * @param distanceMatrix Created from the CL distance matrix operation
 * @param result Array that stores the results of kth_element operation
 * @param groupSize Integer size of the groups which will be iterated through
 * @param K the Kth smallest value which is searched for by QuickSelect
 */
__kernel void kth_element(__global T* distanceMatrix,
                          __global T* result,
                          const int groupSize,
                          const int K)
{

    // Declare local variables
    __local int lowerBound; // The lowerBound of the partitioning algorithm
    __local int upperBound; // The upperBound of the partitioning algorithm
    __local int left_swap; // Element to swap on the left
    __local int right_swap; // Element to swap on the right
    __local int pivotIndex; // Current pivot index
    __local T pivot; // Current pivot value
    __local T kth_value; // Calculated kth smallest value

    // Get current thread ID / #
    const int i = get_global_id(0);

    // Clamp K to groupSize, Adjust to correct position in distance array
    const int kClamped = (K > groupSize - 1) ? (groupSize - 1) : K;
    const int kAdjusted = kClamped + i * groupSize;

    // ------------------------ START QUICKSELECT ------------------------ //
    // Credit - Based off of: https://www.geeksforgeeks.org/quickselect-algorithm/

    // Gets the first and last indexes of the current grou
    lowerBound = groupSize * i;
    upperBound = groupSize * i + groupSize - 1;

    // Iterate until the lowerBound and upperBound are the same
    while ( lowerBound <= upperBound ) {
        
        // Partition Context
        pivot = distanceMatrix[upperBound];
        left_swap = lowerBound - 1;

        // --------------- START PARTITION (Lomuto) --------------- //
        // Credit - Based off of: https://www.geeksforgeeks.org/lomuto-partition-algorithm/
        
        // Set the element
        right_swap = lowerBound;
        while (right_swap < upperBound) {
        
            if (distanceMatrix[right_swap] < pivot) {
                left_swap++; // Move to the next swap position
                T temp = distanceMatrix[left_swap]; // Swap left and right positions 
                distanceMatrix[left_swap] = distanceMatrix[right_swap];
                distanceMatrix[right_swap] = temp;
            }
            // Move to the next swap position
            right_swap++;
        }

        // Swap pivot element & last unswapped element
        T temp = distanceMatrix[left_swap+1];
        distanceMatrix[left_swap+1] = distanceMatrix[upperBound];
        distanceMatrix[upperBound] = temp;

        // ---------------- END PARTITION (Lomuto) ---------------- //

        // Get the current pivot index
        pivotIndex = left_swap + 1;

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
    kth_value = distanceMatrix[kAdjusted];
    // Calculate the result to be read back to the program
    result[i] = kClamped / (groupSize * 2.0f * kth_value);

}