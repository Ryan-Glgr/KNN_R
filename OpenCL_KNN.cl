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
                          __global float* result,
                                 const int groupSize,
                                 const int K)
{

    // Since distanceMatrix is 1D memory in OpenCL, i and groupSize determines the row starting position

    // Get the current thread number (iteration in the "loop")
    int i = get_global_id(0);

    // Clamp k to the max group size
    int kClamped = (K > groupSize - 1) ? (groupSize - 1) : K;
    // Adjusts k to the correct position in the array
    int kAdjusted = kClamped + (i * groupSize);

    // ------------------------ START QUICKSELECT ------------------------ //

    int lowerBound = i * groupSize;
    int upperBound = i * groupSize + groupSize - 1;

    while (upperBound > lowerBound) {

        // -------------------- START PARTITION -------------------- //
        // Creates starting points
        int left = lowerBound - 1;
        int right = upperBound + 1;
    
        // Sets the pivot to the first element in the array
  	    int pivot = distanceMatrix[lowerBound];

        // Loops to partition
  	    while (true) {
      
      	    // Find element larger than pivot
      	    do {
                left++;
            } while (distanceMatrix[left] < pivot && left < upperBound);

      	    // Find element smaller than pivot
            do {
                right--;
            } while (distanceMatrix[right] > pivot && right > lowerBound);
      	
      	    // If left and right cross, break;
            if (left >= right) {break;}
      	
      	    // Swap if left and right don't cross
            if (left < right) {
                float temp = distanceMatrix[left];
                distanceMatrix[left] = distanceMatrix[right];
                distanceMatrix[right] = temp;
            }
        }
        // -------------------- END PARTITION -------------------- //

        // Adjusts the bounds of the algorithm given
        // the position relative to the pivot
        if (kAdjusted == right) {
            break;
        } else if (kAdjusted < right) {
            upperBound = right - 1;
        } else {
            lowerBound = right + 1;
        }

    }
    // ------------------------- END QUICKSELECT ------------------------- //


    // After QuickSelect, store the found element in Ri
    // Calculate row element
    float Ri = distanceMatrix[kAdjusted];
    // Calculate the result to be read back to the program
    result[i] = kClamped / (groupSize * 2.0f * Ri);

}