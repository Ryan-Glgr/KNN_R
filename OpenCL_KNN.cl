// WE WRITE THE KERNEL CODE HERE FOR SYNTAX HIGHLIGHTING AND EASE OF USE
// WHEN COMPLETE, PASTE THE KERNEL CODE INTO THE CPP FILE IN THE PARANTHESES.

__kernel void computeDistance(__global float *xAttributes, __global float *distances, __global float *results, __global int *numXsPerY, int numXs, int numYs, int K, __global float *finalResult){

    int blockNum = get_group_id(0);
    int localID = get_local_id(0); // openCL version of threadIdx.x
    int blockSize = get_local_size(0);

    // each work group is going to work on different y's in parallel. so we stride by the amount of work groups.
    for(int i = blockNum; i < numYs; i += get_global_size(0)) {

        // compute how many x's are the offset into the array for this y.
        int numXsOffset = 0;
        for(int j = 0; j < i; j++){
            numXsOffset += numXsPerY[j];
        }
        // pointer math to get our x's that correspond to the y we are calculating now.
        __global float *thisY_xs = xAttributes + numXsOffset;

        // have a work group iterate through each particular x for the y they are working on. after one iteration, we sort distances, and do the result calculation.
        for(int startingX = 0; startingX < numXsPerY[i]; startingX++){
            
            for(int particularX = localID; particularX < numXsPerY[i]; particularX += blockSize){
                distances[particularX + numXsOffset] = fabs(thisY_xs[particularX] - thisY_xs[startingX]);
            }

            // sync the work group. (__syncthreads() in real programming languages)
            barrier(CLK_LOCAL_MEM_FENCE);

            
            // Allocate a temporary local buffer.
            __local float temp[256];  // Adjust this size as needed.
    
            // Iterative bottom-up merge sort:
            // 'width' is the size of each subarray to merge.

            // ----------------------------MERGE SORT---------------------------
            for (int width = 1; width < numXsPerY[i]; width *= 2) {
                // Process each merge segment of size (2 * width).
                // Each segment starts at index 'start'. We distribute merge segments among threads.
                for (int start = localID * (2 * width); start < numXsPerY[i]; start += blockSize * (2 * width)) {
                    int mid = min(start + width, numXsPerY[i]);         // End of left subarray.
                    int end = min(start + 2 * width, numXsPerY[i]);         // End of right subarray.
                    
                    // Merge the two sorted subarrays [start, mid) and [mid, end) into temp.
                    int i = start;   // Pointer into left subarray.
                    int j = mid;     // Pointer into right subarray.
                    int k = start;   // Insertion index in temp.
                    while (i < mid && j < end) {
                        if (distances[i + numXsOffset] <= distances[j + numXsOffset])
                            temp[k++] = distances[i++ + numXsOffset];
                        else
                            temp[k++] = distances[j++ + numXsOffset];
                    }
                    // Copy any remaining elements.
                    while (i < mid)
                        temp[k++] = distances[i++ + numXsOffset];
                    while (j < end)
                        temp[k++] = distances[j++ + numXsOffset];
                    
                    // Copy the merged segment from temp back into the global array.
                    // Each thread copies a portion of the merged segment.
                    for (int p = start + localID; p < end; p += blockSize) {
                        distances[p + numXsOffset] = temp[p];
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);  // Synchronize threads within the work-group.
                }
                barrier(CLK_LOCAL_MEM_FENCE);      // Ensure all merge segments are finished before next pass.
            } // END MERGE SORT -------------------------------------------------

            // ----------------------------RESULT CALCULATION---------------------------
            if (localID == 0){
                results[startingX + numXsOffset] = K / (2 * numXsPerY[i] * distances[numXsOffset + K]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        } // END OF ONE X LOOP.

        // NOW WE AGGREGATE THE RESULTS OF THIS Y, AND ADD IT TO THE GLOBAL RESULT.
        float mean;
        if (localID == 0){
            float sum = 0;
            for(int j = 0; j < numXsPerY[i]; j++){
                sum += results[j + numXsOffset];
            }
            mean = sum / numXsPerY[i];
        }

        // NOW WE ADD THE WEIGHTED RESULT OF THIS Y TO THE GLOBAL RESULT.
        if (localID == 0){
            float weight = (float)numXsPerY[i] / (float)numXs;
            float addValue = mean * weight;
            int oldVal, newVal;
            do {
                oldVal = atomic_cmpxchg((__global int*)finalResult, 
                                       oldVal, 
                                       as_int(as_float(oldVal) + addValue));
            } while (oldVal != newVal);
        }



    }
}
