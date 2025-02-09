__kernel void computeDistance(__global float *xAttributes, __global float *distances, __global float *results, __global int *numXsPerY, int numXs, int numYs, int K, __global float *finalResult,__global float *mergeTemp) {

    int blockNum = get_group_id(0);
    int localID = get_local_id(0); // openCL version of threadIdx.x
    int blockSize = get_local_size(0); // openCL version of blockDim.x

    // each work group is going to work on different y's in parallel. a work group is a cuda block.
    // so we stride by the amount of work groups.
    for (int i = blockNum; i < numYs; i += get_global_size(0)) {

        // compute how many x's are the offset into the array for this y.
        int numXsOffset = 0;
        for (int j = 0; j < i; j++) {
            numXsOffset += numXsPerY[j];
        }
        // pointer math to get our x's that correspond to the y we are calculating now.
        __global float *thisY_xs = xAttributes + numXsOffset;

        // have a work group iterate through each particular x for the y they are working on.
        // after one iteration, we sort distances, and do the result calculation.
        for (int startingX = 0; startingX < numXsPerY[i]; startingX++) {

            // Compute the absolute distances for this startingX in parallel.
            for (int particularX = localID; particularX < numXsPerY[i]; particularX += blockSize) {
                distances[particularX + numXsOffset] = fabs(thisY_xs[particularX] - thisY_xs[startingX]);
            }

            // sync the work group. (__syncthreads() in real programming languages)
            barrier(CLK_LOCAL_MEM_FENCE);

            // Instead of using a fixed-size local temporary array, we use a second global buffer.
            // We define two pointers: 'src' initially points to the unsorted distances
            // and 'dst' points into the mergeTemp buffer.
            __global float *src = distances + numXsOffset;
            __global float *dst = mergeTemp + numXsOffset;

            // Iterative bottom-up merge sort:
            // 'width' is the size of each subarray to merge.
            for (int width = 1; width < numXsPerY[i]; width *= 2) {
                // Distribute merge segments among threads.
                for (int start = localID * (2 * width); start < numXsPerY[i]; start += blockSize * (2 * width)) {
                    int mid = min(start + width, numXsPerY[i]);    // End of left subarray.
                    int end = min(start + 2 * width, numXsPerY[i]);  // End of right subarray.
                    
                    // Merge the two sorted subarrays from 'src' into 'dst'.
                    int leftIdx = start;   // Pointer into left subarray of 'src'.
                    int rightIdx = mid;    // Pointer into right subarray of 'src'.
                    int k = start;         // Insertion index in 'dst'.
                    
                    while (leftIdx < mid && rightIdx < end) {
                        if (src[leftIdx] <= src[rightIdx])
                            dst[k++] = src[leftIdx++];
                        else
                            dst[k++] = src[rightIdx++];
                    }
                    while (leftIdx < mid)
                        dst[k++] = src[leftIdx++];
                    while (rightIdx < end)
                        dst[k++] = src[rightIdx++];
                }
                barrier(CLK_LOCAL_MEM_FENCE); // Synchronize all threads after this merge pass.
                
                // Swap the pointers: now, 'src' points to the merged data.
                __global float *tempPtr = src;
                src = dst;
                dst = tempPtr;
                barrier(CLK_LOCAL_MEM_FENCE); // Synchronize after swapping.
            } // END merge sort passes

            // After merging, if the final sorted data is not in the original distances buffer,
            // copy it back. (This happens when an odd number of passes occurs.)
            if (src != (distances + numXsOffset)) {
                for (int p = localID; p < numXsPerY[i]; p += blockSize) {
                    distances[p + numXsOffset] = src[p];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            // ---------- END MERGE SORT USING GLOBAL DOUBLE BUFFER ----------

            // ----------------------------RESULT CALCULATION---------------------------
            // If K is out of bounds, use the last element in the sorted distances.
            int indexK = (K < numXsPerY[i]) ? K : numXsPerY[i] - 1;
            if (localID == 0) {
                results[startingX + numXsOffset] = ((float)K) / (2.0f * numXsPerY[i] * distances[numXsOffset + indexK]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        } // END OF ONE X LOOP.
    } // END OF ONE Y LOOP.

    // ----------------------------FINAL AGGREGATION---------------------------
    // each work-group aggregates the results for the y's it processed and writes the partial sum.
    if (localID == 0){
        // Loop over all y's that this work-group processed.
        // Each work-group processes y indices: i = blockNum, blockNum + get_global_size(0), etc.
        for (int i = blockNum; i < numYs; i += get_global_size(0)) {
            int numXsOffset = 0;
            for (int j = 0; j < i; j++) {
                numXsOffset += numXsPerY[j];
            }
            float sum = 0.0f;  // Reset the sum for each y.
            for (int j = 0; j < numXsPerY[i]; j++) {
                sum += results[j + numXsOffset];
            }
            float mean = sum / numXsPerY[i];
            float weight = ((float)numXsPerY[i]) / ((float)numXs);
            float addValue = mean * weight;
            results[numXsOffset] = addValue; // Store the per-y sum in the first element.
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE); // once we have added up all the results
    if (get_global_id(0) == 0) {
        int numXsOffset = 0;
        float sum = 0;
        for (int i = 0; i < numYs; i++) {
            sum += results[numXsOffset];
            numXsOffset += numXsPerY[i];
        }
        *finalResult = sum;
    }
}