// WE WRITE THE KERNEL CODE HERE FOR SYNTAX HIGHLIGHTING AND EASE OF USE
// WHEN COMPLETE, PASTE THE KERNEL CODE INTO THE CPP FILE IN THE PARANTHESES.

__kernel void computeDistance(__global float *xAttributes, __global float *yAttributes, __global float *distances, __global float *results, __global int *numXsPerY,__global float *finalResults, int N){

    int globalID = get_global_id(0); // OpenCL version of cuda computation of blockId.x * blockDim.x + threadId.x
    int localID = get_local_id(0); // openCL version of threadIdx.x






}
