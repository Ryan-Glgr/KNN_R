#' Compute Information Energy using KNN
#'
#' This function loads the shared library and calls the exported C++ function `KNN`
#' directly.
#'
#' @param data_x Numeric vector representing the first data set.
#' @param data_y Numeric vector representing the second data set.
#' @param K Integer specifying the number of neighbors.
#' @param mode Character string specifying the mode ("cpp", "cuda", or "openCL").
#'             Defaults to "cpp".
#' @param precision Character string specifying the precision ("float" or "double").
#'                  Defaults to "float".
#' @return The result returned by the native C++ function.
#' @export
Compute_IE <- function(data_x, data_y, K, mode = "cpp", precision = "float") {

    # Build the DLL/dylib/so path in a platform-independent fashion.
    dll_path <- system.file("libs", paste0("CWUKNN", .Platform$dynlib.ext), package = "CWUKNN")
    # Load the shared library.
    dyn.load(dll_path)

    # Directly call the C++ function (assumes it is registered with R via Rcpp or similar).
    result <- KNN(data_x, data_y, as.integer(K), mode, precision)

    # Unload the shared library.
    dyn.unload(dll_path)

    return(result)
}
