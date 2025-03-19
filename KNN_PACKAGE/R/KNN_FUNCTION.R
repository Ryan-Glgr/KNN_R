#’ @title K‑Nearest Neighbors wrapper
#’ @description Provides an R‑callable interface to our C++ KNN implementation (Library.cpp).
#’ @author Ryan Gallagher
#’ @date 2025‑03‑18
#’ @useDynLib YourPackageName
#’ @importFrom Rcpp sourceCpp
NULL

#' Compute Information Energy using KNN
#'
#' This function calls the exported C++ function `KNN` directly.
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
  # Directly call the C++ function (assumes it is registered with R via Rcpp)
  result <- KNN(data_x, data_y, as.integer(K), mode, precision)
  return(result)
}