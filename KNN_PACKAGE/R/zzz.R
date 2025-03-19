#’ @title Package Load/Unload Hooks
#’ @description Automatically load the compiled CWUKNN dynamic library when the package is loaded,
#’   and unload it when the package is unloaded.
#’ @author Ryan Gallagher
#’ @date 2025‑03‑18
#’ @keywords internal
NULL

.onLoad <- function(libname, pkgname) {
  dll_path <- system.file("libs", paste0("CWUKNN", .Platform$dynlib.ext), package = pkgname)
  if (dll_path != "") {
    dyn.load(dll_path)
  }
}

.onUnload <- function(libpath) {
  dll_path <- system.file("libs", paste0("CWUKNN", .Platform$dynlib.ext), package = utils::packageName())
  if (dll_path != "") {
    dyn.unload(dll_path)
  }
}
