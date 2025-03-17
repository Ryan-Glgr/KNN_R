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
