#' Compile a pharmsol JIT model from a text definition
#'
#' @param text Character scalar containing a pharmsol JIT model definition
#'   (see the package README for the format).
#' @return An external pointer wrapping the compiled model. Pass to
#'   [simulate_subject()].
#' @export
compile_model <- function(text) {
  stopifnot(is.character(text), length(text) == 1L)
  .Call("wrap__compile_model", text, PACKAGE = "pharmsolr")
}

#' Simulate a single subject through a compiled JIT model
#'
#' @param model A model handle returned by [compile_model()].
#' @param params Numeric vector of parameter values, in the order declared by
#'   the `params = ...` line of the model.
#' @param events A data.frame with one row per event and columns:
#'   \describe{
#'     \item{time}{numeric — event time}
#'     \item{evid}{integer — 0 = observation, 1 = dose (bolus when `dur` is 0, infusion when `dur` > 0)}
#'     \item{amt}{numeric — dose amount (ignored for observations)}
#'     \item{dur}{numeric — infusion duration; use 0 for a bolus}
#'     \item{cmt}{integer — input compartment for bolus/infusion (0-based)}
#'     \item{outeq}{integer — output equation index for observation (0-based)}
#'   }
#' @param covariates Optional named list. Each entry is a data.frame with
#'   columns `time` and `value`, naming the covariate (must match the model's
#'   `covariates = ...` line).
#' @return A data.frame with columns `time` and `pred`.
#' @export
simulate_subject <- function(model, params, events, covariates = list()) {
  stopifnot(is.data.frame(events))
  required <- c("time", "evid", "amt", "dur", "cmt", "outeq")
  missing  <- setdiff(required, names(events))
  if (length(missing)) {
    stop("`events` is missing required columns: ", paste(missing, collapse = ", "))
  }
  if (!is.list(covariates)) covariates <- list()

  cov_names  <- names(covariates) %||% character(0)
  cov_times  <- lapply(covariates, function(df) as.numeric(df$time))
  cov_values <- lapply(covariates, function(df) as.numeric(df$value))

  out <- .Call(
    "wrap__simulate_subject",
    model,
    as.numeric(params),
    as.numeric(events$time),
    as.integer(events$evid),
    as.numeric(events$amt),
    as.numeric(events$dur),
    as.integer(events$cmt),
    as.integer(events$outeq),
    as.character(cov_names),
    cov_times,
    cov_values,
    PACKAGE = "pharmsolr"
  )
  data.frame(time = out$time, pred = out$pred)
}

`%||%` <- function(a, b) if (is.null(a)) b else a
