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

#' Compartment, output, parameter, and covariate names of a compiled model
#'
#' These accessors return the canonical name-to-index map fixed at compile
#' time. Use them to avoid hard-coding integers for `cmt` and `outeq`.
#'
#' `compartments()` index is the value to pass as `cmt` for a dose targeting
#' that compartment. `outputs()` index is the value to pass as `outeq` for an
#' observation of that output. `params()` is the order of the `params`
#' argument to [simulate_subject()].
#'
#' @param model A model handle from [compile_model()].
#' @return Character vector in declaration order.
#' @name model_metadata
NULL

#' @rdname model_metadata
#' @export
compartments <- function(model) {
  .Call("wrap__model_compartments", model, PACKAGE = "pharmsolr")
}

#' @rdname model_metadata
#' @export
outputs <- function(model) {
  .Call("wrap__model_outputs", model, PACKAGE = "pharmsolr")
}

#' @rdname model_metadata
#' @export
params <- function(model) {
  .Call("wrap__model_params", model, PACKAGE = "pharmsolr")
}

#' @rdname model_metadata
#' @export
covariates <- function(model) {
  .Call("wrap__model_covariates", model, PACKAGE = "pharmsolr")
}

#' Look up a compartment's `cmt` index by name (zero-based)
#'
#' @param model A model handle from [compile_model()].
#' @param name Character scalar. Stops with an informative error if not found.
#' @return Integer scalar (zero-based).
#' @export
cmt <- function(model, name) {
  cmts <- compartments(model)
  i <- match(name, cmts)
  if (is.na(i)) {
    stop(sprintf("unknown compartment %s; declared: [%s]",
                 dQuote(name), paste(cmts, collapse = ", ")))
  }
  as.integer(i - 1L)
}

#' Look up an output's `outeq` index by name (zero-based)
#'
#' @param model A model handle from [compile_model()].
#' @param name Character scalar. Stops with an informative error if not found.
#' @return Integer scalar (zero-based).
#' @export
outeq <- function(model, name) {
  outs <- outputs(model)
  i <- match(name, outs)
  if (is.na(i)) {
    stop(sprintf("unknown output %s; declared: [%s]",
                 dQuote(name), paste(outs, collapse = ", ")))
  }
  as.integer(i - 1L)
}

#' Simulate a single subject through a compiled JIT model
#'
#' @param model A model handle returned by [compile_model()].
#' @param params Numeric vector of parameter values. May be unnamed (uses the
#'   declared order from the model) or named (will be reordered to match
#'   [params()]).
#' @param events A data.frame with one row per event and columns:
#'   \describe{
#'     \item{time}{numeric -- event time}
#'     \item{evid}{integer -- 0 = observation, 1 = dose (bolus when `dur` is 0, infusion when `dur` > 0)}
#'     \item{amt}{numeric -- dose amount (ignored for observations)}
#'     \item{dur}{numeric -- infusion duration; use 0 for a bolus}
#'     \item{cmt}{integer or character -- input compartment for bolus/infusion.
#'       If character, names are resolved against the compiled model's
#'       compartments (see [compartments()]). Integer values are zero-based.}
#'     \item{outeq}{integer or character -- output equation for an observation.
#'       If character, names are resolved against the compiled model's outputs
#'       (see [outputs()]). Integer values are zero-based.}
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

  # Resolve character cmt / outeq columns to zero-based integers using the
  # compiled model's name-to-index map. Integer columns pass through.
  cmt_int   <- resolve_indices(events$cmt,   compartments(model), "cmt")
  outeq_int <- resolve_indices(events$outeq, outputs(model),      "outeq")

  # Reorder named params to match the model's declared parameter order.
  param_vec <- as.numeric(coerce_params(model, params))

  cov_names  <- names(covariates) %||% character(0)
  cov_times  <- lapply(covariates, function(df) as.numeric(df$time))
  cov_values <- lapply(covariates, function(df) as.numeric(df$value))

  out <- .Call(
    "wrap__simulate_subject",
    model,
    param_vec,
    as.numeric(events$time),
    as.integer(events$evid),
    as.numeric(events$amt),
    as.numeric(events$dur),
    cmt_int,
    outeq_int,
    as.character(cov_names),
    cov_times,
    cov_values,
    PACKAGE = "pharmsolr"
  )
  data.frame(time = out$time, pred = out$pred)
}

# Resolve a column that may be character (names) or numeric (zero-based
# indices) into an integer vector of zero-based indices. Stops with a clear
# error listing the available names if a name does not match.
resolve_indices <- function(col, names_vec, label) {
  if (is.character(col)) {
    idx <- match(col, names_vec)
    bad <- !is.na(col) & is.na(idx)
    if (any(bad)) {
      stop(sprintf(
        "unknown %s name(s): [%s]; declared: [%s]",
        label,
        paste(unique(col[bad]), collapse = ", "),
        paste(names_vec, collapse = ", ")
      ))
    }
    # NA cells (e.g. cmt for an observation row) collapse to 0; the Rust side
    # only consults cmt for doses and outeq for observations.
    idx[is.na(col)] <- 1L
    as.integer(idx - 1L)
  } else {
    as.integer(col)
  }
}

coerce_params <- function(model, p) {
  declared <- params(model)
  if (is.null(names(p))) {
    if (length(p) != length(declared)) {
      stop(sprintf(
        "params has length %d but model declares %d (%s)",
        length(p), length(declared), paste(declared, collapse = ", ")
      ))
    }
    return(p)
  }
  missing_p <- setdiff(declared, names(p))
  if (length(missing_p)) {
    stop("params is missing values for: ", paste(missing_p, collapse = ", "))
  }
  unname(p[declared])
}

`%||%` <- function(a, b) if (is.null(a)) b else a
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
