# Example: text -> JIT model -> simulation, end-to-end in R.
#
# Demonstrates the name-based API: events refer to compartments and outputs
# by name, parameters are passed by name, and the model owns the canonical
# name-to-index map.
#
# Run from the package root after installing with e.g.
#   R CMD INSTALL pharmsolr
# or, during development:
#   devtools::load_all("pharmsolr")

library(pharmsolr)

model_text <- "
  # Two-compartment IV with allometric scaling on CL.
  # Two outputs: central concentration and peripheral concentration.
  name         = twocmt-allo
  compartments = central, periph
  params       = CL, V, Q, Vp
  covariates   = WT
  ndrugs       = 1

  dxdt(central) = rateiv[0] - (CL * pow(WT / 70.0, 0.75) / V) * central
                            - (Q / V) * central + (Q / Vp) * periph
  dxdt(periph)  =             (Q / V) * central - (Q / Vp) * periph

  out(cp)        = central / V
  out(cp_periph) = periph  / Vp
"

mod <- compile_model(model_text)

# Inspect the canonical name-to-index map. These are what the data layer uses.
cat("compartments:", paste(compartments(mod), collapse = ", "), "\n")
cat("outputs:     ", paste(outputs(mod),      collapse = ", "), "\n")
cat("params:      ", paste(params(mod),       collapse = ", "), "\n")
cat("covariates:  ", paste(covariates(mod),   collapse = ", "), "\n\n")

# Build events using *names*, not magic integers. Reordering compartments or
# outputs in the model definition above will not break this code.
events <- data.frame(
  time  = c(0.0,        1.0,    2.0,    4.0,         8.0,    12.0),
  evid  = c(1L,         0L,     0L,     0L,          0L,     0L),
  amt   = c(100,        0,      0,      0,           0,      0),
  dur   = c(0.5,        0,      0,      0,           0,      0),
  cmt   = c("central",  NA,     NA,     NA,          NA,     NA),
  outeq = c(NA,         "cp",   "cp",   "cp_periph", "cp",   "cp"),
  stringsAsFactors = FALSE
)

covs <- list(
  WT = data.frame(time = 0.0, value = 80.0)
)

# Pass parameters by name -- order is irrelevant on the call site; the
# wrapper reorders to match the model's declared param order.
out <- simulate_subject(
  mod,
  params     = c(V = 50.0, CL = 5.0, Vp = 80.0, Q = 4.0),
  events     = events,
  covariates = covs
)

print(out)

# Lookup helpers, equivalent to mrgsolve::cmtn().
stopifnot(cmt(mod, "central")    == 0L)
stopifnot(cmt(mod, "periph")     == 1L)
stopifnot(outeq(mod, "cp")       == 0L)
stopifnot(outeq(mod, "cp_periph") == 1L)
