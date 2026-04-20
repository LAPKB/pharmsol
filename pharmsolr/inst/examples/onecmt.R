# Example: text → JIT model → simulation, end-to-end in R.
#
# Run from the package root after installing with e.g.
#   R CMD INSTALL pharmsolr
# or, during development:
#   devtools::load_all("pharmsolr")

library(pharmsolr)

model_text <- "
  # One-compartment IV with allometric scaling on CL
  name         = onecmt-allo
  compartments = central
  params       = CL, V
  covariates   = WT
  ndrugs       = 1

  dxdt(central) = rateiv[0] - (CL * pow(WT / 70.0, 0.75) / V) * central
  out(cp)       = central / V
"

mod <- compile_model(model_text)

# 100 mg infused over 0.5 h, observe at five time points.
events <- data.frame(
  time  = c(0.0, 1.0, 2.0, 4.0, 8.0, 12.0),
  evid  = c(2L,  0L,  0L,  0L,  0L,  0L),
  amt   = c(100, 0,   0,   0,   0,   0),
  dur   = c(0.5, 0,   0,   0,   0,   0),
  cmt   = c(0L,  0L,  0L,  0L,  0L,  0L),
  outeq = c(0L,  0L,  0L,  0L,  0L,  0L)
)

covariates <- list(
  WT = data.frame(time = 0.0, value = 80.0)
)

out <- simulate_subject(
  mod,
  params     = c(CL = 5.0, V = 50.0),
  events     = events,
  covariates = covariates
)
print(out)
