# Benchmark: R-side calls vs native Rust.
#
# Run from the pharmsol repo root:
#   Rscript pharmsolr/inst/examples/bench.R

suppressMessages(library(pharmsolr))

text <- "
  model = onecmt_allo
  kind = ode

  params = CL, V
  covariates = WT@locf
  states = central
  outputs = cp

  infusion(iv_central) -> central

  dx(central) = -(CL * pow(WT / 70.0, 0.75) / V) * central
  out(cp) = central / V ~ continuous()
"

# Compile time
t0 <- Sys.time()
mod <- compile_model(text)
compile_s <- as.numeric(Sys.time() - t0, units = "secs")
cat(sprintf("compile: %.3f ms\n", compile_s * 1000))

events <- data.frame(
  time  = c(0, 1, 2, 4, 8, 12),
  evid  = c(1L, 0L, 0L, 0L, 0L, 0L),
  amt   = c(100, 0, 0, 0, 0, 0),
  dur   = c(0.5, 0, 0, 0, 0, 0),
  cmt   = c(0L, 0L, 0L, 0L, 0L, 0L),
  outeq = c(0L, 0L, 0L, 0L, 0L, 0L)
)
covs <- list(WT = data.frame(time = 0, value = 80))
params <- c(5, 50)

# Warm-up
invisible(replicate(50, simulate_subject(mod, params, events, covs)))

n <- as.integer(Sys.getenv("BENCH_N", "10000"))

t0 <- Sys.time()
for (i in seq_len(n)) {
  res <- simulate_subject(mod, params, events, covs)
}
elapsed <- as.numeric(Sys.time() - t0, units = "secs")
per_us <- elapsed * 1e6 / n
cat(sprintf(
  "simulate_subject: n=%d  total=%.3f s  per_call=%.3f us  (%.0f calls/s)\n",
  n, elapsed, per_us, n / elapsed
))
