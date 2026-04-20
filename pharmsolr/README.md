# pharmsolr

A tiny example R package wrapping `pharmsol`'s JIT-compiled model API via
[extendr](https://extendr.github.io/). Define a pharmacometric model in plain
text, JIT-compile it through Cranelift, and run simulations from R — no
recompilation per model.

## Requirements

- R ≥ 4.1
- A working Rust toolchain (`rustup`, `cargo`) at install time only. Once the
  package is installed, end users do **not** need Rust to use it: they just
  load the package and call `compile_model()` on any text definition.

## Install

From the repository root:

```r
# devtools / remotes
install.packages("rextendr")
rextendr::document("pharmsolr")  # regenerates wrappers if you change Rust code
install.packages("pharmsolr", repos = NULL, type = "source")
```

Or directly:

```sh
R CMD INSTALL pharmsolr
```

## Usage

```r
library(pharmsolr)

mod <- compile_model("
  name         = onecmt-allo
  compartments = central
  params       = CL, V
  covariates   = WT
  dxdt(central) = rateiv[0] - (CL * pow(WT / 70.0, 0.75) / V) * central
  out(cp)       = central / V
")

events <- data.frame(
  time  = c(0,         1,    2,    4,    8,    12),
  evid  = c(1L,        0L,   0L,   0L,   0L,   0L),    # 1=dose, 0=observation
  amt   = c(100,       0,    0,    0,    0,    0),
  dur   = c(0.5,       0,    0,    0,    0,    0),    # dur > 0 -> infusion; dur = 0 -> bolus
  cmt   = c("central", NA,   NA,   NA,   NA,   NA),   # name or 0-based integer
  outeq = c(NA,        "cp", "cp", "cp", "cp", "cp"), # name or 0-based integer
  stringsAsFactors = FALSE
)

simulate_subject(mod,
  params     = c(CL = 5, V = 50),    # named -> reordered to model's params()
  events     = events,
  covariates = list(WT = data.frame(time = 0, value = 80))
)
#>   time     pred
#> 1  1.0   1.8432
#> 2  2.0   1.6503
#> 3  4.0   1.3230
#> 4  8.0   0.8502
#> 5 12.0   0.5464
```

The model owns the canonical name-to-index map. Use `compartments(mod)`,
`outputs(mod)`, `params(mod)`, and `covariates(mod)` to inspect it, or
`cmt(mod, "central")` / `outeq(mod, "cp")` for ad-hoc lookups. Events may
reference compartments and outputs by name (recommended) or by zero-based
integer (matches the order declared in the model).

## Model text format

See `?pharmsolr::compile_model` and the `pharmsol::jit::text` Rust module for
the full grammar. Briefly:

```text
name         = my_model
compartments = depot, central
params       = ka, CL, V
covariates   = WT          # optional
ndrugs       = 1            # optional, default 1

dxdt(depot)   = -ka * depot
dxdt(central) =  ka * depot - (CL * pow(WT/70.0, 0.75) / V) * central
out(cp)       =  central / V
```

Expression syntax: `+ - * / ^`, plus `exp`, `ln`, `log`, `log10`, `sqrt`,
`abs`, `pow(a, b)`. Identifiers may be parameter names, compartment names,
covariate names, or `t`. `rateiv[i]` is the current infusion rate on
input channel `i`.

## What this demonstrates

End-to-end runtime model authoring:

```
   text in R           extendr             pharmsol::jit          Cranelift
+-------------+   →  +---------+   →   +----------------+   →   +--------+
| model_text  |     | bridge  |       | parse + lower  |       | native |
| events df   |     |  layer  |       | compile        |       | code   |
+-------------+     +---------+       +----------------+       +--------+
                                                                    │
                                                                    ▼
                                                           predictions in R
```

The R user never touches Rust source files, never invokes `cargo build`, and
never restarts R between models.
