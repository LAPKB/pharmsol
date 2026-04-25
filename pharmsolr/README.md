# pharmsolr

A tiny example R package wrapping `pharmsol`'s DSL runtime API via
[extendr](https://extendr.github.io/). Define a pharmacometric model in the
recommended sugared DSL, compile it through the runtime JIT, and run
simulations from R.

## Requirements

- R ≥ 4.1
- A working Rust toolchain (`rustup`, `cargo`) at install time only. Once the
  package is installed, end users do **not** need Rust to use it: they just
  load the package and call `compile_model()` on any DSL definition.

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
  model = onecmt_allo
  kind = ode

  params = CL, V
  covariates = WT@locf
  states = central
  outputs = cp

  infusion(iv_central) -> central

  dx(central) = -(CL * pow(WT / 70.0, 0.75) / V) * central
  out(cp) = central / V ~ continuous()
")

events <- data.frame(
  time  = c(0,         1,    2,    4,    8,    12),
  evid  = c(1L,        0L,   0L,   0L,   0L,   0L),    # 1=dose, 0=observation
  amt   = c(100,       0,    0,    0,    0,    0),
  dur   = c(0.5,       0,    0,    0,    0,    0),    # dur > 0 -> infusion; dur = 0 -> bolus
  cmt   = c("iv_central", NA,   NA,   NA,   NA,   NA),   # route name or 0-based integer
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

The model owns the canonical name-to-index map. Use `routes(mod)`,
`outputs(mod)`, `params(mod)`, and `covariates(mod)` to inspect it, or
`route(mod, "central")` / `outeq(mod, "cp")` for ad-hoc lookups. Events may
reference routes and outputs by name (recommended) or by zero-based integer.

## Model source format

See `?pharmsolr::compile_model` for the full grammar. Briefly:

```text
model = my_model
kind = ode

params = ka, CL, V
covariates = WT@locf
states = depot, central
outputs = cp

bolus(oral) -> depot
infusion(iv_central) -> central

dx(depot) = -ka * depot
dx(central) = ka * depot - (CL * pow(WT/70.0, 0.75) / V) * central
out(cp) = central / V ~ continuous()
```

Expression syntax: `+ - * / ^`, plus `exp`, `ln`, `log`, `log10`, `sqrt`,
`abs`, `pow(a, b)`. Identifiers may be parameter names, state names,
covariate names, or `t`. Doses enter through named routes declared with
`bolus(...) -> state` or `infusion(...) -> state`.

## What this demonstrates

End-to-end runtime model authoring:

```
  DSL in R            extendr           pharmsol::dsl          Cranelift
+-------------+   →  +---------+   →   +----------------+   →   +--------+
| model_text  |     | bridge  |       | parse + lower  |       | native |
| events df   |     |  layer  |       | runtime JIT    |       | code   |
+-------------+     +---------+       +----------------+       +--------+
                                                                    │
                                                                    ▼
                                                           predictions in R
```

The R user never touches Rust source files, never authors legacy text/JIT
models, and never needs more than the DSL plus the runtime API.
