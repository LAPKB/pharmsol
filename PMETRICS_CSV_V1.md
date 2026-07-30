# `pmetrics-csv.v1`

`pmetrics-csv.v1` is pharmsol's canonical, lossless CSV encoding for execution-relevant `Data` state. It is intended for hashing, transport, storage, and exact re-encoding; ordinary Pmetrics CSV remains readable through `read_pmetrics`.

## Byte contract

- UTF-8 without a BOM, comma-delimited, LF line endings, and a final LF. Comment records are not part of the format.
- The first 15 columns are exactly:
  `ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,CENS,C0,C1,C2,C3`.
- Covariate columns follow in lexical order. A trailing `!` marks carry-forward/fixed interpolation. Names that collide case-insensitively with core columns, end in `!`, or collide case-insensitively with another covariate are rejected.
- The writer emits `.` for missing values; the reader accepts `.` and `NA`.
- Finite floating-point values use Rust's shortest round-trip-safe decimal representation. Nonfinite values are rejected.
- Subject rows are ordered by subject ID. Each stored occasion starts with a boundary-only `EVID=4` row, including the first occasion.
- Within an occasion, rows are ordered by time, then observation, bolus, infusion, and covariate-only row. Existing order is retained for same-time events of the same kind.

## Row meanings

- `EVID=0`: observation. `OUT` may be missing, but `OUTEQ` is required. `CENS` and a complete `C0`-`C3` polynomial are preserved.
- `EVID=1`: one explicit bolus or infusion. `INPUT` remains a string label, including numeric-looking labels. Canonical output does not use `ADDL`/`II`; expanded doses are written explicitly.
- `EVID=2`: covariate-only row. It must contain at least one covariate and no event fields.
- `EVID=4`: occasion boundary. It contains only `ID`, `EVID`, and `TIME`; all other fields must be missing.

Covariate observations are written at their exact stored times and source values rather than reconstructed from interpolation segments or sampled at event times. A covariate's fixed/nonfixed setting must be consistent wherever that covariate appears because the setting is encoded in its column header. Empty named covariates are rejected because an all-missing column cannot preserve per-occasion ownership.

## Rust API

```rust
use pharmsol::prelude::data::{read_pmetrics_csv_v1, Data};

let bytes: Vec<u8> = data.to_pmetrics_csv_v1()?;
let decoded: Data = read_pmetrics_csv_v1(bytes.as_slice())?;
data.write_pmetrics_csv_v1(&mut destination)?;
```

`Data::write_pmetrics(&File)` remains available and now writes the same canonical representation. Structural ambiguity, malformed marker rows, partial error polynomials, nonfinite values, and CSV/I/O failures return `DataError` rather than being silently normalized.
