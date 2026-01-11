#!/usr/bin/env Rscript
# =============================================================================
# PKNCA Cross-Validation: Generate Expected Values
# =============================================================================
#
# This script reads test scenarios from test_scenarios.json, runs PKNCA to
# compute NCA parameters, and saves the expected values to expected_values.json.
#
# Usage: Rscript generate_expected.R
#
# Requirements: R with PKNCA, jsonlite packages installed
# =============================================================================

library(PKNCA)
library(jsonlite)

cat("PKNCA Cross-Validation - Generating Expected Values\n")
cat("====================================================\n\n")

# Read test scenarios
scenarios_raw <- fromJSON("test_scenarios.json", simplifyVector = FALSE)
scenarios <- scenarios_raw$scenarios
cat(sprintf("Loaded %d test scenarios\n\n", length(scenarios)))

# Helper function to map our route names to PKNCA expectations
get_route_for_pknca <- function(route) {
    switch(route,
        "extravascular" = "extravascular",
        "iv_bolus" = "intravascular",
        "iv_infusion" = "intravascular",
        route
    )
}

# Helper function to get AUC method name for PKNCA
get_auc_method <- function(method) {
    if (is.null(method)) {
        return("lin up/log down")
    }
    method
}

# Process each scenario
results <- list()

for (scenario in scenarios) {
    cat(sprintf("Processing: %s (%s)\n", scenario$name, scenario$id))

    tryCatch(
        {
            # Build concentration data frame - unlist JSON arrays
            times <- unlist(scenario$times)
            concs <- unlist(scenario$concentrations)

            conc_data <- data.frame(
                ID = 1,
                time = times,
                conc = concs
            )

            # Handle BLQ if specified
            if (!is.null(scenario$blq_indices)) {
                # Mark BLQ as 0 (PKNCA convention)
                # Note: blq_indices are 0-based from JSON
                blq_idx <- unlist(scenario$blq_indices)
                for (idx in blq_idx) {
                    conc_data$conc[idx + 1] <- 0
                }
            }

            # Build dose data frame
            dose_data <- data.frame(
                ID = 1,
                time = scenario$dose$time,
                dose = scenario$dose$amount
            )

            # Add duration for infusions
            if (scenario$route == "iv_infusion" && !is.null(scenario$dose$duration)) {
                dose_data$duration <- scenario$dose$duration
            }

            # Create PKNCA objects
            conc_obj <- PKNCAconc(conc_data, conc ~ time | ID)

            if (scenario$route == "iv_infusion" && !is.null(scenario$dose$duration)) {
                dose_obj <- PKNCAdose(dose_data, dose ~ time | ID,
                    route = "intravascular",
                    duration = "duration"
                )
            } else {
                dose_obj <- PKNCAdose(dose_data, dose ~ time | ID,
                    route = get_route_for_pknca(scenario$route)
                )
            }

            # Set up intervals - request all parameters up to infinity
            intervals <- data.frame(
                start = 0,
                end = Inf,
                cmax = TRUE,
                tmax = TRUE,
                tlast = TRUE,
                clast.obs = TRUE,
                auclast = TRUE,
                aucall = TRUE,
                aumclast = TRUE,
                half.life = TRUE,
                lambda.z = TRUE,
                r.squared = TRUE,
                adj.r.squared = TRUE,
                lambda.z.n.points = TRUE,
                clast.pred = TRUE,
                aucinf.obs = TRUE,
                aucinf.pred = TRUE,
                aumcinf.obs = TRUE,
                aumcinf.pred = TRUE,
                mrt.obs = TRUE,
                tlag = TRUE
            )

            # Add route-specific parameters
            if (scenario$route == "iv_bolus") {
                intervals$c0 <- TRUE
                intervals$vz.obs <- TRUE
                intervals$cl.obs <- TRUE
                intervals$vss.obs <- TRUE
            } else if (scenario$route == "iv_infusion") {
                intervals$cl.obs <- TRUE
                intervals$vss.obs <- TRUE
            } else {
                intervals$vz.obs <- TRUE
                intervals$cl.obs <- TRUE
            }

            # Add partial AUC if specified
            if (!is.null(scenario$partial_auc_interval)) {
                partial_int <- unlist(scenario$partial_auc_interval)
                partial_interval <- data.frame(
                    start = partial_int[1],
                    end = partial_int[2],
                    auclast = TRUE
                )
            }

            # Set PKNCA options
            auc_method <- get_auc_method(scenario$auc_method)

            # Determine BLQ handling
            blq_handling <- if (!is.null(scenario$blq_rule)) {
                switch(scenario$blq_rule,
                    "exclude" = "drop",
                    "zero" = "keep",
                    "positional" = list(first = "keep", middle = "drop", last = "keep"),
                    "drop"
                )
            } else {
                "drop"
            }

            # Create PKNCAdata with options
            data_obj <- PKNCAdata(
                conc_obj, dose_obj,
                intervals = intervals,
                options = list(
                    auc.method = auc_method,
                    conc.blq = blq_handling
                )
            )

            # Run NCA
            nca_result <- pk.nca(data_obj)

            # Extract results
            result_df <- as.data.frame(nca_result)

            # Convert to named list
            param_values <- list()
            for (i in 1:nrow(result_df)) {
                param_name <- result_df$PPTESTCD[i]
                param_value <- result_df$PPORRES[i]
                if (!is.na(param_value)) {
                    param_values[[param_name]] <- param_value
                }
            }

            # Calculate partial AUC if requested
            if (!is.null(scenario$partial_auc_interval)) {
                partial_int <- unlist(scenario$partial_auc_interval)
                start_t <- partial_int[1]
                end_t <- partial_int[2]
                partial_auc <- pk.calc.auc(
                    conc_data$conc, conc_data$time,
                    interval = c(start_t, end_t),
                    method = auc_method,
                    auc.type = "AUClast"
                )
                param_values[["partial_auc"]] <- partial_auc
                param_values[["partial_auc_start"]] <- start_t
                param_values[["partial_auc_end"]] <- end_t
            }

            # Store results
            results[[scenario$id]] <- list(
                id = scenario$id,
                name = scenario$name,
                pknca_version = as.character(packageVersion("PKNCA")),
                auc_method = auc_method,
                blq_rule = scenario$blq_rule,
                parameters = param_values
            )

            cat(sprintf("  -> Computed %d parameters\n", length(param_values)))
        },
        error = function(e) {
            cat(sprintf("  -> ERROR: %s\n", e$message))
            results[[scenario$id]] <<- list(
                id = scenario$id,
                name = scenario$name,
                error = e$message
            )
        }
    )
}

# Create output structure
output <- list(
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
    r_version = R.version.string,
    pknca_version = as.character(packageVersion("PKNCA")),
    scenario_count = length(results),
    results = results
)

# Write to JSON
output_file <- "expected_values.json"
write_json(output, output_file, pretty = TRUE, auto_unbox = TRUE)

cat(sprintf("\n✓ Generated expected values for %d scenarios\n", length(results)))
cat(sprintf("✓ Saved to: %s\n", output_file))
