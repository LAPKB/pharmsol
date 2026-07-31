use super::{read_pmetrics, DataError};
use crate::{Censor, Covariate, Data, ErrorPoly, Event, Subject, SubjectBuilderExt};
use csv::StringRecord;
use std::collections::BTreeMap;
use tempfile::NamedTempFile;

const GOLDEN: &[u8] = include_bytes!("../../tests/data/pmetrics_csv.csv");
const CORE_HEADER: &str = "ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,CENS,C0,C1,C2,C3\n";

fn fixture_data() -> Data {
    let alpha = Subject::builder("alpha")
        .infusion(0.0, 25.0, "1", 2.0)
        .observation(0.0, 4.5, "central")
        .covariate("wt", 0.0, 60.0)
        .observation(2.0, -98.5, "neg")
        .build();

    let mut ten = Subject::builder("10")
        .bolus(0.0, 100.0, "iv")
        .observation_with_error(
            0.0,
            1.25,
            "cp",
            ErrorPoly::new(0.1, 0.2, 0.3, 0.4),
            Censor::BLOQ,
        )
        .missing_observation(1.0, "2")
        .infusion(2.0, 50.0, "1", 4.0)
        .covariate("age", 0.0, 40.0)
        .covariate("wt", 0.0, 70.0)
        .covariate("wt", 2.0, 72.0)
        .covariate("age", 2.0, 41.0)
        .reset()
        .bolus(0.0, 80.0, "oral")
        .censored_observation(3.0, 9.0, "1", Censor::ALOQ)
        .covariate("age", 0.0, 42.0)
        .covariate("crcl", 3.0, 80.0)
        .build();
    for occasion in ten.occasions_mut() {
        assert!(occasion.covariates_mut().set_covariate_fixed("age", true));
    }

    Data::new(vec![alpha, ten])
}

fn records(bytes: &[u8]) -> Vec<StringRecord> {
    csv::Reader::from_reader(bytes)
        .records()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

fn assert_f64(left: f64, right: f64) {
    assert_eq!(left.to_bits(), right.to_bits());
}

fn subjects_by_id(data: &Data) -> BTreeMap<String, &Subject> {
    data.subjects()
        .into_iter()
        .map(|subject| (subject.id().clone(), subject))
        .collect()
}

fn assert_data_equivalent(left: &Data, right: &Data) {
    let left_subjects = subjects_by_id(left);
    let right_subjects = subjects_by_id(right);
    assert_eq!(
        left_subjects.keys().collect::<Vec<_>>(),
        right_subjects.keys().collect::<Vec<_>>()
    );

    for (id, left_subject) in left_subjects {
        let right_subject = right_subjects[&id];
        assert_eq!(
            left_subject.occasions().len(),
            right_subject.occasions().len()
        );
        for (left_occasion, right_occasion) in left_subject
            .occasions()
            .iter()
            .zip(right_subject.occasions())
        {
            assert_eq!(left_occasion.index(), right_occasion.index());
            assert_eq!(left_occasion.events().len(), right_occasion.events().len());
            for (left_event, right_event) in
                left_occasion.events().iter().zip(right_occasion.events())
            {
                assert_f64(left_event.time(), right_event.time());
                assert_eq!(left_event.occasion(), right_event.occasion());
                match (left_event, right_event) {
                    (Event::Bolus(left), Event::Bolus(right)) => {
                        assert_f64(left.amount(), right.amount());
                        assert_eq!(left.input(), right.input());
                    }
                    (Event::Infusion(left), Event::Infusion(right)) => {
                        assert_f64(left.amount(), right.amount());
                        assert_f64(left.duration(), right.duration());
                        assert_eq!(left.input(), right.input());
                    }
                    (Event::Observation(left), Event::Observation(right)) => {
                        assert_eq!(
                            left.value().map(f64::to_bits),
                            right.value().map(f64::to_bits)
                        );
                        assert_eq!(left.outeq(), right.outeq());
                        assert_eq!(left.censoring(), right.censoring());
                        assert_eq!(
                            left.errorpoly().map(|poly| poly.coefficients()),
                            right.errorpoly().map(|poly| poly.coefficients())
                        );
                    }
                    _ => panic!("event variants differ"),
                }
            }

            let left_covariates = left_occasion.covariates().covariates();
            let right_covariates = right_occasion.covariates().covariates();
            let mut names = left_covariates.keys().collect::<Vec<_>>();
            names.sort();
            let mut right_names = right_covariates.keys().collect::<Vec<_>>();
            right_names.sort();
            assert_eq!(names, right_names);
            for name in names {
                let left_covariate = left_covariates[name];
                let right_covariate = right_covariates[name];
                assert_eq!(left_covariate.fixed(), right_covariate.fixed());
                assert_eq!(
                    left_covariate
                        .observations()
                        .into_iter()
                        .map(|(time, value)| (time.to_bits(), value.to_bits()))
                        .collect::<Vec<_>>(),
                    right_covariate
                        .observations()
                        .into_iter()
                        .map(|(time, value)| (time.to_bits(), value.to_bits()))
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn representable_data_round_trips_with_exact_covariate_values() {
    let data = fixture_data();
    let first = data.as_bytes().unwrap();
    let second = data.as_bytes().unwrap();
    assert_eq!(first, GOLDEN);
    assert_eq!(first, second);

    let parsed = Data::from_bytes(&first).unwrap();
    assert_data_equivalent(&data, &parsed);
    assert_eq!(parsed.as_bytes().unwrap(), first);
}

#[test]
fn write_pmetrics_writes_as_bytes() {
    let data = fixture_data();
    let expected = data.as_bytes().unwrap();
    let file = NamedTempFile::new().unwrap();
    data.write_pmetrics(file.as_file()).unwrap();
    assert_eq!(std::fs::read(file.path()).unwrap(), expected);
}

#[test]
fn existing_pmetrics_files_remain_readable() {
    let path = format!(
        "{}/src/tests/data/addl_test.csv",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap();
    let from_file = read_pmetrics(path).unwrap();
    let from_bytes = Data::from_bytes(&bytes).unwrap();
    assert_data_equivalent(&from_file, &from_bytes);

    let uppercase_covariate = concat!(
        "ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,CENS,C0,C1,C2,C3,WT\n",
        "s,1,0,0,1,.,.,iv,.,.,.,.,.,.,.,70\n"
    );
    let parsed = Data::from_bytes(uppercase_covariate.as_bytes()).unwrap();
    assert!(parsed.subjects()[0].occasions()[0]
        .covariates()
        .get_covariate("wt")
        .is_some());

    let populated_unused_fields = concat!(
        "ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,CENS,C0,C1,C2,C3\n",
        "s,0,0,0,0,0,0,unused,1,cp,0,0,0,0,0\n",
        "s,1,1,0,1,0,0,iv,-99,unused,0,0,0,0,0\n"
    );
    assert!(Data::from_bytes(populated_unused_fields.as_bytes()).is_ok());
}

#[test]
fn duplicate_and_conflicting_headers_are_rejected() {
    let core = CORE_HEADER.trim_end();
    let invalid_headers = [
        format!("{core},WT,wt\n"),
        format!("{core},WT!,wt!\n"),
        format!("{core},WT,wt!\n"),
        format!("id,{core}\n"),
        format!("{core},wt!!\n"),
        format!("{core},wt!x\n"),
    ];

    for header in invalid_headers {
        assert!(matches!(
            Data::from_bytes(header.as_bytes()),
            Err(DataError::InvalidPmetricsData(_))
        ));
    }
}

#[test]
fn valid_mixed_case_covariate_headers_are_normalized() {
    let input = concat!(
        "ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,CENS,C0,C1,C2,C3,WT!,Ka\n",
        "s,1,0,0,1,.,.,iv,.,.,.,.,.,.,.,70,0.5\n"
    );
    let data = Data::from_bytes(input.as_bytes()).unwrap();
    let covariates = data.subjects()[0].occasions()[0].covariates();
    let wt = covariates.get_covariate("wt").unwrap();
    assert_eq!(wt.name(), "wt");
    assert!(wt.fixed());
    let ka = covariates.get_covariate("ka").unwrap();
    assert_eq!(ka.name(), "ka");
    assert!(!ka.fixed());
}

#[test]
fn standalone_covariate_time_is_not_exportable() {
    let data = Data::new(vec![Subject::builder("s")
        .bolus(0.0, 1.0, "iv")
        .covariate("wt", 1.0, 70.0)
        .build()]);
    assert!(matches!(
        data.as_bytes(),
        Err(DataError::InvalidPmetricsData(message))
            if message.contains("has no dose or observation row")
    ));
}

#[test]
fn first_occasion_has_no_invented_reset_row() {
    let data = Data::new(vec![Subject::builder("s")
        .bolus(0.0, 1.0, "iv")
        .observation(1.0, 2.0, "cp")
        .build()]);
    let rows = records(&data.as_bytes().unwrap());
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].get(1), Some("1"));
    assert_eq!(rows[1].get(1), Some("0"));
}

#[test]
fn later_occasion_starts_with_a_real_reset_dose() {
    let data = Data::new(vec![Subject::builder("s")
        .bolus(0.0, 1.0, "iv")
        .reset()
        .bolus(0.0, 2.0, "oral")
        .observation(1.0, 3.0, "cp")
        .build()]);
    let rows = records(&data.as_bytes().unwrap());
    assert_eq!(
        rows.iter()
            .map(|row| row.get(1).unwrap())
            .collect::<Vec<_>>(),
        ["1", "4", "0"]
    );
    assert_eq!(rows[1].get(4), Some("2"));
    assert_eq!(rows[1].get(7), Some("oral"));
}

#[test]
fn later_occasion_without_an_initial_dose_is_not_exportable() {
    let data = Data::new(vec![Subject::builder("s")
        .bolus(0.0, 1.0, "iv")
        .reset()
        .observation(0.0, 3.0, "cp")
        .build()]);
    assert!(matches!(
        data.as_bytes(),
        Err(DataError::InvalidPmetricsData(message)) if message.contains("must begin with a dose")
    ));
}

#[test]
fn observation_at_reset_time_makes_later_occasion_unsafe_to_export() {
    let input = concat!(
        "ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,CENS,C0,C1,C2,C3\n",
        "s,1,0,0,1,.,.,iv,.,.,.,.,.,.,.\n",
        "s,4,0,0,2,.,.,iv,.,.,.,.,.,.,.\n",
        "s,0,0,.,.,.,.,.,3,cp,0,.,.,.,.\n"
    );
    let data = Data::from_bytes(input.as_bytes()).unwrap();
    assert!(matches!(
        data.as_bytes(),
        Err(DataError::InvalidPmetricsData(message))
            if message.contains("must begin with a dose at time 0")
    ));
}

#[test]
fn negative_addl_reset_is_rejected_while_reading() {
    let input = concat!(
        "ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,CENS,C0,C1,C2,C3\n",
        "s,1,0,0,1,.,.,iv,.,.,.,.,.,.,.\n",
        "s,4,0,0,2,-2,1,iv,.,.,.,.,.,.,.\n"
    );
    assert!(matches!(
        Data::from_bytes(input.as_bytes()),
        Err(DataError::InvalidPmetricsData(message))
            if message.contains("cannot use negative ADDL with positive II")
    ));
}

#[test]
fn negative_addl_reset_at_later_time_is_rejected_before_expansion() {
    let input = concat!(
        "ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,CENS,C0,C1,C2,C3\n",
        "s,1,0,0,1,.,.,iv,.,.,.,.,.,.,.\n",
        "s,4,2,0,2,-2,1,iv,.,.,.,.,.,.,.\n"
    );
    assert!(matches!(
        Data::from_bytes(input.as_bytes()),
        Err(DataError::InvalidPmetricsData(message))
            if message.contains("cannot use negative ADDL with positive II")
    ));
}

#[test]
fn positive_addl_reset_round_trips_as_individual_doses() {
    let input = concat!(
        "ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,CENS,C0,C1,C2,C3\n",
        "s,1,0,0,1,.,.,iv,.,.,.,.,.,.,.\n",
        "s,4,0,0,2,2,1,iv,.,.,.,.,.,.,.\n"
    );
    let data = Data::from_bytes(input.as_bytes()).unwrap();
    let bytes = data.as_bytes().unwrap();
    let rows = records(&bytes);
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[1].get(1), Some("4"));
    assert_eq!(rows[1].get(2), Some("0"));
    assert_eq!(rows[2].get(1), Some("1"));
    assert_eq!(rows[2].get(2), Some("1"));
    assert_eq!(rows[3].get(1), Some("1"));
    assert_eq!(rows[3].get(2), Some("2"));
    assert!(rows[1..].iter().all(|row| row.get(5) == Some(".")));

    let reparsed = Data::from_bytes(&bytes).unwrap();
    assert_data_equivalent(&data, &reparsed);
}

#[test]
fn missing_observations_use_minus_99() {
    let data = Data::new(vec![Subject::builder("s")
        .missing_observation(0.0, "cp")
        .build()]);
    let bytes = data.as_bytes().unwrap();
    let rows = records(&bytes);
    assert_eq!(rows[0].get(8), Some("-99"));

    let parsed = Data::from_bytes(&bytes).unwrap();
    match &parsed.subjects()[0].occasions()[0].events()[0] {
        Event::Observation(observation) => assert_eq!(observation.value(), None),
        _ => panic!("expected observation"),
    }

    let placeholders = format!(
        "{CORE_HEADER}s,0,0,.,.,.,.,.,.,cp,0,.,.,.,.\ns,0,1,.,.,.,.,.,NA,cp,0,.,.,.,.\ns,0,2,.,.,.,.,.,,cp,0,.,.,.,.\n"
    );
    assert!(Data::from_bytes(placeholders.as_bytes()).is_ok());
}

#[test]
fn actual_minus_99_observation_is_not_exportable() {
    let data = Data::new(vec![Subject::builder("s")
        .observation(0.0, -99.0, "cp")
        .build()]);
    assert!(matches!(
        data.as_bytes(),
        Err(DataError::InvalidPmetricsData(message)) if message.contains("reserved for missing")
    ));
}

#[test]
fn old_bincode_data_retains_interpolation_behavior() {
    let bytes = include_bytes!("../../tests/data/pmetrics_data_origin_main.bincode");
    let mut data: Data = bincode::deserialize(bytes).unwrap();
    let subject = data.subjects().into_iter().next().unwrap();
    assert_eq!(subject.id(), "legacy-covariates");
    assert_eq!(subject.occasions()[0].events().len(), 2);
    let covariates = subject.occasions()[0].covariates();
    let weight = covariates.get_covariate("wt").unwrap();
    let original = weight.observations();
    assert_eq!(original.len(), 2);
    let midpoint = (original[0].0 + original[1].0) / 2.0;
    let original_midpoint_value = weight.interpolate(midpoint).unwrap();
    assert!((original_midpoint_value - 0.65).abs() < 1e-12);
    let age = covariates.get_covariate("age").unwrap();
    assert!(age.fixed());
    assert_eq!(age.interpolate(10.0).unwrap(), 40.0);

    let added = (original[1].0 + 0.1, original[1].1 + 1.0);
    let updated = (original[1].0, original[1].1 + 0.25);
    let weight = data
        .get_subject_mut("legacy-covariates")
        .unwrap()
        .occasions_mut()[0]
        .covariates_mut()
        .get_covariate_mut("wt")
        .unwrap();
    weight.add_observation(added.0, added.1);
    assert_eq!(weight.observations(), [original[0], original[1], added]);
    assert!((weight.interpolate(midpoint).unwrap() - original_midpoint_value).abs() < 1e-12);

    weight.update_observation(updated.0, updated.1);
    assert_eq!(weight.observations(), [original[0], updated, added]);
    assert_eq!(weight.interpolate(updated.0).unwrap(), updated.1);
}

#[test]
fn evid_2_is_neither_written_nor_read() {
    let bytes = fixture_data().as_bytes().unwrap();
    assert!(records(&bytes).iter().all(|row| row.get(1) != Some("2")));

    let unsupported = concat!(
        "ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,CENS,C0,C1,C2,C3,wt\n",
        "s,2,0,.,.,.,.,.,.,.,.,.,.,.,.,70\n"
    );
    assert!(matches!(
        Data::from_bytes(unsupported.as_bytes()),
        Err(DataError::InvalidPmetricsData(message)) if message.contains("unsupported EVID=2")
    ));
}

#[test]
fn empty_reset_rows_are_rejected() {
    let empty_reset = format!("{CORE_HEADER}s,4,0,.,.,.,.,.,.,.,.,.,.,.,.\n");
    assert!(matches!(
        Data::from_bytes(empty_reset.as_bytes()),
        Err(DataError::InvalidPmetricsData(message)) if message.contains("must contain a dose")
    ));
}

#[test]
fn programmatic_covariate_name_is_lowercased_for_export() {
    let data = Data::new(vec![Subject::builder("s")
        .bolus(0.0, 1.0, "iv")
        .covariate("WT", 0.0, 70.0)
        .build()]);
    let bytes = data.as_bytes().unwrap();
    let mut reader = csv::Reader::from_reader(bytes.as_slice());
    assert_eq!(reader.headers().unwrap().iter().last(), Some("wt"));

    let parsed = Data::from_bytes(&bytes).unwrap();
    let weight = parsed.subjects()[0].occasions()[0]
        .covariates()
        .get_covariate("wt")
        .unwrap();
    assert_eq!(weight.observations(), [(0.0, 70.0)]);
}

#[test]
fn covariate_key_and_name_may_differ_by_ascii_case() {
    let mut subject = Subject::builder("s").bolus(0.0, 1.0, "iv").build();
    let mut weight = Covariate::new("WT".to_string(), false);
    weight.add_observation(0.0, 70.0);
    subject.occasions_mut()[0]
        .covariates_mut()
        .add_covariate("wt".to_string(), weight);

    let bytes = Data::new(vec![subject]).as_bytes().unwrap();
    let mut reader = csv::Reader::from_reader(bytes.as_slice());
    assert_eq!(reader.headers().unwrap().iter().last(), Some("wt"));
}

#[test]
fn same_occasion_covariate_case_variants_are_rejected() {
    let data = Data::new(vec![Subject::builder("s")
        .bolus(0.0, 1.0, "iv")
        .covariate("WT", 0.0, 70.0)
        .covariate("wt", 0.0, 71.0)
        .build()]);
    assert!(matches!(
        data.as_bytes(),
        Err(DataError::InvalidPmetricsData(message)) if message.contains("both map to `wt`")
    ));
}

#[test]
fn covariate_case_variants_across_occasions_share_one_column() {
    let data = Data::new(vec![Subject::builder("s")
        .bolus(0.0, 1.0, "iv")
        .covariate("WT", 0.0, 70.0)
        .reset()
        .bolus(0.0, 2.0, "iv")
        .covariate("wt", 0.0, 71.0)
        .build()]);
    let bytes = data.as_bytes().unwrap();
    let mut reader = csv::Reader::from_reader(bytes.as_slice());
    assert_eq!(
        reader
            .headers()
            .unwrap()
            .iter()
            .filter(|header| *header == "wt")
            .count(),
        1
    );

    let parsed = Data::from_bytes(&bytes).unwrap();
    assert_eq!(
        parsed.subjects()[0].occasions()[0]
            .covariates()
            .get_covariate("wt")
            .unwrap()
            .observations(),
        [(0.0, 70.0)]
    );
    assert_eq!(
        parsed.subjects()[0].occasions()[1]
            .covariates()
            .get_covariate("wt")
            .unwrap()
            .observations(),
        [(0.0, 71.0)]
    );
}

#[test]
fn covariate_case_variants_with_different_behavior_are_rejected() {
    let mut subject = Subject::builder("s")
        .bolus(0.0, 1.0, "iv")
        .covariate("WT", 0.0, 70.0)
        .reset()
        .bolus(0.0, 2.0, "iv")
        .covariate("wt", 0.0, 71.0)
        .build();
    assert!(subject.occasions_mut()[0]
        .covariates_mut()
        .set_covariate_fixed("WT", true));

    assert!(matches!(
        Data::new(vec![subject]).as_bytes(),
        Err(DataError::InvalidPmetricsData(message))
            if message.contains("inconsistent fixed settings")
    ));
}

#[test]
fn exact_linear_covariate_values_round_trip() {
    let data = Data::new(vec![Subject::builder("s")
        .bolus(0.1, 1.0, "iv")
        .observation(0.2, 2.0, "cp")
        .covariate("x", 0.1, 0.1)
        .covariate("x", 0.2, 1.2)
        .build()]);
    let first = data.as_bytes().unwrap();
    let parsed = Data::from_bytes(&first).unwrap();
    assert_eq!(parsed.as_bytes().unwrap(), first);
    assert_data_equivalent(&data, &parsed);
}

#[test]
fn malformed_and_nonfinite_values_fail_cleanly() {
    let partial_error = format!("{CORE_HEADER}s,0,0,.,.,.,.,.,1,cp,0,0.1,.,.,.\n");
    assert!(matches!(
        Data::from_bytes(partial_error.as_bytes()),
        Err(DataError::InvalidPmetricsData(_))
    ));

    let nonfinite = format!("{CORE_HEADER}s,1,NaN,0,1,.,.,iv,.,.,.,.,.,.,.\n");
    assert!(matches!(
        Data::from_bytes(nonfinite.as_bytes()),
        Err(DataError::NonFiniteValue { .. })
    ));

    let invalid_data = Data::new(vec![Subject::builder("s")
        .bolus(f64::INFINITY, 1.0, "iv")
        .build()]);
    assert!(matches!(
        invalid_data.as_bytes(),
        Err(DataError::NonFiniteValue { .. })
    ));
}
