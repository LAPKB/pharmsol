use super::{read_pmetrics, DataError};
use crate::{Censor, Data, ErrorPoly, Event, Subject, SubjectBuilderExt};
use std::collections::BTreeMap;
use tempfile::NamedTempFile;

const GOLDEN: &[u8] = include_bytes!("../../tests/data/pmetrics_csv.csv");
const CORE_HEADER: &str = "ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,CENS,C0,C1,C2,C3\n";

fn fixture_data() -> Data {
    let alpha = Subject::builder("alpha")
        .infusion(0.0, 25.0, "1", 2.0)
        .observation(0.0, 4.5, "central")
        .covariate("wt", 1.0, 60.0)
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
        .covariate("wt", 6.0, 72.0)
        .covariate("age", 12.0, 41.0)
        .reset()
        .bolus(0.0, 80.0, "oral")
        .censored_observation(3.0, 9.0, "1", Censor::ALOQ)
        .covariate("age", 0.0, 42.0)
        .covariate("crcl", 3.0, 80.0)
        .build();
    for occasion in ten.occasions_mut() {
        assert!(occasion.covariates_mut().set_covariate_fixed("age", true));
    }

    // Deliberately unsorted to check deterministic subject ordering.
    Data::new(vec![alpha, ten])
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
fn bytes_include_the_complete_dataset_and_are_deterministic() {
    let data = fixture_data();
    let first = data.as_bytes().unwrap();
    let second = data.as_bytes().unwrap();
    assert_eq!(first, GOLDEN);
    assert_eq!(first, second);
    assert!(std::str::from_utf8(&first).unwrap().contains(",4,"));
    assert!(std::str::from_utf8(&first).unwrap().contains(",2,"));
}

#[test]
fn bytes_round_trip_the_complete_dataset() {
    let data = fixture_data();
    let bytes = data.as_bytes().unwrap();
    let parsed = Data::from_bytes(&bytes).unwrap();
    assert_data_equivalent(&data, &parsed);
    assert_eq!(parsed.as_bytes().unwrap(), bytes);
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
fn read_pmetrics_uses_the_byte_reader() {
    let data = fixture_data();
    let bytes = data.as_bytes().unwrap();
    let file = NamedTempFile::new().unwrap();
    std::fs::write(file.path(), &bytes).unwrap();

    let from_file = read_pmetrics(file.path().display().to_string()).unwrap();
    let from_bytes = Data::from_bytes(&bytes).unwrap();
    assert_data_equivalent(&from_file, &from_bytes);
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

    let with_comment_and_missing = format!(
        "# comment\n{CORE_HEADER}s,4,0,.,.,.,.,.,.,.,.,.,.,.,.\ns,0,1,.,.,.,.,.,NA,cp,none,.,.,.,.\n"
    );
    let parsed = Data::from_bytes(with_comment_and_missing.as_bytes()).unwrap();
    match &parsed.subjects()[0].occasions()[0].events()[0] {
        Event::Observation(observation) => assert_eq!(observation.value(), None),
        _ => panic!("expected observation"),
    }
}

#[test]
fn origin_main_bincode_data_with_covariates_is_readable() {
    let bytes = include_bytes!("../../tests/data/pmetrics_data_origin_main.bincode");
    let mut data: Data = bincode::deserialize(bytes).unwrap();
    {
        let subject = data.subjects().into_iter().next().unwrap();
        assert_eq!(subject.id(), "legacy-covariates");
        assert_eq!(subject.occasions()[0].events().len(), 2);
        let covariates = subject.occasions()[0].covariates();
        let weight = covariates.get_covariate("wt").unwrap();
        assert_eq!(weight.observations().len(), 2);
        assert!((weight.interpolate(0.15).unwrap() - 0.65).abs() < 1e-12);
        let age = covariates.get_covariate("age").unwrap();
        assert!(age.fixed());
        assert_eq!(age.interpolate(10.0).unwrap(), 40.0);
    }

    assert!(matches!(
        data.as_bytes(),
        Err(DataError::LegacyLinearCovariate { name, id, occasion: 0 })
            if name == "wt" && id == "legacy-covariates"
    ));

    data.get_subject_mut("legacy-covariates")
        .unwrap()
        .occasions_mut()[0]
        .covariates_mut()
        .remove_covariate("wt");
    let encoded = data.as_bytes().unwrap();
    assert!(Data::from_bytes(&encoded).is_ok());
}

#[test]
fn malformed_and_nonfinite_data_fail_cleanly() {
    let boundary_with_covariate = concat!(
        "ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,CENS,C0,C1,C2,C3,age\n",
        "s,4,0,.,.,.,.,.,.,.,.,.,.,.,.,40\n"
    );
    assert!(matches!(
        Data::from_bytes(boundary_with_covariate.as_bytes()),
        Err(DataError::MalformedBoundaryRow { .. })
    ));

    let empty_covariate = format!("{CORE_HEADER}s,2,0,.,.,.,.,.,.,.,.,.,.,.,.\n");
    assert!(matches!(
        Data::from_bytes(empty_covariate.as_bytes()),
        Err(DataError::MalformedCovariateRow { .. })
    ));

    let nonfinite =
        format!("{CORE_HEADER}s,4,0,.,.,.,.,.,.,.,.,.,.,.,.\ns,1,NaN,0,1,.,.,iv,.,.,.,.,.,.,.\n");
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

#[test]
fn invalid_data_fails_cleanly() {
    let mut subject = Subject::builder("s").bolus(0.0, 1.0, "iv").build();
    subject.occasions_mut()[0].events_mut()[0].set_occasion(7);
    assert!(matches!(
        Data::new(vec![subject]).as_bytes(),
        Err(DataError::InvalidPmetricsData(_))
    ));

    let zero_duration = Data::new(vec![Subject::builder("s")
        .infusion(0.0, 1.0, "iv", 0.0)
        .build()]);
    assert!(matches!(
        zero_duration.as_bytes(),
        Err(DataError::InvalidPmetricsData(_))
    ));

    let reserved_label = Data::new(vec![Subject::builder("s").bolus(0.0, 1.0, "NA").build()]);
    assert!(matches!(
        reserved_label.as_bytes(),
        Err(DataError::InvalidPmetricsData(_))
    ));
}

#[test]
fn linear_covariate_values_reencode_bit_exactly() {
    let data = Data::new(vec![Subject::builder("s")
        .covariate("x", 0.1, 0.1)
        .covariate("x", 0.2, 1.2)
        .build()]);
    let first = data.as_bytes().unwrap();
    let parsed = Data::from_bytes(&first).unwrap();
    let second = parsed.as_bytes().unwrap();
    assert_eq!(first, second);
    assert_data_equivalent(&data, &parsed);
}

#[test]
fn csv_special_strings_round_trip() {
    let data = Data::new(vec![Subject::builder("#s,\ncontinued")
        .bolus(0.0, 1.0, "iv,#\nroute")
        .observation(1.0, 2.0, "cp,#\nlabel")
        .build()]);
    let bytes = data.as_bytes().unwrap();
    let parsed = Data::from_bytes(&bytes).unwrap();
    assert_data_equivalent(&data, &parsed);
    assert_eq!(parsed.as_bytes().unwrap(), bytes);
}
