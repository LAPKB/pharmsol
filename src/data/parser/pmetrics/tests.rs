use super::core_headers;
use super::{read_pmetrics, DataError};
use crate::{Censor, Covariate, Data, ErrorPoly, Event, Subject, SubjectBuilderExt};
use csv::StringRecord;
use std::collections::BTreeMap;
use std::io::Write;
use tempfile::NamedTempFile;

const GOLDEN: &[u8] = concat!(
    "ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,CENS,C0,C1,C2,C3,age!,crcl,wt\n",
    "10,0,0,.,.,.,.,.,1.25,cp,1,0.1,0.2,0.3,0.4,40,.,70\n",
    "10,1,0,0,100,.,.,iv,.,.,.,.,.,.,.,40,.,70\n",
    "10,0,1,.,.,.,.,.,-99,2,0,.,.,.,.,.,.,.\n",
    "10,1,2,4,50,.,.,1,.,.,.,.,.,.,.,41,.,72\n",
    "10,4,0,0,80,.,.,oral,.,.,.,.,.,.,.,42,.,.\n",
    "10,0,3,.,.,.,.,.,9,1,-1,.,.,.,.,.,80,.\n",
    "alpha,0,0,.,.,.,.,.,4.5,central,0,.,.,.,.,.,.,60\n",
    "alpha,1,0,2,25,.,.,1,.,.,.,.,.,.,.,.,.,60\n",
    "alpha,0,2,.,.,.,.,.,-98.5,neg,0,.,.,.,.,.,.,.\n",
)
.as_bytes();

fn pmetrics_input(covariates: &[&str], rows: &str) -> String {
    let mut headers = core_headers().collect::<Vec<_>>();
    headers.extend(covariates.iter().copied());
    format!("{}\n{rows}", headers.join(","))
}

fn core_header() -> String {
    pmetrics_input(&[], "")
}

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
    ::csv::Reader::from_reader(bytes)
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
    let first = data.to_pmetrics_csv_bytes().unwrap();
    let second = data.to_pmetrics_csv_bytes().unwrap();
    assert_eq!(first, GOLDEN);
    assert_eq!(first, second);

    let parsed = Data::from_pmetrics_csv_bytes(&first).unwrap();
    assert_data_equivalent(&data, &parsed);
    assert_eq!(parsed.to_pmetrics_csv_bytes().unwrap(), first);
}

#[test]
fn write_pmetrics_replaces_existing_file_contents() {
    let data = fixture_data();
    let expected = data.to_pmetrics_csv_bytes().unwrap();
    let file = NamedTempFile::new().unwrap();
    let mut output = file.as_file();
    output.write_all(&vec![b'x'; expected.len() + 100]).unwrap();

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
    let from_bytes = Data::from_pmetrics_csv_bytes(&bytes).unwrap();
    assert_data_equivalent(&from_file, &from_bytes);

    let uppercase_covariate = pmetrics_input(&["WT"], "s,1,0,0,1,.,.,iv,.,.,.,.,.,.,.,70\n");
    let parsed = Data::from_pmetrics_csv_bytes(uppercase_covariate.as_bytes()).unwrap();
    assert!(parsed.subjects()[0].occasions()[0]
        .covariates()
        .get_covariate("wt")
        .is_some());

    let populated_unused_fields = pmetrics_input(
        &[],
        concat!(
            "s,0,0,0,0,0,0,unused,1,cp,0,0,0,0,0\n",
            "s,1,1,0,1,0,0,iv,-99,unused,0,0,0,0,0\n"
        ),
    );
    assert!(Data::from_pmetrics_csv_bytes(populated_unused_fields.as_bytes()).is_ok());
}

#[test]
fn duplicate_and_conflicting_headers_are_rejected() {
    let core = core_headers().collect::<Vec<_>>().join(",");
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
            Data::from_pmetrics_csv_bytes(header.as_bytes()),
            Err(DataError::InvalidPmetricsData(_))
        ));
    }
}

#[test]
fn required_core_headers_are_validated_without_data_rows() {
    for (input, missing) in [
        ("", "ID"),
        ("EVID,TIME\n", "ID"),
        ("ID,TIME\n", "EVID"),
        ("ID,EVID\n", "TIME"),
    ] {
        assert!(matches!(
            Data::from_pmetrics_csv_bytes(input.as_bytes()),
            Err(DataError::InvalidPmetricsData(message))
                if message.contains(&format!("missing required core header `{missing}`"))
        ));
    }
}

#[test]
fn unused_core_headers_may_be_omitted() {
    let dose = Data::from_pmetrics_csv_bytes(b"ID,EVID,TIME,DOSE,INPUT\ns,1,0,100,iv\n").unwrap();
    assert!(matches!(
        dose.subjects()[0].occasions()[0].events()[0],
        Event::Bolus(_)
    ));

    let observation =
        Data::from_pmetrics_csv_bytes(b"ID,EVID,TIME,OUT,OUTEQ\ns,0,0,1.5,cp\n").unwrap();
    assert!(matches!(
        observation.subjects()[0].occasions()[0].events()[0],
        Event::Observation(_)
    ));
}

#[test]
fn valid_mixed_case_covariate_headers_are_normalized() {
    let input = pmetrics_input(&["WT!", "Ka"], "s,1,0,0,1,.,.,iv,.,.,.,.,.,.,.,70,0.5\n");
    let data = Data::from_pmetrics_csv_bytes(input.as_bytes()).unwrap();
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
        data.to_pmetrics_csv_bytes(),
        Err(DataError::UnrepresentablePmetricsData(message))
            if message.contains("has no dose or observation row")
    ));
}

#[test]
fn first_occasion_has_no_invented_reset_row() {
    let data = Data::new(vec![Subject::builder("s")
        .bolus(0.0, 1.0, "iv")
        .observation(1.0, 2.0, "cp")
        .build()]);
    let rows = records(&data.to_pmetrics_csv_bytes().unwrap());
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].get(1), Some("1"));
    assert_eq!(rows[1].get(1), Some("0"));
}

#[test]
fn later_occasion_starts_with_a_real_reset_dose() {
    let data = Data::new(vec![Subject::builder("s")
        .bolus(0.0, 1.0, "iv")
        .reset()
        .bolus(24.0, 2.0, "oral")
        .observation(25.0, 3.0, "cp")
        .build()]);
    let rows = records(&data.to_pmetrics_csv_bytes().unwrap());
    assert_eq!(
        rows.iter()
            .map(|row| row.get(1).unwrap())
            .collect::<Vec<_>>(),
        ["1", "4", "0"]
    );
    assert_eq!(rows[1].get(2), Some("24"));
    assert_eq!(rows[1].get(4), Some("2"));
    assert_eq!(rows[1].get(7), Some("oral"));
}

#[test]
fn mixed_dose_types_at_reset_time_are_not_exportable() {
    let input = pmetrics_input(
        &[],
        concat!(
            "s,1,0,0,1,.,.,iv,.,.,.,.,.,.,.\n",
            "s,4,24,2,2,.,.,iv,.,.,.,.,.,.,.\n",
            "s,1,24,0,3,.,.,oral,.,.,.,.,.,.,.\n"
        ),
    );
    let data = Data::from_pmetrics_csv_bytes(input.as_bytes()).unwrap();
    assert!(matches!(
        data.to_pmetrics_csv_bytes(),
        Err(DataError::UnrepresentablePmetricsData(message))
            if message.contains("both bolus and infusion doses at reset time 24")
    ));
}

#[test]
fn ordering_error_describes_all_equal_time_rules() {
    let mut subject = Subject::builder("s")
        .bolus(0.0, 1.0, "iv")
        .infusion(0.0, 2.0, "iv", 1.0)
        .build();
    subject.occasions_mut()[0].events_mut().reverse();

    assert!(matches!(
        Data::new(vec![subject]).to_pmetrics_csv_bytes(),
        Err(DataError::UnrepresentablePmetricsData(message))
            if message.contains("boluses before infusions at equal times")
    ));
}

#[test]
fn same_time_bolus_doses_keep_the_reset_order() {
    let input = pmetrics_input(
        &[],
        concat!(
            "s,1,0,0,1,.,.,iv,.,.,.,.,.,.,.\n",
            "s,4,0,0,2,.,.,iv,.,.,.,.,.,.,.\n",
            "s,1,0,0,3,.,.,oral,.,.,.,.,.,.,.\n"
        ),
    );
    let rows = records(
        &Data::from_pmetrics_csv_bytes(input.as_bytes())
            .unwrap()
            .to_pmetrics_csv_bytes()
            .unwrap(),
    );
    assert_eq!(rows[1].get(1), Some("4"));
    assert_eq!(rows[1].get(4), Some("2"));
    assert_eq!(rows[1].get(7), Some("iv"));
    assert_eq!(rows[2].get(1), Some("1"));
    assert_eq!(rows[2].get(4), Some("3"));
    assert_eq!(rows[2].get(7), Some("oral"));
}

#[test]
fn same_time_infusion_doses_keep_the_reset_order() {
    let input = pmetrics_input(
        &[],
        concat!(
            "s,1,0,0,1,.,.,iv,.,.,.,.,.,.,.\n",
            "s,4,0,2,2,.,.,iv,.,.,.,.,.,.,.\n",
            "s,1,0,3,3,.,.,peripheral,.,.,.,.,.,.,.\n"
        ),
    );
    let rows = records(
        &Data::from_pmetrics_csv_bytes(input.as_bytes())
            .unwrap()
            .to_pmetrics_csv_bytes()
            .unwrap(),
    );
    assert_eq!(rows[1].get(1), Some("4"));
    assert_eq!(rows[1].get(3), Some("2"));
    assert_eq!(rows[1].get(7), Some("iv"));
    assert_eq!(rows[2].get(1), Some("1"));
    assert_eq!(rows[2].get(3), Some("3"));
    assert_eq!(rows[2].get(7), Some("peripheral"));
}

#[test]
fn later_occasion_without_an_initial_dose_is_not_exportable() {
    let data = Data::new(vec![Subject::builder("s")
        .bolus(0.0, 1.0, "iv")
        .reset()
        .observation(0.0, 3.0, "cp")
        .build()]);
    assert!(matches!(
        data.to_pmetrics_csv_bytes(),
        Err(DataError::UnrepresentablePmetricsData(message))
            if message.contains("later occasions must begin with a dose")
    ));
}

#[test]
fn observation_at_reset_time_makes_later_occasion_unsafe_to_export() {
    let input = pmetrics_input(
        &[],
        concat!(
            "s,1,0,0,1,.,.,iv,.,.,.,.,.,.,.\n",
            "s,4,0,0,2,.,.,iv,.,.,.,.,.,.,.\n",
            "s,0,0,.,.,.,.,.,3,cp,0,.,.,.,.\n"
        ),
    );
    let data = Data::from_pmetrics_csv_bytes(input.as_bytes()).unwrap();
    assert!(matches!(
        data.to_pmetrics_csv_bytes(),
        Err(DataError::UnrepresentablePmetricsData(message))
            if message.contains("later occasions must begin with a dose")
    ));
}

#[test]
fn negative_addl_reset_is_accepted_and_resets_at_earliest_expanded_dose() {
    let input = pmetrics_input(
        &[],
        concat!(
            "s,1,0,0,1,.,.,iv,.,.,.,.,.,.,.\n",
            "s,4,0,0,2,-2,1,iv,.,.,.,.,.,.,.\n"
        ),
    );
    let data = Data::from_pmetrics_csv_bytes(input.as_bytes()).unwrap();
    let occasion = &data.subjects()[0].occasions()[1];
    assert_eq!(
        occasion
            .events()
            .iter()
            .map(|event| event.time())
            .collect::<Vec<_>>(),
        [-2.0, -1.0, 0.0]
    );
    assert!(occasion.events().iter().all(|event| event.occasion() == 1));

    let bytes = data.to_pmetrics_csv_bytes().unwrap();
    let rows = records(&bytes);
    assert_eq!(rows[1].get(1), Some("4"));
    assert_eq!(rows[1].get(2), Some("-2"));
    assert_eq!(rows[2].get(1), Some("1"));
    assert_eq!(rows[2].get(2), Some("-1"));
    assert_eq!(rows[3].get(1), Some("1"));
    assert_eq!(rows[3].get(2), Some("0"));

    let reparsed = Data::from_pmetrics_csv_bytes(&bytes).unwrap();
    assert_data_equivalent(&data, &reparsed);
}

#[test]
fn negative_addl_reset_at_later_time_uses_the_first_expanded_dose() {
    let input = pmetrics_input(
        &[],
        concat!(
            "s,1,0,0,1,.,.,iv,.,.,.,.,.,.,.\n",
            "s,4,2,0,2,-2,1,iv,.,.,.,.,.,.,.\n"
        ),
    );
    let data = Data::from_pmetrics_csv_bytes(input.as_bytes()).unwrap();
    let occasion = &data.subjects()[0].occasions()[1];
    assert_eq!(
        occasion
            .events()
            .iter()
            .map(|event| event.time())
            .collect::<Vec<_>>(),
        [0.0, 1.0, 2.0]
    );

    let bytes = data.to_pmetrics_csv_bytes().unwrap();
    let rows = records(&bytes);
    assert_eq!(rows[1].get(1), Some("4"));
    assert_eq!(rows[1].get(2), Some("0"));

    let reparsed = Data::from_pmetrics_csv_bytes(&bytes).unwrap();
    assert_data_equivalent(&data, &reparsed);
}

#[test]
fn nonzero_addl_requires_positive_ii() {
    for ii in [".", "0", "-1"] {
        let input = format!("{}s,1,0,0,1,2,{ii},iv,.,.,.,.,.,.,.\n", core_header());
        assert!(matches!(
            Data::from_pmetrics_csv_bytes(input.as_bytes()),
            Err(DataError::InvalidDataRow(message))
                if message.contains("requires a positive II")
        ));
    }
}

#[test]
fn minimum_addl_fails_without_panicking() {
    let input = format!(
        "{}s,1,0,0,1,{},1,iv,.,.,.,.,.,.,.\n",
        core_header(),
        i64::MIN
    );
    assert!(matches!(
        Data::from_pmetrics_csv_bytes(input.as_bytes()),
        Err(DataError::InvalidDataRow(message))
            if message.contains("too large to expand")
    ));
}

#[test]
fn additional_dose_time_overflow_is_rejected() {
    let input = format!("{}s,1,0,0,1,2,1e308,iv,.,.,.,.,.,.,.\n", core_header());
    assert!(matches!(
        Data::from_pmetrics_csv_bytes(input.as_bytes()),
        Err(DataError::NonFiniteValue { field, .. }) if field == "expanded TIME"
    ));
}

#[test]
fn positive_addl_reset_round_trips_as_individual_doses() {
    let input = pmetrics_input(
        &[],
        concat!(
            "s,1,0,0,1,.,.,iv,.,.,.,.,.,.,.\n",
            "s,4,0,0,2,2,1,iv,.,.,.,.,.,.,.\n"
        ),
    );
    let data = Data::from_pmetrics_csv_bytes(input.as_bytes()).unwrap();
    let bytes = data.to_pmetrics_csv_bytes().unwrap();
    let rows = records(&bytes);
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[1].get(1), Some("4"));
    assert_eq!(rows[1].get(2), Some("0"));
    assert_eq!(rows[2].get(1), Some("1"));
    assert_eq!(rows[2].get(2), Some("1"));
    assert_eq!(rows[3].get(1), Some("1"));
    assert_eq!(rows[3].get(2), Some("2"));
    assert!(rows[1..].iter().all(|row| row.get(5) == Some(".")));

    let reparsed = Data::from_pmetrics_csv_bytes(&bytes).unwrap();
    assert_data_equivalent(&data, &reparsed);
}

#[test]
fn identical_covariate_values_at_one_time_are_deduplicated() {
    let input = pmetrics_input(
        &["wt"],
        concat!(
            "s,1,0,0,1,.,.,iv,.,.,.,.,.,.,.,70\n",
            "s,0,0,.,.,.,.,.,1,cp,0,.,.,.,.,70\n"
        ),
    );
    let data = Data::from_pmetrics_csv_bytes(input.as_bytes()).unwrap();
    assert_eq!(
        data.subjects()[0].occasions()[0]
            .covariates()
            .get_covariate("wt")
            .unwrap()
            .observations(),
        [(0.0, 70.0)]
    );
}

#[test]
fn conflicting_covariate_values_at_one_time_are_rejected() {
    let input = pmetrics_input(
        &["wt"],
        concat!(
            "s,1,0,0,1,.,.,iv,.,.,.,.,.,.,.,70\n",
            "s,0,0,.,.,.,.,.,1,cp,0,.,.,.,.,71\n"
        ),
    );
    assert!(matches!(
        Data::from_pmetrics_csv_bytes(input.as_bytes()),
        Err(DataError::InvalidDataRow(message))
            if message.contains("conflicting covariate `wt` values")
                && message.contains("subject `s` occasion 0")
                && message.contains("time 0")
    ));
}

#[test]
fn covariate_values_at_different_times_still_interpolate() {
    let input = pmetrics_input(
        &["wt"],
        concat!(
            "s,1,0,0,1,.,.,iv,.,.,.,.,.,.,.,70\n",
            "s,0,24,.,.,.,.,.,1,cp,0,.,.,.,.,72\n"
        ),
    );
    let data = Data::from_pmetrics_csv_bytes(input.as_bytes()).unwrap();
    let weight = data.subjects()[0].occasions()[0]
        .covariates()
        .get_covariate("wt")
        .unwrap();
    assert_eq!(weight.observations(), [(0.0, 70.0), (24.0, 72.0)]);
    assert_eq!(weight.interpolate(12.0).unwrap(), 71.0);
}

#[test]
fn missing_observations_use_minus_99() {
    let data = Data::new(vec![Subject::builder("s")
        .missing_observation(0.0, "cp")
        .build()]);
    let bytes = data.to_pmetrics_csv_bytes().unwrap();
    let rows = records(&bytes);
    assert_eq!(rows[0].get(8), Some("-99"));

    let parsed = Data::from_pmetrics_csv_bytes(&bytes).unwrap();
    match &parsed.subjects()[0].occasions()[0].events()[0] {
        Event::Observation(observation) => assert_eq!(observation.value(), None),
        _ => panic!("expected observation"),
    }

    let placeholders = format!(
        "{}s,0,0,.,.,.,.,.,.,cp,0,.,.,.,.\ns,0,1,.,.,.,.,.,NA,cp,0,.,.,.,.\ns,0,2,.,.,.,.,.,,cp,0,.,.,.,.\n",
        core_header()
    );
    assert!(Data::from_pmetrics_csv_bytes(placeholders.as_bytes()).is_ok());
}

#[test]
fn actual_minus_99_observation_is_not_exportable() {
    let data = Data::new(vec![Subject::builder("s")
        .observation(0.0, -99.0, "cp")
        .build()]);
    assert!(matches!(
        data.to_pmetrics_csv_bytes(),
        Err(DataError::UnrepresentablePmetricsData(message))
            if message.contains("reserved for missing")
    ));
}

#[test]
fn evid_2_is_neither_written_nor_read() {
    let bytes = fixture_data().to_pmetrics_csv_bytes().unwrap();
    assert!(records(&bytes).iter().all(|row| row.get(1) != Some("2")));

    let unsupported = pmetrics_input(&["wt"], "s,2,0,.,.,.,.,.,.,.,.,.,.,.,.,70\n");
    assert!(matches!(
        Data::from_pmetrics_csv_bytes(unsupported.as_bytes()),
        Err(DataError::UnknownEvid { evid: 2, ref id, .. }) if id == "s"
    ));
}

#[test]
fn empty_reset_rows_are_rejected() {
    let empty_reset = format!("{}s,4,0,.,.,.,.,.,.,.,.,.,.,.,.\n", core_header());
    assert!(matches!(
        Data::from_pmetrics_csv_bytes(empty_reset.as_bytes()),
        Err(DataError::InvalidDataRow(message)) if message.contains("must contain a dose")
    ));
}

#[test]
fn programmatic_covariate_name_is_lowercased_for_export() {
    let data = Data::new(vec![Subject::builder("s")
        .bolus(0.0, 1.0, "iv")
        .covariate("WT", 0.0, 70.0)
        .build()]);
    let bytes = data.to_pmetrics_csv_bytes().unwrap();
    let mut reader = ::csv::Reader::from_reader(bytes.as_slice());
    assert_eq!(reader.headers().unwrap().iter().last(), Some("wt"));

    let parsed = Data::from_pmetrics_csv_bytes(&bytes).unwrap();
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

    let bytes = Data::new(vec![subject]).to_pmetrics_csv_bytes().unwrap();
    let mut reader = ::csv::Reader::from_reader(bytes.as_slice());
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
        data.to_pmetrics_csv_bytes(),
        Err(DataError::UnrepresentablePmetricsData(message))
            if message.contains("both map to `wt`")
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
    let bytes = data.to_pmetrics_csv_bytes().unwrap();
    let mut reader = ::csv::Reader::from_reader(bytes.as_slice());
    assert_eq!(
        reader
            .headers()
            .unwrap()
            .iter()
            .filter(|header| *header == "wt")
            .count(),
        1
    );

    let parsed = Data::from_pmetrics_csv_bytes(&bytes).unwrap();
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
        Data::new(vec![subject]).to_pmetrics_csv_bytes(),
        Err(DataError::UnrepresentablePmetricsData(message))
            if message.contains("inconsistent fixed settings")
    ));
}

#[test]
fn subject_id_starting_with_comment_marker_is_rejected() {
    let data = Data::new(vec![Subject::builder("#subject")
        .bolus(0.0, 1.0, "iv")
        .build()]);
    assert!(matches!(
        data.to_pmetrics_csv_bytes(),
        Err(DataError::UnrepresentablePmetricsData(message))
            if message.contains("cannot start with #")
    ));
}

#[test]
fn empty_subject_ids_are_rejected_on_import_and_export() {
    let input = format!("{},0,0,.,.,.,.,.,1,cp,0,.,.,.,.\n", core_header());
    assert!(matches!(
        Data::from_pmetrics_csv_bytes(input.as_bytes()),
        Err(DataError::InvalidDataRow(message)) if message.contains("subject ID cannot be empty")
    ));

    let data = Data::new(vec![Subject::builder("").bolus(0.0, 1.0, "iv").build()]);
    assert!(matches!(
        data.to_pmetrics_csv_bytes(),
        Err(DataError::UnrepresentablePmetricsData(message))
            if message.contains("subject ID cannot be empty")
    ));
}

#[test]
fn unicode_covariate_names_round_trip_with_one_normalization() {
    let data = Data::new(vec![Subject::builder("s")
        .bolus(0.0, 1.0, "iv")
        .covariate("Ä", 0.0, 70.0)
        .build()]);
    let bytes = data.to_pmetrics_csv_bytes().unwrap();
    let parsed = Data::from_pmetrics_csv_bytes(&bytes).unwrap();
    assert!(parsed.subjects()[0].occasions()[0]
        .covariates()
        .get_covariate("ä")
        .is_some());
}

#[test]
fn unicode_covariate_name_collisions_are_rejected() {
    let data = Data::new(vec![Subject::builder("s")
        .bolus(0.0, 1.0, "iv")
        .covariate("Ä", 0.0, 70.0)
        .covariate("ä", 0.0, 71.0)
        .build()]);
    assert!(matches!(
        data.to_pmetrics_csv_bytes(),
        Err(DataError::UnrepresentablePmetricsData(message))
            if message.contains("both map to `ä`")
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
    let first = data.to_pmetrics_csv_bytes().unwrap();
    let parsed = Data::from_pmetrics_csv_bytes(&first).unwrap();
    assert_eq!(parsed.to_pmetrics_csv_bytes().unwrap(), first);
    assert_data_equivalent(&data, &parsed);
}

#[test]
fn malformed_and_nonfinite_values_fail_cleanly() {
    let partial_error = format!("{}s,0,0,.,.,.,.,.,1,cp,0,0.1,.,.,.\n", core_header());
    assert!(matches!(
        Data::from_pmetrics_csv_bytes(partial_error.as_bytes()),
        Err(DataError::InvalidDataRow(_))
    ));

    let nonfinite = format!("{}s,1,NaN,0,1,.,.,iv,.,.,.,.,.,.,.\n", core_header());
    assert!(matches!(
        Data::from_pmetrics_csv_bytes(nonfinite.as_bytes()),
        Err(DataError::NonFiniteValue { .. })
    ));

    let invalid_data = Data::new(vec![Subject::builder("s")
        .bolus(f64::INFINITY, 1.0, "iv")
        .build()]);
    assert!(matches!(
        invalid_data.to_pmetrics_csv_bytes(),
        Err(DataError::NonFiniteValue { .. })
    ));
}
