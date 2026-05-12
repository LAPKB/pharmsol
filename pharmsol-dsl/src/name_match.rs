pub(crate) fn is_high_confidence_match(
    needle: &str,
    candidate: &str,
    distance: usize,
    prefix: usize,
) -> bool {
    let max_len = needle.len().max(candidate.len());
    let max_distance = match max_len {
        0..=4 => 1,
        5..=8 => 2,
        _ => 3,
    };
    distance <= max_distance && (prefix > 0 || distance <= 1)
}

pub(crate) fn common_prefix_len(lhs: &str, rhs: &str) -> usize {
    lhs.chars()
        .zip(rhs.chars())
        .take_while(|(lhs, rhs)| lhs == rhs)
        .count()
}

pub(crate) fn is_single_adjacent_transposition(lhs: &str, rhs: &str) -> bool {
    let lhs: Vec<char> = lhs.chars().collect();
    let rhs: Vec<char> = rhs.chars().collect();
    if lhs.len() != rhs.len() {
        return false;
    }

    let differing = lhs
        .iter()
        .zip(rhs.iter())
        .enumerate()
        .filter_map(|(index, (lhs, rhs))| (lhs != rhs).then_some(index))
        .collect::<Vec<_>>();

    if differing.len() != 2 || differing[1] != differing[0] + 1 {
        return false;
    }

    let first = differing[0];
    lhs[first] == rhs[first + 1] && lhs[first + 1] == rhs[first]
}

pub(crate) fn edit_distance(lhs: &str, rhs: &str) -> usize {
    let lhs: Vec<char> = lhs.chars().collect();
    let rhs: Vec<char> = rhs.chars().collect();
    if lhs.is_empty() {
        return rhs.len();
    }
    if rhs.is_empty() {
        return lhs.len();
    }

    let mut previous: Vec<usize> = (0..=rhs.len()).collect();
    let mut current = vec![0; rhs.len() + 1];

    for (lhs_index, lhs_char) in lhs.iter().enumerate() {
        current[0] = lhs_index + 1;
        for (rhs_index, rhs_char) in rhs.iter().enumerate() {
            let substitution_cost = usize::from(lhs_char != rhs_char);
            current[rhs_index + 1] = (current[rhs_index] + 1)
                .min(previous[rhs_index + 1] + 1)
                .min(previous[rhs_index] + substitution_cost);
        }
        previous.clone_from_slice(&current);
    }

    previous[rhs.len()]
}