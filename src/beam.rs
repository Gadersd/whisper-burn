use std::cmp::Ordering;

pub struct BeamNode<T> {
    pub seq: Vec<T>,
    pub score: f64,
}

pub fn beam_search<T, F, G>(
    initial_beams: Vec<BeamNode<T>>,
    next: F,
    is_finished: G, 
    beam_size: usize,
    max_depth: usize,
) -> Vec<T>
where
    T: Clone,
    F: Fn(&[BeamNode<T>]) -> Vec<Vec<(T, f64)>> + Clone,
    G: Fn(&[T]) -> bool + Clone,
{
    /*for _ in 0..depth {
        beam = beam.into_iter().flat_map(|beam_node| {
            let mut continuations = next(&beam_node.seq);
            continuations.sort_unstable_by(|(tok1, score1), (tok2, score2)| score2.partial_cmp(&score1).unwrap());

            continuations
                .into_iter()
                .map(move |(tok, score)| BeamNode {
                    seq: [beam_node.seq.clone(), vec![tok]].concat(),
                    score: score,
                })
                .take(beam_size)
        }).collect();
    }*/

    let final_beams = (0..max_depth).into_iter().fold(initial_beams, |beam, _| beam_search_step(beam, next.clone(), is_finished.clone(), beam_size));

    final_beams.into_iter()
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
        .map(|x| x.seq)
        .unwrap_or_else(Vec::new)
}

pub fn beam_search_step<T, F, G>(
    beams: Vec<BeamNode<T>>,
    next: F,
    is_finished: G, 
    beam_size: usize,
) -> Vec<BeamNode<T>>
where
    T: Clone,
    F: Fn(&[BeamNode<T>]) -> Vec<Vec<(T, f64)>>,
    G: Fn(&[T]) -> bool,
{
    let mut new_beams = Vec::with_capacity(beams.len());
    let continuations = next(&beams);

    for (beam_node, continuations) in beams.into_iter().zip(continuations) {
        if is_finished(&beam_node.seq) {
            new_beams.push(beam_node);
        } else {
            let mut sorted = continuations;
            sorted.sort_unstable_by(|(tok1, score1), (tok2, score2)| score1.partial_cmp(&score2).unwrap());

            new_beams.extend(
                sorted
                    .into_iter()
                    .rev()
                    .map(move |(tok, score)| BeamNode {
                        seq: [beam_node.seq.clone(), vec![tok]].concat(),
                        score: score,
                    })
                    .take(beam_size)
            )
        }
    }

    // keep only the top beams
    new_beams.sort_unstable_by(|beam_node1, beam_node2| beam_node1.score.partial_cmp(&beam_node2.score).unwrap());

    new_beams.into_iter().rev().take(beam_size).collect()
}
