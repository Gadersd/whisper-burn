use std::cmp::Ordering;

pub struct BeamNode<T> {
    pub seq: Vec<T>,
    pub log_prob: f64,
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
    F: Fn(&[BeamNode<T>]) -> Vec<(Vec<(T, f64)>, usize)> + Clone,
    G: Fn(&[T]) -> bool + Clone
{
    /*for _ in 0..depth {
        beam = beam.into_iter().flat_map(|beam_node| {
            let mut continuations = next(&beam_node.seq);
            continuations.sort_unstable_by(|(tok1, log_prob1), (tok2, log_prob2)| log_prob2.partial_cmp(&log_prob1).unwrap());

            continuations
                .into_iter()
                .map(move |(tok, log_prob)| BeamNode {
                    seq: [beam_node.seq.clone(), vec![tok]].concat(),
                    log_prob: log_prob,
                })
                .take(beam_size)
        }).collect();
    }*/

    let mut beams = initial_beams;
    for i in 0..max_depth {
        if let Some(beam) = beams.iter().max_by(|a, b| a.log_prob.partial_cmp(&b.log_prob).unwrap()) {
            if is_finished(&beam.seq) {
                break;
            }
        }
        
        beams = beam_search_step(beams, next.clone(), is_finished.clone(), beam_size);
        println!("Depth: {}", i);
    }

    //let beams = (0..max_depth).into_iter().fold(initial_beams, |beam, d| {println!("Depth {}", d); beam_search_step(beam, next.clone(), is_finished.clone(), beam_size)});

    beams.into_iter()
        .max_by(|a, b| a.log_prob.partial_cmp(&b.log_prob).unwrap())
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
    F: Fn(&[BeamNode<T>]) -> Vec<(Vec<(T, f64)>, usize)>,
    G: Fn(&[T]) -> bool,
{
    let mut finished_beams = Vec::new();
    let mut new_beams = Vec::with_capacity(beams.len());
    let continuations = next(&beams);

    for (beam_node, (continuations, end_node_index)) in beams.into_iter().zip(continuations) {
        if is_finished(&beam_node.seq) {
            finished_beams.push(beam_node);
        } else {
            let end_node = continuations[end_node_index].clone();

            let mut sorted = continuations;
            sorted.sort_unstable_by(|(tok1, log_prob1), (tok2, log_prob2)| log_prob1.partial_cmp(&log_prob2).unwrap());

            // Create a WeightedIndex distribution
            //let dist = WeightedIndex::new(continuations.into_iter().map(|(_, log_prob)| log_prob)).unwrap();

            
            /*finished_beams.push(BeamNode {
                seq: [beam_node.seq.clone(), vec![end_node.0]].concat(), 
                log_prob: end_node.1, 
            });*/

            new_beams.extend(
                sorted
                    .into_iter()
                    .rev()
                    .map(move |(tok, log_prob)| BeamNode {
                        seq: [beam_node.seq.clone(), vec![tok]].concat(),
                        log_prob: log_prob,
                    })
                    .take(beam_size)
            )
        }
    }

    // keep only the top beams
    new_beams.sort_unstable_by(|beam_node1, beam_node2| beam_node1.log_prob.partial_cmp(&beam_node2.log_prob).unwrap());
    finished_beams.sort_unstable_by(|beam_node1, beam_node2| beam_node1.log_prob.partial_cmp(&beam_node2.log_prob).unwrap());

    new_beams.into_iter().rev().take(beam_size).chain(
        finished_beams.into_iter().rev().take(beam_size)
    ).collect()
}
