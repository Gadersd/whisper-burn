use std::cmp::Ordering;

#[derive(Clone)]
pub struct BeamNode<T: Clone> {
    pub seq: Vec<T>,
    pub log_prob: f64,
}

pub fn beam_search<T, F, G>(
    initial_beams: Vec<BeamNode<T>>,
    mut next: F,
    is_finished: G, 
    beam_size: usize,
    max_depth: usize,
) -> Vec<T>
where
    T: Clone,
    F: FnMut(&[BeamNode<T>]) -> Vec<Vec<(T, f64)>> + Clone,
    G: Fn(&[T]) -> bool + Clone
{
    let is_finished_cloned = is_finished.clone();

    let mut beam_search_step = move |beams: Vec<BeamNode<T>>,
    beam_size: usize| {
        let mut finished_beams: Vec<BeamNode<T>> = Vec::with_capacity(beam_size);
        let mut new_beams: Vec<BeamNode<T>> = Vec::with_capacity(beam_size);
    
        let continuations = next(&beams);
    
        for (beam_node, continuations) in beams.into_iter().zip(continuations) {
            if is_finished(&beam_node.seq) {
                //finished_beams.push(beam_node);
                new_beams.push(beam_node);
            } else {
                let top_new_beams = get_top_elements(&continuations, |(_, log_prob)| *log_prob, beam_size)
                    .into_iter()
                    .map(move |(tok, log_prob)| {
                        BeamNode {
                            seq: [beam_node.seq.clone(), vec![tok.clone()]].concat(),
                            log_prob: *log_prob,
                        }
                    });
    
                new_beams.extend(top_new_beams);
            }
        }
    
        get_top_elements(&new_beams, |beam| beam.log_prob, beam_size)
            .into_iter()
            /*.chain(
                get_top_elements(&finished_beams, |beam| beam.log_prob, beam_size)
            )*/
            .cloned()
            .collect()
    };

    let mut beams = initial_beams;
    for i in 0..max_depth {
        if let Some(beam) = beams.iter().max_by(|a, b| a.log_prob.partial_cmp(&b.log_prob).unwrap()) {
            if is_finished_cloned(&beam.seq) {
                break;
            }
        }
        
        //beams = beam_search_step(beams, next.clone(), is_finished.clone(), beam_size);
        beams = beam_search_step(beams, beam_size);
        println!("Depth: {}", i);
    }

    beams.into_iter()
        .max_by(|a, b| a.log_prob.partial_cmp(&b.log_prob).unwrap())
        .map(|x| x.seq)
        .unwrap_or_else(Vec::new)
}

pub fn beam_search_step<T, F, G>(
    beams: Vec<BeamNode<T>>,
    mut next: F,
    is_finished: G, 
    beam_size: usize,
) -> Vec<BeamNode<T>>
where
    T: Clone,
    F: FnMut(&[BeamNode<T>]) -> Vec<Vec<(T, f64)>>,
    G: Fn(&[T]) -> bool,
{
    let mut finished_beams: Vec<BeamNode<T>> = Vec::with_capacity(beam_size);
    let mut new_beams: Vec<BeamNode<T>> = Vec::with_capacity(beam_size);

    let continuations = next(&beams);

    for (beam_node, continuations) in beams.into_iter().zip(continuations) {
        if is_finished(&beam_node.seq) {
            //finished_beams.push(beam_node);
            new_beams.push(beam_node);
        } else {
            let top_new_beams = get_top_elements(&continuations, |(_, log_prob)| *log_prob, beam_size)
                .into_iter()
                .map(move |(tok, log_prob)| {
                    BeamNode {
                        seq: [beam_node.seq.clone(), vec![tok.clone()]].concat(),
                        log_prob: *log_prob,
                    }
                });

            new_beams.extend(top_new_beams);
        }
    }

    get_top_elements(&new_beams, |beam| beam.log_prob, beam_size)
        .into_iter()
        /*.chain(
            get_top_elements(&finished_beams, |beam| beam.log_prob, beam_size)
        )*/
        .cloned()
        .collect()
}

pub fn get_top_elements<T>(elems: &[T], score: impl Fn(&T) -> f64, num: usize) -> Vec<&T> {
    let mut top_elems = Vec::with_capacity(num);
    let mut scores = Vec::with_capacity(num);

    for elem in elems {
        let score = score(elem);

        // most common scenario
        if top_elems.len() == num {
            if score < scores[0] {
                continue;
            }
        }

        if let Some( (idx, _) ) = scores.iter().enumerate().find(|(_, &s)| s >= score) {
            top_elems.insert(idx, elem);
            scores.insert(idx, score);
        } else {
            top_elems.push(elem);
            scores.push(score);
        }

        if top_elems.len() > num {
            top_elems.remove(0);
            scores.remove(0);
        }
    }

    top_elems
}