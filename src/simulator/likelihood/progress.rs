use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub struct ProgressTracker {
    counter: Arc<AtomicUsize>,
    total: usize,
    start_time: Instant,
}

impl ProgressTracker {
    pub fn new(total: usize) -> Self {
        Self {
            counter: Arc::new(AtomicUsize::new(0)),
            total,
            start_time: Instant::now(),
        }
    }

    pub fn inc(&self) {
        let current = self.counter.fetch_add(1, Ordering::Relaxed) + 1;

        // Print progress every 5% or every 1000 iterations
        if current % 1000 == 0 || (current * 20) % self.total == 0 {
            let percent = (current * 100) / self.total;
            let elapsed = self.start_time.elapsed();

            let eta_text = if current > 0 {
                let estimated_total_time =
                    elapsed.as_secs_f64() * (self.total as f64) / (current as f64);
                let remaining_time = estimated_total_time - elapsed.as_secs_f64();

                if remaining_time > 0.0 {
                    format_duration(remaining_time)
                } else {
                    "00:00".to_string()
                }
            } else {
                "calculating...".to_string()
            };

            print!(
                "\rProgress: {}/{} ({}%) ETA: {}",
                current, self.total, percent, eta_text
            );
            std::io::stdout().flush().unwrap();
        }
    }

    pub fn finish(&self) {
        println!("\nSimulation complete!");
    }
}

fn format_duration(seconds: f64) -> String {
    let total_seconds = seconds as u64;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let secs = total_seconds % 60;

    if hours > 0 {
        format!("{:02}:{:02}:{:02}", hours, minutes, secs)
    } else {
        format!("{:02}:{:02}", minutes, secs)
    }
}
