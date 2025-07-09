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
                let estimated_total_duration = elapsed.mul_f64(self.total as f64 / current as f64);

                if let Some(remaining_duration) = estimated_total_duration.checked_sub(elapsed) {
                    format_duration(remaining_duration)
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
            std::io::stdout().flush().unwrap_or_default();
        }
    }

    pub fn finish(&self) {
        println!("\nSimulation complete!");
    }
}

fn format_duration(duration: std::time::Duration) -> String {
    let total_seconds = duration.as_secs();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let secs = total_seconds % 60;

    if hours > 0 {
        format!("{:02}h:{:02}m:{:02}s", hours, minutes, secs)
    } else {
        format!("{:02}m:{:02}s", minutes, secs)
    }
}
