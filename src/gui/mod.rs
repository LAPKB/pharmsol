use eframe::{
    egui::{self, Color32, Response},
    App,
};
use egui_plot::{Legend, Line, Plot, PlotPoints};

use crate::prelude::simulator::SubjectPredictions;
pub trait GUI: Default + App + 'static {
    fn run(self) -> eframe::Result {
        // env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([600.0, 400.0])
                .with_maximize_button(true)
                // .with_fullscreen(true)
                .with_minimize_button(true)
                .with_resizable(true)
                .with_min_inner_size([300.0, 220.0]),
            ..Default::default()
        };
        eframe::run_native("PharmSol", options, Box::new(|_cc| Ok(Box::new(self))))
    }
}

impl SubjectPredictions {
    fn op(&self) -> Line {
        let points = PlotPoints::from_iter(
            self.flat_observations()
                .iter()
                .zip(self.flat_predictions().iter())
                .map(|(obs, pred)| [*obs, *pred]),
        );
        dbg!(&points.points());
        Line::new(points)
            .color(Color32::from_rgb(100, 150, 250))
            .name("OP Plot")
    }

    fn line_demo(&self, ui: &mut egui::Ui) -> Response {
        let mut plot = Plot::new("lines_demo")
            .legend(Legend::default())
            .show_axes(true)
            .show_grid(true);
        // if true {
        //     plot = plot.view_aspect(1.0);
        // }
        // if true {
        //     plot = plot.data_aspect(1.0);
        // }

        plot.show(ui, |plot_ui| {
            plot_ui.line(self.op());
        })
        .response
    }
}

impl App for SubjectPredictions {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:

            egui::menu::bar(ui, |ui| {
                // NOTE: no File->Quit on web pages!
                let is_web = cfg!(target_arch = "wasm32");
                if !is_web {
                    ui.menu_button("File", |ui| {
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    ui.add_space(16.0);
                }

                egui::widgets::global_dark_light_mode_buttons(ui);
            });
        });
        // egui::SidePanel::left("left_panel")
        //     .resizable(false)
        //     // .default_width(100.0)
        //     .show(ctx, |ui| {
        //         ui.heading("Side Panel");
        //     });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.collapsing("Instructions", |ui| {
                    ui.label("Pan by dragging, or scroll (+ shift = horizontal).");
                    ui.label("Box zooming: Right click to zoom in and zoom out using a selection.");
                    if cfg!(target_arch = "wasm32") {
                        ui.label("Zoom with ctrl / ⌘ + pointer wheel, or with pinch gesture.");
                    } else if cfg!(target_os = "macos") {
                        ui.label("Zoom with ctrl / ⌘ + scroll.");
                    } else {
                        ui.label("Zoom with ctrl + scroll.");
                    }
                    ui.label("Reset view with double-click.");
                });
            });
            ui.separator();
            self.line_demo(ui);
            //THIS CODE DO THE SAME AS line_demo
            // The central panel the region left after adding TopPanel's and SidePanel's
            // let plot = Plot::new("op_plot")
            //     .legend(Legend::default())
            //     .show_axes(true)
            //     .show_grid(true);

            // let res = plot
            //     .show(ui, |plot_ui| {
            //         let points: PlotPoints = self
            //             .flat_observations()
            //             .iter()
            //             .zip(self.flat_predictions())
            //             .map(|(obs, pred)| [*obs, pred])
            //             .collect();
            //         plot_ui.line(Line::new(points).name("OP Plot").color(egui::Color32::RED));
            //     })
            //     .response;
            // dbg!(res)
        });
    }
}

impl GUI for SubjectPredictions {}
