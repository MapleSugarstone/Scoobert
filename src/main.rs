#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() -> iced::Result {
    scoobert::sys::use_login_path();
    scoobert::ui::run()
}
