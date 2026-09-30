//! Line icons drawn on a 24 by 24 grid and tinted by the theme.

use std::sync::OnceLock;

use iced::widget::svg::{self, Handle};
use iced::widget::{Svg, svg as svg_widget};
use iced::{Color, Theme};

use super::theme::{Tokens, tokens};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Icon {
    Plus,
    Close,
    ChevronRight,
    ChevronDown,
    ArrowRight,
    ArrowLeft,
    Search,
    Folder,
    FolderOpen,
    Calendar,
    Expand,
    Shrink,
    More,
    Copy,
    Trash,
    Pencil,
    Stop,
    Image,
    Panel,
    Graph,
    Gear,
    Check,
    Download,
    Eye,
    Refresh,
    External,
    File,
    Key,
    Undo,
    Minimize,
    Maximize,
    Restore,
}

const ALL: [Icon; 32] = [
    Icon::Plus, Icon::Close, Icon::ChevronRight, Icon::ChevronDown, Icon::ArrowRight, Icon::ArrowLeft, Icon::Search,
    Icon::Folder, Icon::FolderOpen, Icon::Calendar, Icon::Expand, Icon::Shrink, Icon::More, Icon::Copy, Icon::Trash,
    Icon::Pencil, Icon::Stop, Icon::Image, Icon::Panel, Icon::Graph, Icon::Gear, Icon::Check, Icon::Download, Icon::Eye,
    Icon::Refresh, Icon::External, Icon::File, Icon::Key, Icon::Undo,
    Icon::Minimize, Icon::Maximize, Icon::Restore,
];

fn body(icon: Icon) -> String {
    let s = match icon {
        Icon::Plus => r#"<path d="M12 5v14M5 12h14"/>"#,
        Icon::Close => r#"<path d="M18 6 6 18M6 6l12 12"/>"#,
        Icon::ChevronRight => r#"<path d="m9 6 6 6-6 6"/>"#,
        Icon::ChevronDown => r#"<path d="m6 9 6 6 6-6"/>"#,
        Icon::ArrowRight => r#"<path d="M5 12h14M13 6l6 6-6 6"/>"#,
        Icon::ArrowLeft => r#"<path d="M19 12H5M11 18l-6-6 6-6"/>"#,
        Icon::Search => r#"<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2"/>"#,
        Icon::Folder => r#"<path d="M3.5 7.5a2 2 0 0 1 2-2h3.8l2 2h7.2a2 2 0 0 1 2 2v7.5a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>"#,
        Icon::FolderOpen => r#"<path d="M3.5 17V7.5a2 2 0 0 1 2-2h3.8l2 2h6.2a2 2 0 0 1 2 2V11"/><path d="M3.5 17l2.3-5.2a1.5 1.5 0 0 1 1.4-.8H20a1 1 0 0 1 .9 1.4L18.6 18a1.5 1.5 0 0 1-1.4 1H5.5a2 2 0 0 1-2-2z"/>"#,
        Icon::Calendar => r#"<rect x="3.5" y="5" width="17" height="15.5" rx="2"/><path d="M3.5 10h17M8 3v4M16 3v4"/>"#,
        Icon::Expand => r#"<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>"#,
        Icon::Shrink => r#"<path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5"/>"#,
        Icon::More => r#"<circle cx="5" cy="12" r="1.2" fill="black"/><circle cx="12" cy="12" r="1.2" fill="black"/><circle cx="19" cy="12" r="1.2" fill="black"/>"#,
        Icon::Copy => r#"<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h8"/>"#,
        Icon::Trash => r#"<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>"#,
        Icon::Pencil => r#"<path d="M4 20h4L19 9a2.8 2.8 0 0 0-4-4L4 16z"/>"#,
        Icon::Stop => r#"<rect x="7" y="7" width="10" height="10" rx="1.5" fill="black"/>"#,
        Icon::Image => r#"<rect x="3.5" y="4.5" width="17" height="15" rx="2"/><circle cx="9" cy="10" r="1.8"/><path d="m20.5 16-5-5-9.5 8.5"/>"#,
        Icon::Panel => r#"<rect x="3.5" y="4.5" width="17" height="15" rx="2"/><path d="M14.5 4.5v15"/>"#,
        Icon::Graph => r#"<circle cx="6" cy="7" r="2.3"/><circle cx="18" cy="8" r="2.3"/><circle cx="11" cy="18" r="2.3"/><path d="M8.2 7.4l7.5.4M7.2 9l2.7 6.9M16.5 9.9l-4 6.2"/>"#,
        Icon::Gear => return gear(),
        Icon::Check => r#"<path d="m5 12.5 4.5 4.5L19 7.5"/>"#,
        Icon::Download => r#"<path d="M12 4v11M7 10.5l5 5 5-5M5 20h14"/>"#,
        Icon::Eye => r#"<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3"/>"#,
        Icon::Refresh => r#"<path d="M20 11.5A8 8 0 1 0 17.7 17M20 4.5v7h-7"/>"#,
        Icon::External => r#"<path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>"#,
        Icon::File => r#"<path d="M14 3.5H7a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8.5z"/><path d="M14 3.5v5h5"/>"#,
        Icon::Key => r#"<circle cx="8" cy="15" r="4"/><path d="m11 12 8.5-8.5M16.5 6.5l2.5 2.5M14.5 8.5l2 2"/>"#,
        Icon::Undo => r#"<path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>"#,
        Icon::Minimize => r#"<path d="M6 12h12"/>"#,
        Icon::Maximize => r#"<rect x="6" y="6" width="12" height="12" rx="1.5"/>"#,
        Icon::Restore => r#"<rect x="5" y="8.5" width="10.5" height="10.5" rx="1.5"/><path d="M8.5 5.5H17a1.5 1.5 0 0 1 1.5 1.5v8.5"/>"#,
    };
    s.to_string()
}

/// An eight-tooth gear outline with a center hole, generated so the teeth are even.
fn gear() -> String {
    let (teeth, outer, inner) = (8, 10.0_f32, 7.4_f32);
    let step = std::f32::consts::TAU / teeth as f32;
    let (tip, base) = (step * 0.17, step * 0.27);
    let pt = |r: f32, a: f32| format!("{:.2} {:.2}", 12.0 + r * a.cos(), 12.0 + r * a.sin());
    let mut d = String::new();
    for i in 0..teeth {
        let a = i as f32 * step - std::f32::consts::FRAC_PI_2;
        let cmd = if i == 0 { "M" } else { "L" };
        d.push_str(&format!("{cmd}{} L{} L{} L{} ", pt(inner, a - base), pt(outer, a - tip), pt(outer, a + tip), pt(inner, a + base)));
        let next = (i + 1) as f32 * step - std::f32::consts::FRAC_PI_2 - base;
        d.push_str(&format!("A{inner} {inner} 0 0 1 {} ", pt(inner, next)));
    }
    d.push('Z');
    format!(r#"<path d="{d}"/><circle cx="12" cy="12" r="3"/>"#)
}

fn handles() -> &'static Vec<(Icon, Handle)> {
    static HANDLES: OnceLock<Vec<(Icon, Handle)>> = OnceLock::new();
    HANDLES.get_or_init(|| {
        ALL.iter()
            .map(|&i| {
                let doc = format!(
                    r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{}</svg>"#,
                    body(i)
                );
                (i, Handle::from_memory(doc.into_bytes()))
            })
            .collect()
    })
}

pub fn handle(icon: Icon) -> Handle {
    handles().iter().find(|(i, _)| *i == icon).map(|(_, h)| h.clone()).expect("every icon has a handle")
}

/// An icon tinted with the muted text color, brightening on hover.
pub fn icon<'a>(icon: Icon, size: f32) -> Svg<'a, Theme> {
    svg_widget(handle(icon)).width(size).height(size).style(|theme: &Theme, status| {
        let t = tokens(theme);
        svg::Style { color: Some(if status == svg::Status::Hovered { t.text } else { t.muted }) }
    })
}

pub fn tinted<'a>(i: Icon, size: f32, color: fn(&Tokens) -> Color) -> Svg<'a, Theme> {
    svg_widget(handle(i)).width(size).height(size).style(move |theme: &Theme, _| svg::Style { color: Some(color(tokens(theme))) })
}
