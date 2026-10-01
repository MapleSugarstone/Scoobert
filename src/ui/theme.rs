//! Colors and widget styles: neutral grays with one green accent, in light and dark schemes.

use iced::border::{self, Radius};
use iced::theme::{Palette, palette};
use iced::widget::{button, checkbox, container, pick_list, progress_bar, rule, scrollable, text_editor, text_input, toggler};
use iced::{Background, Border, Color, Shadow, Theme, Vector, color};

#[derive(Clone, Copy, Debug)]
pub struct Tokens {
    pub bg: Color,
    pub surface: Color,
    pub surface2: Color,
    pub line: Color,
    pub text: Color,
    pub muted: Color,
    /// Fills buttons and markers.
    pub accent: Color,
    /// The accent where green text or thin lines sit on the background.
    pub accent_ink: Color,
    pub accent_text: Color,
    pub ok: Color,
    pub warn: Color,
    pub danger: Color,
    pub input: Color,
    pub code_bg: Color,
    pub diff_add: Color,
    pub diff_del: Color,
    pub shadow: Color,
}

pub const DARK: Tokens = Tokens {
    bg: color!(0x1e1f22),
    surface: color!(0x141517),
    surface2: color!(0x32343a),
    line: color!(0x494c54),
    text: color!(0xf7f7f9),
    muted: color!(0xbbbdc4),
    // The mascot's screen is pure green (hue 120), so the accent keeps that hue at a softer strength.
    accent: color!(0x2fd22f),
    accent_ink: color!(0x5ae25a),
    accent_text: color!(0x061a06),
    ok: color!(0x4fb84f),
    warn: color!(0xd4a72c),
    danger: color!(0xf47067),
    input: color!(0x2e3035),
    code_bg: color!(0x121315),
    diff_add: Color::from_rgba(0.18, 0.82, 0.18, 0.16),
    diff_del: Color::from_rgba(0.96, 0.44, 0.40, 0.16),
    shadow: Color::from_rgba(0.0, 0.0, 0.0, 0.4),
};

pub const LIGHT: Tokens = Tokens {
    bg: color!(0xfcfcfd),
    surface: color!(0xeeeef1),
    surface2: color!(0xdcdce2),
    line: color!(0xb6b7be),
    text: color!(0x0b0c0e),
    muted: color!(0x3e4047),
    accent: color!(0x157915),
    accent_ink: color!(0x157915),
    accent_text: color!(0xffffff),
    ok: color!(0x2a7f2a),
    warn: color!(0x9a6700),
    danger: color!(0xc93c37),
    input: color!(0xffffff),
    code_bg: color!(0xeeeef0),
    diff_add: Color::from_rgba(0.08, 0.47, 0.08, 0.14),
    diff_del: Color::from_rgba(0.79, 0.24, 0.22, 0.12),
    shadow: Color::from_rgba(0.0, 0.0, 0.0, 0.12),
};

pub const RADIUS: f32 = 6.0;
pub const RADIUS_LG: f32 = 10.0;

pub fn build(dark: bool) -> Theme {
    let t = if dark { DARK } else { LIGHT };
    let p = Palette { background: t.bg, text: t.text, primary: t.accent, success: t.ok, warning: t.warn, danger: t.danger };
    Theme::custom_with_fn(if dark { "Scoobert Dark" } else { "Scoobert Light" }, p, move |p| {
        let mut e = palette::Extended::generate(p);
        e.primary.base = palette::Pair { color: t.accent, text: t.accent_text };
        e.background.weak = palette::Pair { color: t.surface2, text: t.text };
        e.background.strong = palette::Pair { color: t.line, text: t.text };
        e
    })
}

pub fn tokens(theme: &Theme) -> &'static Tokens {
    if theme.extended_palette().is_dark { &DARK } else { &LIGHT }
}

fn line_border(t: &Tokens, radius: f32) -> Border {
    Border { color: t.line, width: 1.0, radius: Radius::new(radius) }
}

// ---- containers ----

pub fn app(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style { background: Some(t.bg.into()), text_color: Some(t.text), ..Default::default() }
}

pub fn sidebar(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style { background: Some(t.surface.into()), text_color: Some(t.text), ..Default::default() }
}

pub fn surface(theme: &Theme) -> container::Style {
    sidebar(theme)
}

pub fn bubble(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style { background: Some(t.surface2.into()), border: line_border(t, RADIUS_LG), text_color: Some(t.text), ..Default::default() }
}

pub fn composer(focused: bool) -> impl Fn(&Theme) -> container::Style {
    move |theme| {
        let t = tokens(theme);
        let border = Border { color: if focused { t.accent_ink } else { t.line }, width: 1.0, radius: Radius::new(RADIUS_LG) };
        container::Style { background: Some(t.input.into()), border, ..Default::default() }
    }
}

pub fn card(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style { background: Some(t.surface.into()), border: line_border(t, RADIUS_LG), text_color: Some(t.text), ..Default::default() }
}

pub fn selected_card(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style {
        background: Some(t.surface.into()),
        border: Border { color: t.accent_ink, width: 1.5, radius: Radius::new(RADIUS_LG) },
        text_color: Some(t.text),
        ..Default::default()
    }
}

pub fn code_block(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style { background: Some(t.code_bg.into()), border: line_border(t, RADIUS), text_color: Some(t.text), ..Default::default() }
}

pub fn diff_line(kind: char) -> impl Fn(&Theme) -> container::Style {
    move |theme| {
        let t = tokens(theme);
        let bg = match kind {
            '+' => Some(t.diff_add.into()),
            '-' => Some(t.diff_del.into()),
            _ => None,
        };
        container::Style { background: bg, ..Default::default() }
    }
}

pub fn accent_bar(theme: &Theme) -> container::Style {
    container::Style { background: Some(tokens(theme).accent.into()), ..Default::default() }
}

pub fn thread_line(theme: &Theme) -> container::Style {
    container::Style { background: Some(tokens(theme).line.into()), ..Default::default() }
}

pub fn dot(color: fn(&Tokens) -> Color) -> impl Fn(&Theme) -> container::Style {
    move |theme| container::Style {
        background: Some(color(tokens(theme)).into()),
        border: border::rounded(99),
        ..Default::default()
    }
}

pub fn backdrop(_: &Theme) -> container::Style {
    container::Style { background: Some(Color::from_rgba(0.0, 0.0, 0.0, 0.5).into()), ..Default::default() }
}

pub fn modal(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style {
        background: Some(t.bg.into()),
        border: line_border(t, RADIUS_LG),
        text_color: Some(t.text),
        shadow: Shadow { color: t.shadow, offset: Vector::new(0.0, 8.0), blur_radius: 24.0 },
        ..Default::default()
    }
}

pub fn toast(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style {
        background: Some(t.surface2.into()),
        border: line_border(t, RADIUS),
        text_color: Some(t.text),
        shadow: Shadow { color: t.shadow, offset: Vector::new(0.0, 4.0), blur_radius: 16.0 },
        ..Default::default()
    }
}

pub fn banner(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style {
        background: Some(t.surface2.into()),
        border: Border { color: t.accent_ink, width: 1.0, radius: Radius::new(RADIUS) },
        text_color: Some(t.text),
        ..Default::default()
    }
}

pub fn error_box(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style {
        background: Some(Color { a: 0.10, ..t.danger }.into()),
        border: Border { color: t.danger, width: 1.0, radius: Radius::new(RADIUS) },
        text_color: Some(t.text),
        ..Default::default()
    }
}

/// Lighter than a selected row and outlined, so a tooltip stands out from the list it covers.
pub fn tooltip(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style {
        background: Some(mix(t.surface2, t.line, 0.6).into()),
        border: Border { color: Color { a: 0.6, ..t.muted }, width: 1.0, radius: Radius::new(RADIUS) },
        text_color: Some(t.text),
        shadow: Shadow { color: t.shadow, offset: Vector::new(0.0, 4.0), blur_radius: 16.0 },
        ..Default::default()
    }
}

/// The sidebar shown over the chat in a narrow window.
pub fn drawer(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style { shadow: Shadow { color: t.shadow, offset: Vector::new(4.0, 0.0), blur_radius: 24.0 }, ..Default::default() }
}

/// A menu that opens under a top bar button.
pub fn popover(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style {
        background: Some(t.surface.into()),
        border: line_border(t, RADIUS),
        text_color: Some(t.text),
        shadow: Shadow { color: t.shadow, offset: Vector::new(0.0, 4.0), blur_radius: 16.0 },
        ..Default::default()
    }
}

/// The project label in the top bar: a filled pill that lightens on hover.
pub fn place(theme: &Theme, status: button::Status) -> button::Style {
    let t = tokens(theme);
    let bg = if matches!(status, button::Status::Hovered | button::Status::Pressed) { t.line } else { t.surface2 };
    button::Style { background: Some(bg.into()), text_color: t.text, border: border::rounded(RADIUS), ..Default::default() }
}

pub fn chip(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style { background: Some(t.surface.into()), border: line_border(t, RADIUS), text_color: Some(t.muted), ..Default::default() }
}

// ---- text ----

pub fn muted(theme: &Theme) -> iced::widget::text::Style {
    iced::widget::text::Style { color: Some(tokens(theme).muted) }
}

pub fn accent_text(theme: &Theme) -> iced::widget::text::Style {
    iced::widget::text::Style { color: Some(tokens(theme).accent_ink) }
}

pub fn danger_text(theme: &Theme) -> iced::widget::text::Style {
    iced::widget::text::Style { color: Some(tokens(theme).danger) }
}

pub fn warn_text(theme: &Theme) -> iced::widget::text::Style {
    iced::widget::text::Style { color: Some(tokens(theme).warn) }
}

// ---- buttons ----

pub fn primary(theme: &Theme, status: button::Status) -> button::Style {
    let t = tokens(theme);
    let bg = match status {
        button::Status::Hovered => mix(t.accent, t.text, 0.12),
        button::Status::Pressed => mix(t.accent, t.bg, 0.15),
        button::Status::Disabled => t.line,
        button::Status::Active => t.accent,
    };
    button::Style {
        background: Some(bg.into()),
        text_color: if status == button::Status::Disabled { t.muted } else { t.accent_text },
        border: border::rounded(RADIUS),
        ..Default::default()
    }
}

pub fn secondary(theme: &Theme, status: button::Status) -> button::Style {
    let t = tokens(theme);
    let bg = match status {
        button::Status::Hovered => t.line,
        button::Status::Pressed => t.surface,
        _ => t.surface2,
    };
    button::Style {
        background: Some(bg.into()),
        text_color: if status == button::Status::Disabled { t.muted } else { t.text },
        border: line_border(t, RADIUS),
        ..Default::default()
    }
}

pub fn ghost(theme: &Theme, status: button::Status) -> button::Style {
    let t = tokens(theme);
    let (bg, fg) = match status {
        button::Status::Hovered => (Some(t.surface2.into()), t.text),
        button::Status::Pressed => (Some(t.line.into()), t.text),
        button::Status::Disabled => (None, Color { a: 0.5, ..t.muted }),
        button::Status::Active => (None, t.muted),
    };
    button::Style { background: bg, text_color: fg, border: border::rounded(RADIUS), ..Default::default() }
}

/// A window button in the top bar, square so it meets the window edge.
pub fn caption(theme: &Theme, status: button::Status) -> button::Style {
    button::Style { border: border::rounded(0), ..ghost(theme, status) }
}

/// The close button turns red on hover, as it does in the system title bar.
pub fn caption_close(theme: &Theme, status: button::Status) -> button::Style {
    let background = match status {
        button::Status::Hovered => Some(color!(0xc42b1c).into()),
        button::Status::Pressed => Some(color!(0xa52618).into()),
        _ => None,
    };
    button::Style { background, text_color: tokens(theme).muted, ..Default::default() }
}

/// Solid red, so Stop is easy to find while Scoobert works.
pub fn stop(theme: &Theme, status: button::Status) -> button::Style {
    let base = if theme.extended_palette().is_dark { color!(0xd9534f) } else { color!(0xc93c37) };
    let bg = match status {
        button::Status::Hovered => mix(base, Color::WHITE, 0.12),
        button::Status::Pressed => mix(base, Color::BLACK, 0.15),
        _ => base,
    };
    button::Style { background: Some(bg.into()), text_color: Color::WHITE, border: border::rounded(RADIUS), ..Default::default() }
}

pub fn danger(theme: &Theme, status: button::Status) -> button::Style {
    let t = tokens(theme);
    let mut s = secondary(theme, status);
    s.text_color = t.danger;
    s
}

pub fn list_item(selected: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |theme, status| {
        let t = tokens(theme);
        let bg = match (selected, status) {
            (true, _) => Some(t.surface2.into()),
            (false, button::Status::Hovered | button::Status::Pressed) => Some(Color { a: 0.6, ..t.surface2 }.into()),
            _ => None,
        };
        button::Style { background: bg, text_color: t.text, border: border::rounded(RADIUS), ..Default::default() }
    }
}

pub fn tab(active: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |theme, status| {
        let t = tokens(theme);
        let fg = if active || status == button::Status::Hovered { t.text } else { t.muted };
        button::Style { background: None, text_color: fg, border: Border::default(), ..Default::default() }
    }
}

pub fn link(theme: &Theme, status: button::Status) -> button::Style {
    let t = tokens(theme);
    let fg = if status == button::Status::Hovered { t.text } else { t.accent_ink };
    button::Style { background: None, text_color: fg, ..Default::default() }
}

/// The buttons a hovered sidebar row shows: they sit on a fade from clear into the hovered row's color, so a long
/// name runs out under them instead of showing through them.
pub fn row_actions(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    fade_into(mix(t.surface, t.surface2, 0.5), 0.3)
}

/// The same fade for a conversation row, in the color of the selected row or of a hovered one. `ramp` is the
/// fraction of the width the fade takes, and the rest is solid.
pub fn conversation_actions(selected: bool, ramp: f32) -> impl Fn(&Theme) -> container::Style {
    move |theme| {
        let t = tokens(theme);
        fade_into(if selected { t.surface2 } else { mix(t.surface, t.surface2, 0.6) }, ramp)
    }
}

fn fade_into(row: Color, ramp: f32) -> container::Style {
    let fade = iced::gradient::Linear::new(iced::Degrees(90.0)).add_stop(0.0, Color { a: 0.0, ..row }).add_stop(ramp, row).add_stop(1.0, row);
    container::Style { background: Some(iced::Background::Gradient(fade.into())), border: border::rounded(RADIUS), ..Default::default() }
}

pub fn row_button(theme: &Theme, status: button::Status) -> button::Style {
    let t = tokens(theme);
    let bg = match status {
        button::Status::Hovered | button::Status::Pressed => Some(Color { a: 0.5, ..t.surface2 }.into()),
        _ => None,
    };
    button::Style { background: bg, text_color: t.text, border: border::rounded(RADIUS), ..Default::default() }
}

// ---- inputs ----

pub fn input(theme: &Theme, status: text_input::Status) -> text_input::Style {
    let t = tokens(theme);
    let border_color = match status {
        text_input::Status::Focused { .. } => t.accent_ink,
        text_input::Status::Hovered => t.muted,
        _ => t.line,
    };
    text_input::Style {
        background: t.input.into(),
        border: Border { color: border_color, width: 1.0, radius: Radius::new(RADIUS) },
        icon: t.muted,
        placeholder: t.muted,
        value: t.text,
        selection: Color { a: 0.35, ..t.accent },
    }
}

pub fn bare_editor(theme: &Theme, _status: text_editor::Status) -> text_editor::Style {
    let t = tokens(theme);
    text_editor::Style {
        background: Background::Color(Color::TRANSPARENT),
        border: Border::default(),
        placeholder: t.muted,
        value: t.text,
        selection: Color { a: 0.35, ..t.accent },
    }
}

pub fn select(theme: &Theme, status: pick_list::Status) -> pick_list::Style {
    let t = tokens(theme);
    let border_color = match status {
        pick_list::Status::Opened { .. } => t.accent_ink,
        pick_list::Status::Hovered => t.muted,
        pick_list::Status::Active => t.line,
    };
    pick_list::Style {
        text_color: t.text,
        placeholder_color: t.muted,
        handle_color: t.muted,
        background: t.input.into(),
        border: Border { color: border_color, width: 1.0, radius: Radius::new(RADIUS) },
    }
}

/// A button in muted text with no background, for a choice shown in a line of other text. Hovering adds a border.
pub fn quiet_button(theme: &Theme, status: button::Status) -> button::Style {
    let t = tokens(theme);
    let (fg, edge) = match status {
        button::Status::Hovered | button::Status::Pressed => (t.text, t.line),
        _ => (t.muted, Color::TRANSPARENT),
    };
    button::Style { background: None, text_color: fg, border: Border { color: edge, width: 1.0, radius: Radius::new(RADIUS) }, ..Default::default() }
}

pub fn quiet_select(theme: &Theme, status: pick_list::Status) -> pick_list::Style {
    let t = tokens(theme);
    let mut s = select(theme, status);
    s.background = Background::Color(Color::TRANSPARENT);
    s.text_color = t.muted;
    s.border = Border { color: if matches!(status, pick_list::Status::Active) { Color::TRANSPARENT } else { t.line }, ..s.border };
    s
}

pub fn menu(theme: &Theme) -> iced::overlay::menu::Style {
    let t = tokens(theme);
    iced::overlay::menu::Style {
        background: t.surface.into(),
        border: line_border(t, RADIUS),
        text_color: t.text,
        selected_text_color: t.text,
        selected_background: t.surface2.into(),
        shadow: Shadow { color: t.shadow, offset: Vector::new(0.0, 4.0), blur_radius: 16.0 },
    }
}

pub fn check(theme: &Theme, status: checkbox::Status) -> checkbox::Style {
    let t = tokens(theme);
    let checked = match status {
        checkbox::Status::Active { is_checked } | checkbox::Status::Hovered { is_checked } | checkbox::Status::Disabled { is_checked } => is_checked,
    };
    checkbox::Style {
        background: if checked { t.accent.into() } else { t.surface2.into() },
        icon_color: t.accent_text,
        border: Border { color: if checked { t.accent } else { t.line }, width: 1.0, radius: Radius::new(4.0) },
        text_color: Some(t.text),
    }
}

pub fn switch(theme: &Theme, status: toggler::Status) -> toggler::Style {
    let t = tokens(theme);
    let on = match status {
        toggler::Status::Active { is_toggled } | toggler::Status::Hovered { is_toggled } | toggler::Status::Disabled { is_toggled } => is_toggled,
    };
    toggler::Style {
        background: if on { t.accent.into() } else { t.surface2.into() },
        background_border_width: 1.0,
        background_border_color: if on { t.accent } else { t.line },
        foreground: if on { t.accent_text.into() } else { t.muted.into() },
        foreground_border_width: 0.0,
        foreground_border_color: Color::TRANSPARENT,
        text_color: Some(t.text),
        border_radius: None,
        padding_ratio: 0.1,
    }
}

pub fn meter(theme: &Theme) -> progress_bar::Style {
    let t = tokens(theme);
    progress_bar::Style { background: t.surface2.into(), bar: t.accent.into(), border: border::rounded(99) }
}

pub fn divider(theme: &Theme) -> rule::Style {
    rule::Style { color: tokens(theme).line, radius: Radius::new(0.0), fill_mode: rule::FillMode::Full, snap: true }
}

pub fn scrollbar(theme: &Theme, status: scrollable::Status) -> scrollable::Style {
    let t = tokens(theme);
    let active = matches!(status, scrollable::Status::Hovered { .. } | scrollable::Status::Dragged { .. });
    let rail = scrollable::Rail {
        background: None,
        border: Border::default(),
        scroller: scrollable::Scroller {
            background: if active { t.line.into() } else { t.surface2.into() },
            border: border::rounded(6),
        },
    };
    let mut style = scrollable::default(theme, status);
    style.container = container::Style::default();
    style.vertical_rail = rail;
    style.horizontal_rail = rail;
    style.gap = None;
    style
}

pub fn icon_tint(color: fn(&Tokens) -> Color) -> impl Fn(&Theme, iced::widget::svg::Status) -> iced::widget::svg::Style {
    move |theme, _| iced::widget::svg::Style { color: Some(color(tokens(theme))) }
}

pub fn mix(a: Color, b: Color, amount: f32) -> Color {
    Color {
        r: a.r + (b.r - a.r) * amount,
        g: a.g + (b.g - a.g) * amount,
        b: a.b + (b.b - a.b) * amount,
        a: a.a + (b.a - a.a) * amount,
    }
}
