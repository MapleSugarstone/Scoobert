//! Draws the notes and their links as a live force-directed graph that can be panned, zoomed, and rearranged.

use iced::mouse;
use iced::widget::canvas::{self, Frame, Geometry, Path, Stroke, Text};
use iced::window;
use iced::{Point, Rectangle, Renderer, Theme, Vector};

use super::Message;
use super::notes::Msg;
use super::theme::tokens;
use crate::notes::Graph;

/// Distance a link pulls its two notes toward, in graph units at zoom 1.
const LINK_LENGTH: f32 = 90.0;
/// How strongly notes push each other apart. Negative values repel.
const CHARGE: f32 = -520.0;
/// Notes farther apart than this stop pushing each other, so separate clusters do not fly apart.
const CHARGE_RANGE: f32 = 420.0;
/// The pull toward the middle that keeps unlinked notes in view.
const GRAVITY: f32 = 0.06;
const VELOCITY_KEEP: f32 = 0.6;
const ALPHA_MIN: f32 = 0.002;
const ALPHA_DECAY: f32 = 0.022;
/// Movement smaller than this many pixels between press and release counts as a click.
const CLICK_SLOP: f32 = 4.0;
const LABEL_CHARS: usize = 24;

pub struct GraphView {
    graph: Graph,
    degree: Vec<usize>,
}

/// The layout and camera, kept by the canvas between frames.
pub struct State {
    ids: Vec<String>,
    pos: Vec<Vector>,
    vel: Vec<Vector>,
    /// How much the layout still moves. It cools toward zero and warms up again when a note is dragged.
    alpha: f32,
    zoom: f32,
    /// Where the graph's origin sits, relative to the middle of the canvas.
    pan: Vector,
    hovered: Option<usize>,
    drag: Drag,
    /// When empty space was last clicked, so a second click soon after resets the view.
    last_click: Option<std::time::Instant>,
}

impl Default for State {
    fn default() -> Self {
        State { ids: Vec::new(), pos: Vec::new(), vel: Vec::new(), alpha: 1.0, zoom: 1.0, pan: Vector::ZERO, hovered: None, drag: Drag::None, last_click: None }
    }
}

enum Drag {
    None,
    Pan { from: Point, pan: Vector, moved: bool },
    Node { index: usize, from: Point, moved: bool },
}

impl GraphView {
    pub fn new(graph: Graph) -> GraphView {
        let mut degree = vec![0; graph.nodes.len()];
        for &(a, b) in &graph.edges {
            degree[a] += 1;
            degree[b] += 1;
        }
        GraphView { graph, degree }
    }

    pub fn is_empty(&self) -> bool {
        self.graph.nodes.is_empty()
    }

    fn radius(&self, i: usize) -> f32 {
        4.0 + (self.degree[i] as f32).sqrt() * 2.5
    }

    fn neighbors(&self, i: usize, j: usize) -> bool {
        self.graph.edges.iter().any(|&(a, b)| (a == i && b == j) || (a == j && b == i))
    }

    /// Positions for the current notes: kept where a note already had one, else on a spiral around the middle
    /// or beside a linked note that is already placed.
    fn sync(&self, state: &mut State) {
        let ids: Vec<String> = self.graph.nodes.iter().map(|n| n.id.clone()).collect();
        if ids == state.ids {
            return;
        }
        let mut pos = Vec::with_capacity(ids.len());
        for (i, id) in ids.iter().enumerate() {
            let kept = state.ids.iter().position(|old| old == id).map(|k| state.pos[k]);
            let beside = || {
                self.graph.edges.iter().find_map(|&(a, b)| {
                    let other = if a == i { b } else if b == i { a } else { return None };
                    state.ids.iter().position(|old| *old == ids[other]).map(|k| state.pos[k] + Vector::new(12.0, 9.0))
                })
            };
            pos.push(kept.or_else(beside).unwrap_or_else(|| spiral(i)));
        }
        state.vel = vec![Vector::ZERO; ids.len()];
        state.pos = pos;
        state.ids = ids;
        state.alpha = state.alpha.max(0.6);
    }

    /// One step of the layout: links pull, notes push apart, everything drifts toward the middle.
    fn tick(&self, state: &mut State) {
        let n = state.pos.len();
        let alpha = state.alpha;
        for &(a, b) in &self.graph.edges {
            let d = (state.pos[b] + state.vel[b]) - (state.pos[a] + state.vel[a]);
            let len = length(d).max(0.01);
            let strength = 0.7 / self.degree[a].min(self.degree[b]).max(1) as f32;
            let pull = d * ((len - LINK_LENGTH) / len * alpha * strength);
            let share = self.degree[a] as f32 / (self.degree[a] + self.degree[b]).max(1) as f32;
            state.vel[b] = state.vel[b] - pull * share;
            state.vel[a] = state.vel[a] + pull * (1.0 - share);
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let d = state.pos[j] - state.pos[i];
                let dist2 = (d.x * d.x + d.y * d.y).max(1.0);
                if dist2 > CHARGE_RANGE * CHARGE_RANGE {
                    continue;
                }
                let push = d * (CHARGE * alpha / dist2);
                state.vel[i] = state.vel[i] + push;
                state.vel[j] = state.vel[j] - push;
                // Circles keep a little space between them.
                let min = self.radius(i) + self.radius(j) + 6.0;
                let dist = dist2.sqrt();
                if dist < min {
                    let shove = d * ((min - dist) / dist * 0.5);
                    state.vel[i] = state.vel[i] - shove;
                    state.vel[j] = state.vel[j] + shove;
                }
            }
        }
        let held = match state.drag {
            Drag::Node { index, .. } => Some(index),
            _ => None,
        };
        for i in 0..n {
            state.vel[i] = state.vel[i] - state.pos[i] * (GRAVITY * alpha);
            if held == Some(i) {
                state.vel[i] = Vector::ZERO;
                continue;
            }
            state.vel[i] = state.vel[i] * VELOCITY_KEEP;
            state.pos[i] = state.pos[i] + state.vel[i];
        }
        state.alpha += (0.0 - state.alpha) * ALPHA_DECAY;
    }

    fn screen(&self, state: &State, world: Vector, bounds: Rectangle) -> Point {
        Point::new(bounds.width / 2.0 + state.pan.x + world.x * state.zoom, bounds.height / 2.0 + state.pan.y + world.y * state.zoom)
    }

    fn world(&self, state: &State, p: Point, bounds: Rectangle) -> Vector {
        Vector::new((p.x - bounds.width / 2.0 - state.pan.x) / state.zoom, (p.y - bounds.height / 2.0 - state.pan.y) / state.zoom)
    }

    fn hit(&self, state: &State, p: Point, bounds: Rectangle) -> Option<usize> {
        (0..state.pos.len()).rev().find(|&i| {
            let c = self.screen(state, state.pos[i], bounds);
            let r = self.radius(i) * state.zoom + 5.0;
            (c.x - p.x).powi(2) + (c.y - p.y).powi(2) <= r * r
        })
    }

    fn open(&self, i: usize) -> Message {
        let node = &self.graph.nodes[i];
        Message::Notes(if node.ghost { Msg::OpenLink(node.name.clone()) } else { Msg::Open(node.id.clone()) })
    }
}

/// The starting spiral: even spacing, the same every time, so the picture does not jump on reopening.
fn spiral(i: usize) -> Vector {
    let r = 14.0 * (0.5 + i as f32).sqrt();
    let a = i as f32 * std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    Vector::new(r * a.cos(), r * a.sin())
}

fn length(v: Vector) -> f32 {
    (v.x * v.x + v.y * v.y).sqrt()
}

impl canvas::Program<Message> for GraphView {
    type State = State;

    fn update(&self, state: &mut State, event: &canvas::Event, bounds: Rectangle, cursor: mouse::Cursor) -> Option<canvas::Action<Message>> {
        self.sync(state);
        let p = cursor.position_in(bounds);
        match event {
            canvas::Event::Window(window::Event::RedrawRequested(_)) => {
                let dragging = matches!(state.drag, Drag::Node { .. });
                if state.alpha < ALPHA_MIN && !dragging {
                    return None;
                }
                self.tick(state);
                Some(canvas::Action::request_redraw())
            }
            canvas::Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                let p = match (&state.drag, p, cursor.position()) {
                    // A drag continues outside the canvas, measured from its top left corner.
                    (Drag::None, None, _) => {
                        let changed = state.hovered.take().is_some();
                        return changed.then(canvas::Action::request_redraw);
                    }
                    (_, Some(p), _) => p,
                    (_, None, Some(abs)) => Point::new(abs.x - bounds.x, abs.y - bounds.y),
                    (_, None, None) => return None,
                };
                match &mut state.drag {
                    Drag::Pan { from, pan, moved } => {
                        *moved |= p.distance(*from) > CLICK_SLOP;
                        state.pan = *pan + (p - *from);
                        Some(canvas::Action::request_redraw())
                    }
                    Drag::Node { index, from, moved } => {
                        *moved |= p.distance(*from) > CLICK_SLOP;
                        let i = *index;
                        if *moved {
                            state.pos[i] = self.world(state, p, bounds);
                            state.vel[i] = Vector::ZERO;
                            state.alpha = state.alpha.max(0.3);
                        }
                        Some(canvas::Action::request_redraw())
                    }
                    Drag::None => {
                        let hovered = self.hit(state, p, bounds);
                        if hovered != state.hovered {
                            state.hovered = hovered;
                            return Some(canvas::Action::request_redraw());
                        }
                        None
                    }
                }
            }
            canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let p = p?;
                state.drag = match self.hit(state, p, bounds) {
                    Some(index) => Drag::Node { index, from: p, moved: false },
                    None => Drag::Pan { from: p, pan: state.pan, moved: false },
                };
                Some(canvas::Action::request_redraw().and_capture())
            }
            canvas::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                let drag = std::mem::replace(&mut state.drag, Drag::None);
                match drag {
                    Drag::Node { index, moved: false, .. } => Some(canvas::Action::publish(self.open(index)).and_capture()),
                    Drag::Pan { moved: false, .. } => {
                        let double = state.last_click.is_some_and(|t| t.elapsed() < std::time::Duration::from_millis(400));
                        state.last_click = (!double).then(std::time::Instant::now);
                        if double {
                            state.zoom = 1.0;
                            state.pan = Vector::ZERO;
                        }
                        Some(canvas::Action::request_redraw().and_capture())
                    }
                    Drag::None => None,
                    _ => Some(canvas::Action::request_redraw().and_capture()),
                }
            }
            canvas::Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
                let p = p?;
                let lines = match delta {
                    mouse::ScrollDelta::Lines { y, .. } => *y,
                    mouse::ScrollDelta::Pixels { y, .. } => *y / 40.0,
                };
                // The point under the cursor stays put while the zoom changes around it.
                let before = self.world(state, p, bounds);
                state.zoom = (state.zoom * 1.15_f32.powf(lines)).clamp(0.15, 4.0);
                let after = self.screen(state, before, bounds);
                state.pan = state.pan + (p - after);
                Some(canvas::Action::request_redraw().and_capture())
            }
            _ => None,
        }
    }

    fn draw(&self, state: &State, renderer: &Renderer, theme: &Theme, bounds: Rectangle, _cursor: mouse::Cursor) -> Vec<Geometry> {
        let t = tokens(theme);
        let mut frame = Frame::new(renderer, bounds.size());
        // A darker ground than the rest of the pane, so the notes stand out.
        frame.fill_rectangle(Point::ORIGIN, bounds.size(), t.code_bg);
        // Before the first update the layout has no positions yet, so the starting spiral is drawn.
        let start: Vec<Vector>;
        let pos: &[Vector] = if state.pos.len() == self.graph.nodes.len() {
            &state.pos
        } else {
            start = (0..self.graph.nodes.len()).map(spiral).collect();
            &start
        };
        let at = |i: usize| self.screen(state, pos[i], bounds);
        let focus = state.hovered.or(match state.drag {
            Drag::Node { index, .. } => Some(index),
            _ => None,
        });
        let near = |i: usize| focus.is_some_and(|h| h == i || self.neighbors(h, i));
        for &(a, b) in &self.graph.edges {
            let lit = focus.is_some_and(|h| h == a || h == b);
            let color = if lit { t.accent_ink } else { iced::Color { a: 0.55, ..t.line } };
            frame.stroke(&Path::line(at(a), at(b)), Stroke::default().with_color(color).with_width(if lit { 1.6 } else { 1.0 }));
        }
        for (i, node) in self.graph.nodes.iter().enumerate() {
            let c = at(i);
            let r = (self.radius(i) * state.zoom).max(2.5);
            let dim = focus.is_some() && !near(i);
            // Notes are green like the mascot. A link to a missing note is an outline.
            let fill = if focus == Some(i) {
                t.accent_ink
            } else if node.ghost {
                t.code_bg
            } else if dim {
                iced::Color { a: 0.3, ..t.accent }
            } else {
                iced::Color { a: 0.85, ..t.accent }
            };
            frame.fill(&Path::circle(c, r), fill);
            if node.ghost {
                frame.stroke(&Path::circle(c, r), Stroke::default().with_color(t.muted).with_width(1.0));
            }
        }
        // Labels fade out when zoomed far out, except around the note being pointed at.
        let label_alpha = ((state.zoom - 0.35) / 0.35).clamp(0.0, 1.0);
        for (i, node) in self.graph.nodes.iter().enumerate() {
            let shown = if near(i) { 1.0 } else if focus.is_some() { label_alpha * 0.35 } else { label_alpha };
            if shown <= 0.02 {
                continue;
            }
            let c = at(i);
            let r = (self.radius(i) * state.zoom).max(2.5);
            frame.fill_text(Text {
                content: if focus == Some(i) { node.name.clone() } else { crate::util::clip(&node.name, LABEL_CHARS) },
                position: c + Vector::new(0.0, r + 4.0),
                color: iced::Color { a: shown, ..t.text },
                size: (12.0 * state.zoom.clamp(0.85, 1.4)).into(),
                font: super::fonts::ui(),
                align_x: iced::widget::text::Alignment::Center,
                ..Text::default()
            });
        }
        vec![frame.into_geometry()]
    }

    fn mouse_interaction(&self, state: &State, bounds: Rectangle, cursor: mouse::Cursor) -> mouse::Interaction {
        match state.drag {
            Drag::Pan { moved: true, .. } | Drag::Node { moved: true, .. } => mouse::Interaction::Grabbing,
            _ if state.hovered.is_some() => mouse::Interaction::Pointer,
            _ if cursor.is_over(bounds) => mouse::Interaction::Grab,
            _ => mouse::Interaction::default(),
        }
    }
}
