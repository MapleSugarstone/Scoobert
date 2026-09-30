//! Draws the notes that link to each other, laid out with a force-directed placement.

use iced::mouse;
use iced::widget::canvas::{self, Frame, Geometry, Path, Stroke, Text};
use iced::{Point, Rectangle, Renderer, Theme, Vector};

use super::Message;
use super::notes::Msg;
use super::theme::tokens;
use crate::notes::Graph;

pub struct GraphView {
    graph: Graph,
    /// Node positions in a unit square.
    pos: Vec<(f32, f32)>,
    degree: Vec<usize>,
    cache: canvas::Cache,
}

#[derive(Default)]
pub struct State {
    hovered: Option<usize>,
}

impl GraphView {
    pub fn new(graph: Graph) -> GraphView {
        let pos = layout(&graph);
        let mut degree = vec![0; graph.nodes.len()];
        for &(a, b) in &graph.edges {
            degree[a] += 1;
            degree[b] += 1;
        }
        GraphView { graph, pos, degree, cache: canvas::Cache::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.graph.nodes.is_empty()
    }

    fn to_screen(&self, i: usize, bounds: Rectangle) -> Point {
        let pad = 40.0;
        let (x, y) = self.pos[i];
        Point::new(pad + x * (bounds.width - 2.0 * pad).max(1.0), pad + y * (bounds.height - 2.0 * pad).max(1.0))
    }

    fn radius(&self, i: usize) -> f32 {
        4.0 + (self.degree[i] as f32).sqrt() * 2.0
    }

    fn hit(&self, p: Point, bounds: Rectangle) -> Option<usize> {
        (0..self.pos.len()).find(|&i| {
            let c = self.to_screen(i, bounds);
            let r = self.radius(i) + 4.0;
            (c.x - p.x).powi(2) + (c.y - p.y).powi(2) <= r * r
        })
    }
}

impl canvas::Program<Message> for GraphView {
    type State = State;

    fn update(&self, state: &mut State, event: &canvas::Event, bounds: Rectangle, cursor: mouse::Cursor) -> Option<canvas::Action<Message>> {
        let p = cursor.position_in(bounds);
        match event {
            canvas::Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                let hovered = p.and_then(|p| self.hit(p, bounds));
                if hovered != state.hovered {
                    state.hovered = hovered;
                    self.cache.clear();
                    return Some(canvas::Action::request_redraw());
                }
                None
            }
            canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let i = p.and_then(|p| self.hit(p, bounds))?;
                let node = &self.graph.nodes[i];
                let msg = if node.ghost { Msg::OpenLink(node.name.clone()) } else { Msg::Open(node.id.clone()) };
                Some(canvas::Action::publish(Message::Notes(msg)).and_capture())
            }
            _ => None,
        }
    }

    fn draw(&self, state: &State, renderer: &Renderer, theme: &Theme, bounds: Rectangle, _cursor: mouse::Cursor) -> Vec<Geometry> {
        let t = tokens(theme);
        let geometry = self.cache.draw(renderer, bounds.size(), |frame: &mut Frame| {
            let near = |i: usize| state.hovered.is_some_and(|h| h == i || self.graph.edges.iter().any(|&(a, b)| (a == h && b == i) || (b == h && a == i)));
            for &(a, b) in &self.graph.edges {
                let lit = state.hovered.is_some_and(|h| h == a || h == b);
                let color = if lit { t.accent_ink } else { t.line };
                frame.stroke(&Path::line(self.to_screen(a, bounds), self.to_screen(b, bounds)), Stroke::default().with_color(color).with_width(if lit { 1.6 } else { 1.0 }));
            }
            for (i, node) in self.graph.nodes.iter().enumerate() {
                let c = self.to_screen(i, bounds);
                let r = self.radius(i);
                let fill = if state.hovered == Some(i) {
                    t.accent
                } else if node.ghost {
                    t.surface2
                } else {
                    t.muted
                };
                frame.fill(&Path::circle(c, r), fill);
                if node.ghost {
                    frame.stroke(&Path::circle(c, r), Stroke::default().with_color(t.muted).with_width(1.0));
                }
                let dim = state.hovered.is_some() && !near(i);
                frame.fill_text(Text {
                    content: node.name.clone(),
                    position: c + Vector::new(0.0, r + 4.0),
                    color: if dim { iced::Color { a: 0.35, ..t.text } } else { t.text },
                    size: 12.0.into(),
                    font: super::fonts::ui(),
                    align_x: iced::widget::text::Alignment::Center,
                    ..Text::default()
                });
            }
        });
        vec![geometry]
    }

    fn mouse_interaction(&self, state: &State, _bounds: Rectangle, _cursor: mouse::Cursor) -> mouse::Interaction {
        if state.hovered.is_some() { mouse::Interaction::Pointer } else { mouse::Interaction::default() }
    }
}

/// Fruchterman-Reingold placement, scaled to fill a unit square.
fn layout(g: &Graph) -> Vec<(f32, f32)> {
    let n = g.nodes.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![(0.5, 0.5)];
    }
    // A fixed start on a circle keeps the picture the same each time the graph opens.
    let mut pos: Vec<(f32, f32)> = (0..n)
        .map(|i| {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            (0.5 + 0.4 * a.cos(), 0.5 + 0.4 * a.sin())
        })
        .collect();
    let k = (1.0 / n as f32).sqrt();
    let mut temp = 0.1;
    for _ in 0..300 {
        let mut disp = vec![(0.0f32, 0.0f32); n];
        for i in 0..n {
            for j in (i + 1)..n {
                let (dx, dy) = (pos[i].0 - pos[j].0, pos[i].1 - pos[j].1);
                let d = (dx * dx + dy * dy).sqrt().max(0.001);
                let f = k * k / d;
                disp[i].0 += dx / d * f;
                disp[i].1 += dy / d * f;
                disp[j].0 -= dx / d * f;
                disp[j].1 -= dy / d * f;
            }
        }
        for &(a, b) in &g.edges {
            let (dx, dy) = (pos[a].0 - pos[b].0, pos[a].1 - pos[b].1);
            let d = (dx * dx + dy * dy).sqrt().max(0.001);
            let f = d * d / k;
            disp[a].0 -= dx / d * f;
            disp[a].1 -= dy / d * f;
            disp[b].0 += dx / d * f;
            disp[b].1 += dy / d * f;
        }
        for i in 0..n {
            // A weak pull to the center keeps unlinked notes from drifting to the edges.
            disp[i].0 += (0.5 - pos[i].0) * k * 0.5;
            disp[i].1 += (0.5 - pos[i].1) * k * 0.5;
            let d = (disp[i].0 * disp[i].0 + disp[i].1 * disp[i].1).sqrt().max(0.0001);
            pos[i].0 += disp[i].0 / d * d.min(temp);
            pos[i].1 += disp[i].1 / d * d.min(temp);
        }
        temp *= 0.985;
    }
    let (minx, maxx) = pos.iter().fold((f32::MAX, f32::MIN), |(a, b), p| (a.min(p.0), b.max(p.0)));
    let (miny, maxy) = pos.iter().fold((f32::MAX, f32::MIN), |(a, b), p| (a.min(p.1), b.max(p.1)));
    let (w, h) = ((maxx - minx).max(0.001), (maxy - miny).max(0.001));
    pos.into_iter().map(|(x, y)| ((x - minx) / w, (y - miny) / h)).collect()
}
