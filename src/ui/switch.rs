//! An on and off switch whose knob slides across and whose color fades when the user flips it.

use std::time::Duration;

use iced::advanced::layout::{self, Layout};
use iced::advanced::widget::{Tree, Widget, tree};
use iced::advanced::{Clipboard, Shell, mouse, renderer};
use iced::animation::Easing;
use iced::time::Instant;
use iced::widget::toggler;
use iced::{Animation, Background, Border, Color, Element, Event, Length, Rectangle, Size, Theme, window};

const SLIDE: Duration = Duration::from_millis(160);

pub struct Switch<'a, Message> {
    on: bool,
    size: f32,
    on_toggle: Box<dyn Fn(bool) -> Message + 'a>,
}

pub fn switch<'a, Message>(on: bool, on_toggle: impl Fn(bool) -> Message + 'a) -> Switch<'a, Message> {
    Switch { on, size: 20.0, on_toggle: Box::new(on_toggle) }
}

struct State {
    animation: Animation<bool>,
    now: Instant,
    /// The user flipped this switch, so the next change of its value slides. A value that changes for another
    /// reason, such as a page that reuses the switch for another setting, jumps.
    flipped: bool,
}

fn animation(on: bool) -> Animation<bool> {
    Animation::new(on).duration(SLIDE).easing(Easing::EaseInOut)
}

impl<Message, Renderer> Widget<Message, Theme, Renderer> for Switch<'_, Message>
where
    Renderer: iced::advanced::Renderer,
{
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(State { animation: animation(self.on), now: Instant::now(), flipped: false })
    }

    fn size(&self) -> Size<Length> {
        Size::new(Length::Fixed(2.0 * self.size), Length::Fixed(self.size))
    }

    fn layout(&mut self, _tree: &mut Tree, _renderer: &Renderer, _limits: &layout::Limits) -> layout::Node {
        layout::Node::new(Size::new(2.0 * self.size, self.size))
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _renderer: &Renderer,
        _clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        _viewport: &Rectangle,
    ) {
        let state = tree.state.downcast_mut::<State>();
        match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) if cursor.is_over(layout.bounds()) => {
                state.flipped = true;
                shell.publish((self.on_toggle)(!self.on));
                shell.capture_event();
            }
            Event::Window(window::Event::RedrawRequested(now)) => {
                state.now = *now;
                if state.animation.value() != self.on {
                    if state.flipped {
                        state.animation.go_mut(self.on, *now);
                    } else {
                        state.animation = animation(self.on);
                    }
                    state.flipped = false;
                }
                if state.animation.is_animating(*now) {
                    shell.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn mouse_interaction(&self, _tree: &Tree, layout: Layout<'_>, cursor: mouse::Cursor, _viewport: &Rectangle, _renderer: &Renderer) -> mouse::Interaction {
        if cursor.is_over(layout.bounds()) { mouse::Interaction::Pointer } else { mouse::Interaction::default() }
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        let state = tree.state.downcast_ref::<State>();
        let t = state.animation.interpolate(0.0f32, 1.0f32, state.now);
        let off = super::theme::switch(theme, toggler::Status::Active { is_toggled: false });
        let on = super::theme::switch(theme, toggler::Status::Active { is_toggled: true });
        let bounds = layout.bounds();
        let border = Border { radius: (bounds.height / 2.0).into(), width: off.background_border_width, color: mix(off.background_border_color, on.background_border_color, t) };
        renderer.fill_quad(renderer::Quad { bounds, border, ..renderer::Quad::default() }, mix(solid(off.background), solid(on.background), t));
        let padding = (off.padding_ratio * bounds.height).round();
        let knob = bounds.height - 2.0 * padding;
        let knob_bounds = Rectangle { x: bounds.x + padding + t * (bounds.width - bounds.height), y: bounds.y + padding, width: knob, height: knob };
        let knob_border = Border { radius: (knob / 2.0).into(), ..Border::default() };
        renderer.fill_quad(renderer::Quad { bounds: knob_bounds, border: knob_border, ..renderer::Quad::default() }, mix(solid(off.foreground), solid(on.foreground), t));
    }
}

fn solid(background: Background) -> Color {
    match background {
        Background::Color(c) => c,
        Background::Gradient(_) => Color::TRANSPARENT,
    }
}

fn mix(a: Color, b: Color, t: f32) -> Color {
    Color { r: a.r + (b.r - a.r) * t, g: a.g + (b.g - a.g) * t, b: a.b + (b.b - a.b) * t, a: a.a + (b.a - a.a) * t }
}

impl<'a, Message, Renderer> From<Switch<'a, Message>> for Element<'a, Message, Theme, Renderer>
where
    Message: 'a,
    Renderer: iced::advanced::Renderer + 'a,
{
    fn from(switch: Switch<'a, Message>) -> Self {
        Element::new(switch)
    }
}
