//! Lays out paragraphs that mix Arabic and English at every width from 120 to 600 in steps of 5 and checks that no
//! line is wider than the width, that a relayout at the measured width (which iced does for right-to-left text) keeps
//! the same lines, and that each line shows its glyphs in the order the Unicode bidirectional algorithm gives.
//! Usage: cargo run --release --example rtl_measure -- <dump file> [baseline dump file]
//! Set `RTL_MEASURE_STEP` to sweep with another step.
//! The dump file gets a fingerprint of every laid out line, and `<dump file>.lines.txt` gets sample lines in visual
//! order. With a baseline dump, the example also reports which layouts changed.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use iced::advanced::graphics::text::cosmic_text::{Attrs, Buffer, Family, FontSystem, LayoutGlyph, Metrics, Shaping, Wrap};

const MIXED: &[(&str, &str)] = &[
    ("notes", "الملاحظات تابعة لمشروع. اختر مشروعًا من الشريط الجانبي أو اطلب من Scoobert بدء مشروع، وستظهر ملاحظاته هنا."),
    ("latin-first", "Scoobert يحفظ الملاحظات في مجلد المشروع ويقرؤها في بداية كل محادثة جديدة حتى يتذكر ما اتفقتما عليه."),
    ("second-word", "مرحبًا، Scoobert هنا. يمكنني قراءة ملفات المشروع وتعديلها وتشغيل الأوامر بعد موافقتك على كل خطوة."),
    ("latin-last", "إذا لم يبدأ الخادم المحلي فتحقق من السجل ثم أعد تشغيل التطبيق وجرّب مرة أخرى مع Scoobert"),
    ("latin-last-period", "تجد هذا الخيار في الإعدادات تحت قسم النماذج المحلية التي يشغلها Scoobert."),
    ("shortcut", "اضغط Shift+Enter لإضافة سطر جديد دون إرسال الرسالة، واضغط Enter وحده لإرسالها إلى النموذج."),
    ("server", "يشغّل التطبيق llama-server على جهازك ويستمع على المنفذ 8080، فلا تغادر بياناتك هذا الحاسوب أبدًا."),
    ("model", "اكتمل تنزيل النموذج Qwen3.5-9B-Q4_K_M بحجم 5.6 GB، ويحتاج إلى 8 GB من الذاكرة على الأقل لتشغيله بسرعة جيدة."),
    ("phrase", "يمكنك اختيار Local model أو Hosted model من الإعدادات، ثم الضغط على Save settings لحفظ التغييرات."),
    ("attached", "استخدم Ctrl+K للبحث وCtrl+N لمحادثة جديدة، وكل الاختصارات تعمل في Windows وLinux بالطريقة نفسها."),
    ("brackets", "افتح الملف \"config.json\" أو اكتب [[wikilinks]] لربط الملاحظات، ثم اطلب من (Scoobert) تلخيصها في ملاحظة واحدة."),
    ("dense", "النموذج Qwen3.8 27B، دقة عالية، يحتاج إلى 24 GB، أما Qwen3.5-9B فيعمل على 8 GB مع llama-server وCUDA أو Vulkan."),
    ("hyphens", "افتح الملف ملاحظات-المشروع-Scoobert.md لترى ما حفظه المساعد، وإذا توقف الخادم فأعد تشغيل llama-server، ثم انتظر قليلًا."),
    ("long-token", "اضبط المتغير وCUDA_VISIBLE_DEVICES=0، ثم أعد تشغيل الخادم ليستخدم بطاقة الرسوميات الأولى فقط."),
];

const PURE: &[(&str, &str)] = &[
    (
        "english",
        "Notes belong to a project. Pick a project from the sidebar or ask Scoobert to start one, and its notes will show up here. Press Shift+Enter for a new line. The model Qwen3.5-9B-Q4_K_M runs on llama-server at port 8080, so your files never leave this computer.",
    ),
    (
        "arabic",
        "الملاحظات تابعة لمشروع. اختر مشروعًا من الشريط الجانبي أو اطلب من المساعد بدء مشروع، وستظهر ملاحظاته هنا. يحفظ التطبيق كل شيء على جهازك ولا يرسل بياناتك إلى أي مكان آخر دون إذنك، ويمكنك حذف أي ملاحظة متى شئت.",
    ),
];

const WRAPS: [Wrap; 2] = [Wrap::Word, Wrap::WordOrGlyph];
const SAMPLE_WIDTHS: [u32; 3] = [380, 240, 160];

struct Line {
    w: f32,
    rtl: bool,
    glyphs: Vec<LayoutGlyph>,
}

fn buffer(fonts: &mut FontSystem, text: &str, wrap: Wrap, width: Option<f32>) -> Buffer {
    let mut buffer = Buffer::new(fonts, Metrics::new(13.0, 17.0));
    buffer.set_wrap(fonts, wrap);
    buffer.set_size(fonts, width, None);
    buffer.set_text(fonts, text, &Attrs::new().family(Family::Name("Segoe UI")), Shaping::Advanced, None);
    buffer
}

fn lines(buffer: &Buffer) -> Vec<Line> {
    buffer.layout_runs().map(|r| Line { w: r.line_w, rtl: r.rtl, glyphs: r.glyphs.to_vec() }).collect()
}

/// Returns the number of spans and, for every word that is not a blank, the starts of its glyphs.
fn shape_info(fonts: &mut FontSystem, text: &str) -> (usize, Vec<Vec<usize>>) {
    let mut buffer = buffer(fonts, text, Wrap::Word, None);
    let shape = buffer.line_shape(fonts, 0).expect("one line");
    let words = shape.spans.iter().flat_map(|s| &s.words).filter(|w| !w.blank).map(|w| w.glyphs.iter().map(|g| g.start).collect()).collect();
    (shape.spans.len(), words)
}

fn words_on(line: &Line, words: &[Vec<usize>]) -> usize {
    let starts: HashSet<usize> = line.glyphs.iter().map(|g| g.start).collect();
    words.iter().filter(|w| w.iter().any(|s| starts.contains(s))).count()
}

/// Returns the glyph clusters of a line from left to right, with their bidi levels.
fn clusters(line: &Line) -> Vec<(usize, usize, u8)> {
    let mut out: Vec<(usize, usize, u8)> = Vec::new();
    for g in &line.glyphs {
        if out.last().is_none_or(|c| (c.0, c.1) != (g.start, g.end)) {
            out.push((g.start, g.end, g.level.number()));
        }
    }
    if line.rtl {
        out.reverse();
    }
    out
}

/// Puts clusters in logical order and then reorders them with rule L2 of the Unicode bidirectional algorithm.
fn bidi_order(mut items: Vec<(usize, usize, u8)>) -> Vec<(usize, usize, u8)> {
    items.sort();
    items.dedup();
    let (Some(lowest), Some(highest)) = (items.iter().map(|c| c.2).min(), items.iter().map(|c| c.2).max()) else {
        return items;
    };
    for level in ((lowest | 1)..=highest).rev() {
        let mut i = 0;
        while i < items.len() {
            if items[i].2 < level {
                i += 1;
                continue;
            }
            let start = i;
            while i < items.len() && items[i].2 >= level {
                i += 1;
            }
            items[start..i].reverse();
        }
    }
    items
}

fn breaks(lines: &[Line]) -> Vec<Vec<(usize, usize)>> {
    lines.iter().map(|l| l.glyphs.iter().map(|g| (g.start, g.end)).collect()).collect()
}

/// Returns the characters on each side of every line break that does not fall on a space.
fn breaks_in_words(text: &str, lines: &[Line]) -> Vec<String> {
    let ranges: Vec<(usize, usize)> = lines
        .iter()
        .filter(|l| !l.glyphs.is_empty())
        .map(|l| (l.glyphs.iter().map(|g| g.start).min().unwrap_or(0), l.glyphs.iter().map(|g| g.end).max().unwrap_or(0)))
        .collect();
    ranges
        .windows(2)
        .filter(|pair| pair[0].1 >= pair[1].0)
        .map(|pair| format!("{}|{}", text[..pair[1].0].chars().last().unwrap_or(' '), text[pair[1].0..].chars().next().unwrap_or(' ')))
        .collect()
}

fn fnv(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
}

fn fingerprint(line: &Line) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for g in &line.glyphs {
        for value in [g.start as u64, g.end as u64, u64::from(g.glyph_id), u64::from(g.level.number())] {
            fnv(&mut hash, value);
        }
        for value in [g.x, g.y, g.w, g.font_size, g.x_offset, g.y_offset] {
            fnv(&mut hash, u64::from(value.to_bits()));
        }
    }
    hash
}

/// Writes a line as runs of one direction from left to right, each run's text in logical order.
fn visual_runs(text: &str, line: &Line) -> String {
    let cs = clusters(line);
    let mut out = String::new();
    let mut i = 0;
    while i < cs.len() {
        let odd = cs[i].2 % 2;
        let mut j = i;
        while j < cs.len() && cs[j].2 % 2 == odd {
            j += 1;
        }
        let from = cs[i..j].iter().map(|c| c.0).min().unwrap_or(0);
        let to = cs[i..j].iter().map(|c| c.1).max().unwrap_or(0);
        let _ = write!(out, "[{} {}] ", if odd == 1 { "R" } else { "L" }, text[from..to].trim());
        i = j;
    }
    out
}

fn parse_dump(dump: &str) -> HashMap<String, (bool, String)> {
    dump.lines()
        .filter_map(|l| {
            let (key, rest) = l.split_once(" over=")?;
            let (flag, body) = rest.split_once(':')?;
            Some((key.to_string(), (flag == "1", body.to_string())))
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(dump_path) = args.first() else {
        eprintln!("usage: rtl_measure <dump file> [baseline dump file]");
        return;
    };
    let mut fonts = FontSystem::new();
    let mut dump = String::new();
    let mut samples = String::new();
    let mut totals: HashMap<&str, [usize; 5]> = HashMap::new();
    let step = std::env::var("RTL_MEASURE_STEP").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let widths: Vec<u32> = (120..=600).step_by(step).collect();

    println!("{} widths from 120 to 600 in steps of {step}. Columns: spans, overflowing widths, single-word overflows, widths with glyphs outside the box, relayout changes, misordered lines", widths.len());
    for (group, texts) in [("mixed", MIXED), ("pure", PURE)] {
        for &(name, text) in texts {
            let (spans, words) = shape_info(&mut fonts, text);
            for wrap in WRAPS {
                let (mut over, mut excused, mut outside, mut relayout, mut misordered) = (Vec::new(), 0, 0, 0, 0);
                let mut in_words: HashMap<String, usize> = HashMap::new();
                for &width in &widths {
                    let w = width as f32;
                    let mut buf = buffer(&mut fonts, text, wrap, Some(w));
                    let laid = lines(&buf);
                    let measured = laid.iter().map(|l| l.w).fold(0.0, f32::max);
                    buf.set_size(&mut fonts, Some(measured), None);
                    let relaid = lines(&buf);

                    let wide: Vec<&Line> = laid.iter().filter(|l| l.w > w).collect();
                    if !wide.is_empty() {
                        if wide.iter().all(|l| words_on(l, &words) == 1) {
                            excused += 1;
                        } else {
                            over.push(width);
                        }
                    }
                    if laid.iter().any(|l| l.glyphs.iter().any(|g| g.x < -0.01 || g.x + g.w > w + 0.01)) {
                        outside += 1;
                    }
                    if breaks(&laid) != breaks(&relaid) {
                        relayout += 1;
                    }
                    for pair in breaks_in_words(text, &laid) {
                        *in_words.entry(pair).or_default() += 1;
                    }
                    let bad: Vec<&Line> = laid.iter().filter(|l| clusters(l) != bidi_order(clusters(l))).collect();
                    misordered += bad.len();
                    for l in &bad {
                        let _ = writeln!(samples, "MISORDERED {name} {wrap:?} {width}: {}", visual_runs(text, l));
                    }

                    let _ = write!(dump, "{group} {name} {wrap:?} {width} over={}:", u8::from(!wide.is_empty()));
                    for l in &laid {
                        let _ = write!(dump, " {:08x}/{}/{:016x}", l.w.to_bits(), l.glyphs.len(), fingerprint(l));
                    }
                    dump.push('\n');

                    if group == "mixed" && wrap == Wrap::Word && SAMPLE_WIDTHS.contains(&width) {
                        let _ = writeln!(samples, "{name} at width {width}:");
                        for l in &laid {
                            let _ = writeln!(samples, "  w={:7.2} {}", l.w, visual_runs(text, l));
                        }
                    }
                }
                println!(
                    "{group:5} {name:17} {:11}: {spans:2} spans, {:2} over {:?}, {excused} single-word, {outside:2} outside, {relayout} relayout, {misordered} misordered, {} breaks not at a space",
                    format!("{wrap:?}"),
                    over.len(),
                    over.iter().take(10).collect::<Vec<_>>(),
                    in_words.values().sum::<usize>(),
                );
                let mut pairs: Vec<_> = in_words.into_iter().collect();
                pairs.sort();
                let _ = writeln!(samples, "BREAKS NOT AT A SPACE {name} {wrap:?}: {pairs:?}");
                let t = totals.entry(group).or_default();
                for (slot, value) in t.iter_mut().zip([over.len(), excused, outside, relayout, misordered]) {
                    *slot += value;
                }
            }
        }
    }
    for group in ["mixed", "pure"] {
        let t = totals[group];
        println!("total {group}: {} over, {} single-word, {} outside, {} relayout, {} misordered", t[0], t[1], t[2], t[3], t[4]);
    }

    std::fs::write(dump_path, &dump).expect("write dump");
    std::fs::write(format!("{dump_path}.lines.txt"), &samples).expect("write sample lines");

    if let Some(path) = args.get(1) {
        let old = parse_dump(&std::fs::read_to_string(path).expect("read baseline dump"));
        let new = parse_dump(&dump);
        for group in ["mixed", "pure"] {
            let (mut same, mut fixed, mut other) = (0, 0, Vec::new());
            for (key, (_, body)) in new.iter().filter(|(k, _)| k.starts_with(group)) {
                match old.get(key) {
                    Some((_, old_body)) if old_body == body => same += 1,
                    Some((true, _)) => fixed += 1,
                    _ => other.push(key.clone()),
                }
            }
            other.sort();
            println!("against baseline, {group}: {same} layouts identical, {fixed} changed that overflowed before, {} changed that did not {:?}", other.len(), other.iter().take(10).collect::<Vec<_>>());
        }
    }
}
