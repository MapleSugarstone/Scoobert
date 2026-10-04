//! Memory and process queries that differ between Windows, Linux, and macOS.

/// Physical memory the system can give to a new process without paging, in bytes.
pub fn available_memory() -> u64 {
    imp::memory().1
}

pub fn total_memory() -> u64 {
    imp::memory().0
}

/// Free memory on the graphics card with the most of it, from `nvidia-smi` for NVIDIA cards and from the amdgpu
/// driver on Linux. None when no card reports it, as for Intel cards, AMD cards on Windows, and Macs, whose graphics
/// share the system's memory.
pub fn free_vram() -> Option<u64> {
    let nvidia = crate::llama::cuda::detect().and_then(|_| nvidia_vram("memory.free"));
    let amd = if cfg!(target_os = "linux") { amd_vram(true) } else { None };
    nvidia.into_iter().chain(amd).max()
}

/// Memory on the graphics card with the most of it, used or not, from the same sources as `free_vram`.
pub fn total_vram() -> Option<u64> {
    static TOTAL: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *TOTAL.get_or_init(|| {
        let nvidia = crate::llama::cuda::detect().and_then(|_| nvidia_vram("memory.total"));
        let amd = if cfg!(target_os = "linux") { amd_vram(false) } else { None };
        nvidia.into_iter().chain(amd).max()
    })
}

fn nvidia_vram(field: &str) -> Option<u64> {
    let mut cmd = std::process::Command::new("nvidia-smi");
    cmd.args([&format!("--query-gpu={field}"), "--format=csv,noheader,nounits"]).stdin(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let out = cmd.output().ok().filter(|o| o.status.success())?;
    String::from_utf8_lossy(&out.stdout).lines().filter_map(|l| l.trim().parse::<u64>().ok()).max().map(|mib| mib * 1024 * 1024)
}

/// The amdgpu driver's own count of each card's memory, in /sys/class/drm/cardN/device.
fn amd_vram(free: bool) -> Option<u64> {
    let read = |path: std::path::PathBuf| std::fs::read_to_string(path).ok()?.trim().parse::<u64>().ok();
    std::fs::read_dir("/sys/class/drm")
        .ok()?
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().strip_prefix("card").is_some_and(|n| n.chars().all(|c| c.is_ascii_digit())))
        .filter_map(|e| {
            let device = e.path().join("device");
            let total = read(device.join("mem_info_vram_total"))?;
            if free { Some(total.saturating_sub(read(device.join("mem_info_vram_used"))?)) } else { Some(total) }
        })
        .max()
}

/// What the clipboard holds besides text: an image with its type, or files copied in the file manager.
#[derive(Debug, Clone)]
pub enum Clipped {
    Image(&'static str, Vec<u8>),
    Files(Vec<std::path::PathBuf>),
}

/// Reads an image or copied files from the clipboard, for pasting into a message. The text box pastes text itself.
pub fn clipboard_extra() -> Option<Clipped> {
    clip::read()
}

#[cfg(windows)]
mod clip {
    use super::Clipped;
    use clipboard_win::{formats, get_clipboard};

    pub fn read() -> Option<Clipped> {
        if let Ok(files) = get_clipboard::<Vec<String>, _>(formats::FileList)
            && !files.is_empty()
        {
            return Some(Clipped::Files(files.into_iter().map(std::path::PathBuf::from).collect()));
        }
        let bmp: Vec<u8> = get_clipboard(formats::Bitmap).ok()?;
        let picture = image::load_from_memory_with_format(&bmp, image::ImageFormat::Bmp).ok()?;
        // Screenshots arrive as 32-bit bitmaps whose fourth byte is often zero, which would read as fully transparent.
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(picture.to_rgb8()).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).ok()?;
        Some(Clipped::Image("image/png", png))
    }
}

/// Linux has no clipboard API outside the desktop's own tools, so this asks wl-paste on Wayland and xclip on X11.
#[cfg(target_os = "linux")]
mod clip {
    use super::Clipped;
    use std::process::{Command, Stdio};

    fn run(program: &str, args: &[&str]) -> Option<Vec<u8>> {
        let out = Command::new(program).args(args).stdin(Stdio::null()).stderr(Stdio::null()).output().ok()?;
        out.status.success().then_some(out.stdout)
    }

    pub fn read() -> Option<Clipped> {
        let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
        let get = |kind: &str| if wayland { run("wl-paste", &["--no-newline", "--type", kind]) } else { run("xclip", &["-selection", "clipboard", "-t", kind, "-o"]) };
        let kinds = if wayland { run("wl-paste", &["--list-types"]) } else { run("xclip", &["-selection", "clipboard", "-t", "TARGETS", "-o"]) }?;
        let kinds = String::from_utf8_lossy(&kinds).into_owned();
        let has = |kind: &str| kinds.lines().any(|l| l.trim() == kind);
        if has("text/uri-list")
            && let Some(list) = get("text/uri-list")
        {
            let files: Vec<std::path::PathBuf> = String::from_utf8_lossy(&list)
                .lines()
                .filter_map(|l| reqwest::Url::parse(l.trim()).ok())
                .filter_map(|u| u.to_file_path().ok())
                .collect();
            if !files.is_empty() {
                return Some(Clipped::Files(files));
            }
        }
        for (kind, mime) in [("image/png", "image/png"), ("image/jpeg", "image/jpeg")] {
            if has(kind) {
                return get(kind).filter(|b| !b.is_empty()).map(|b| Clipped::Image(mime, b));
            }
        }
        None
    }
}

/// macOS hands the clipboard to AppleScript as hexadecimal, which needs no extra tool.
#[cfg(target_os = "macos")]
mod clip {
    use super::Clipped;
    use std::process::{Command, Stdio};

    fn script(line: &str) -> Option<String> {
        let out = Command::new("osascript").args(["-e", line]).stdin(Stdio::null()).stderr(Stdio::null()).output().ok()?;
        out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
    }

    pub fn read() -> Option<Clipped> {
        if let Some(path) = script("POSIX path of (the clipboard as «class furl»)").filter(|p| p.starts_with('/')) {
            return Some(Clipped::Files(vec![std::path::PathBuf::from(path)]));
        }
        let data = script("get the clipboard as «class PNGf»")?;
        let hex = data.strip_prefix("«data PNGf")?.strip_suffix('»')?;
        Some(Clipped::Image("image/png", hex::decode(hex).ok()?))
    }
}

/// Free space on the drive that holds `path`, measured at the nearest folder that exists, since a models folder is
/// often created only when the first download starts.
pub fn free_space(path: &std::path::Path) -> Option<u64> {
    imp::free_space(path.ancestors().find(|p| p.exists())?)
}

/// Stops a process left behind by an earlier run, but only when it is still the named program.
pub fn kill_if_named(pid: u32, name: &str) -> bool {
    imp::kill_if_named(pid, name)
}

/// Rounds the corners of the window with this raw id, or squares them while it is maximized. Windows 11 rounds
/// them itself when asked. Windows 10 cannot, so the window is clipped to a rounded rectangle, which loses its
/// shadow, and a separate window behind it draws a rounded shadow instead. Other systems leave the window as it is.
pub fn round_corners(window: u64, maximized: bool) {
    #[cfg(windows)]
    imp::round_corners(window, maximized);
    #[cfg(not(windows))]
    let _ = (window, maximized);
}

/// Keeps the Windows 10 shadow behind the window after it moves or comes to the front.
pub fn follow_window(window: u64) {
    #[cfg(windows)]
    imp::shadow::follow(window);
    #[cfg(not(windows))]
    let _ = window;
}

/// The primary screen without the taskbar, as x, y, width, and height in logical pixels.
pub fn work_area() -> Option<(f32, f32, f32, f32)> {
    imp::work_area()
}

/// Gives this process the PATH of the user's login shell on macOS. An app opened from Finder or the Dock gets
/// only the system folders, so tools installed with Homebrew and similar would not be found. Call it before any
/// other thread starts, since it changes the environment.
pub fn use_login_path() {
    #[cfg(target_os = "macos")]
    {
        const MARK: &str = "SCOOBERT_PATH=";
        let shell = std::env::var("SHELL").ok().filter(|s| s.starts_with('/')).unwrap_or_else(|| "/bin/zsh".into());
        // A profile that prints text or waits would otherwise delay the window, so the answer gets 3 seconds.
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let out = std::process::Command::new(&shell).args(["-l", "-c", "printf 'SCOOBERT_PATH=%s' \"$PATH\""]).stdin(std::process::Stdio::null()).output();
            let _ = tx.send(out.ok().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()));
        });
        let printed = rx.recv_timeout(std::time::Duration::from_secs(3)).ok().flatten().unwrap_or_default();
        let path = printed.rsplit_once(MARK).map(|(_, p)| p.trim().to_string()).unwrap_or_default();
        let fallback = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin";
        let path = if path.contains("/usr/bin") { path } else { format!("{fallback}:{}", std::env::var("PATH").unwrap_or_default()) };
        // SAFETY: main calls this before the app starts its threads, and the helper thread above only waits on the
        // shell it already started.
        unsafe { std::env::set_var("PATH", path) };
    }
}

/// The language code the Windows installer wrote beside the program, if any.
pub fn installer_language() -> Option<String> {
    static FOUND: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    FOUND
        .get_or_init(|| {
            let file = std::env::current_exe().ok()?.parent()?.join("language.txt");
            Some(std::fs::read_to_string(file).ok()?.trim().to_string()).filter(|c| !c.is_empty())
        })
        .clone()
}

#[cfg(windows)]
mod imp {
    use windows_sys::Win32::Foundation::{CloseHandle, FALSE, RECT};
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_TERMINATE, QueryFullProcessImageNameW,
        TerminateProcess,
    };
    use windows_sys::Win32::UI::HiDpi::GetDpiForSystem;
    use windows_sys::Win32::UI::WindowsAndMessaging::{SPI_GETWORKAREA, SystemParametersInfoW};

    pub fn work_area() -> Option<(f32, f32, f32, f32)> {
        let mut r = RECT { left: 0, top: 0, right: 0, bottom: 0 };
        if unsafe { SystemParametersInfoW(SPI_GETWORKAREA, 0, (&mut r as *mut RECT).cast(), 0) } == 0 {
            return None;
        }
        // Both calls answer in the same units whether or not the process is DPI aware yet.
        let scale = unsafe { GetDpiForSystem() } as f32 / 96.0;
        Some((r.left as f32 / scale, r.top as f32 / scale, (r.right - r.left) as f32 / scale, (r.bottom - r.top) as f32 / scale))
    }

    pub fn round_corners(window: u64, maximized: bool) {
        use windows_sys::Win32::Graphics::Dwm::{DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND, DwmSetWindowAttribute};
        use windows_sys::Win32::Graphics::Gdi::{CreateRoundRectRgn, SetWindowRgn};
        use windows_sys::Win32::UI::HiDpi::GetDpiForWindow;
        use windows_sys::Win32::UI::WindowsAndMessaging::GetWindowRect;
        // iced's raw window id on Windows is the window handle.
        let hwnd = window as usize as windows_sys::Win32::Foundation::HWND;
        unsafe {
            let round = DWMWCP_ROUND;
            let size = std::mem::size_of_val(&round) as u32;
            if DwmSetWindowAttribute(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE as _, (&round as *const i32).cast(), size) == 0 {
                return;
            }
            if maximized {
                SetWindowRgn(hwnd, std::ptr::null_mut(), 1);
                shadow::place(window, false);
                return;
            }
            let mut r = RECT { left: 0, top: 0, right: 0, bottom: 0 };
            if GetWindowRect(hwnd, &mut r) == 0 {
                return;
            }
            // The 8-pixel radius Windows 11 uses, as a diameter at the window's scale.
            let diameter = (2 * shadow::RADIUS * GetDpiForWindow(hwnd).max(96) / 96) as i32;
            let region = CreateRoundRectRgn(0, 0, r.right - r.left + 1, r.bottom - r.top + 1, diameter, diameter);
            // The system owns the region once it is set.
            SetWindowRgn(hwnd, region, 1);
            shadow::place(window, true);
        }
    }

    /// A click-through window that sits right behind Scoobert's on Windows 10 and draws a soft shadow with the same
    /// rounded corners, since clipping the window to its rounded shape removes the shadow Windows draws.
    pub mod shadow {
        use std::sync::Mutex;

        use windows_sys::Win32::Foundation::{HWND, POINT, RECT, SIZE};
        use windows_sys::Win32::Graphics::Gdi::{
            AC_SRC_ALPHA, AC_SRC_OVER, BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BLENDFUNCTION, CreateCompatibleDC, CreateDIBSection, DIB_RGB_COLORS,
            DeleteDC, DeleteObject, GetDC, ReleaseDC, SelectObject,
        };
        use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
        use windows_sys::Win32::UI::HiDpi::GetDpiForWindow;
        use windows_sys::Win32::UI::WindowsAndMessaging::{
            CreateWindowExW, DefWindowProcW, GetWindowRect, IsIconic, IsWindowVisible, RegisterClassExW, SW_HIDE, SWP_NOACTIVATE, SWP_NOSIZE,
            SWP_SHOWWINDOW, SetWindowPos, ShowWindow, ULW_ALPHA, UpdateLayeredWindow, WNDCLASSEXW, WS_EX_LAYERED, WS_EX_NOACTIVATE,
            WS_EX_TOOLWINDOW, WS_EX_TRANSPARENT, WS_POPUP,
        };

        /// The corner radius, in pixels at 100% scale.
        pub const RADIUS: u32 = 8;
        /// How far the shadow reaches past the window, and how far it drops below it, at 100% scale.
        const REACH: i32 = 16;
        const DROP: i32 = 2;
        /// The shadow's darkness right at the window's edge, out of 255.
        const DARKEST: f32 = 70.0;

        struct Shadow {
            /// The shadow window, kept as a number so the state can live in a static.
            hwnd: usize,
            size: (i32, i32),
            maximized: bool,
        }

        static SHADOW: Mutex<Option<Shadow>> = Mutex::new(None);

        /// Shows the shadow behind the window, or hides it while the window is maximized.
        pub fn place(window: u64, show: bool) {
            let mut state = SHADOW.lock().unwrap();
            if state.is_none() {
                if !show {
                    return;
                }
                let Some(hwnd) = create() else { return };
                *state = Some(Shadow { hwnd, size: (0, 0), maximized: false });
            }
            let Some(s) = state.as_mut() else { return };
            s.maximized = !show;
            update(s, window);
        }

        /// Moves the shadow with the window. Does nothing until the window has one.
        pub fn follow(window: u64) {
            if let Some(s) = SHADOW.lock().unwrap().as_mut() {
                update(s, window);
            }
        }

        fn update(s: &mut Shadow, window: u64) {
            let main = window as usize as HWND;
            let shadow = s.hwnd as HWND;
            unsafe {
                if s.maximized || IsIconic(main) != 0 || IsWindowVisible(main) == 0 {
                    ShowWindow(shadow, SW_HIDE);
                    return;
                }
                let mut r = RECT { left: 0, top: 0, right: 0, bottom: 0 };
                if GetWindowRect(main, &mut r) == 0 {
                    return;
                }
                let scale = GetDpiForWindow(main).max(96) as f32 / 96.0;
                let reach = (REACH as f32 * scale).round() as i32;
                let (w, h) = (r.right - r.left, r.bottom - r.top);
                let at = POINT { x: r.left - reach, y: r.top - reach };
                if s.size != (w, h) && draw(shadow, at, w, h, reach, scale) {
                    s.size = (w, h);
                }
                // Right below the window in the stacking order, so nothing else comes between them.
                SetWindowPos(shadow, main, at.x, at.y, 0, 0, SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW);
            }
        }

        fn create() -> Option<usize> {
            let class: Vec<u16> = "ScoobertShadow\0".encode_utf16().collect();
            unsafe {
                let instance = GetModuleHandleW(std::ptr::null());
                let wc = WNDCLASSEXW {
                    cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                    style: 0,
                    lpfnWndProc: Some(DefWindowProcW),
                    cbClsExtra: 0,
                    cbWndExtra: 0,
                    hInstance: instance,
                    hIcon: std::ptr::null_mut(),
                    hCursor: std::ptr::null_mut(),
                    hbrBackground: std::ptr::null_mut(),
                    lpszMenuName: std::ptr::null(),
                    lpszClassName: class.as_ptr(),
                    hIconSm: std::ptr::null_mut(),
                };
                RegisterClassExW(&wc);
                // Layered for its alpha, transparent to clicks, never activated, and kept off the taskbar.
                let hwnd = CreateWindowExW(
                    WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
                    class.as_ptr(),
                    std::ptr::null(),
                    WS_POPUP,
                    0,
                    0,
                    0,
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    instance,
                    std::ptr::null(),
                );
                (!hwnd.is_null()).then_some(hwnd as usize)
            }
        }

        /// Draws the shadow for a window of `w` by `h` into the layered window at `at`.
        fn draw(shadow: HWND, at: POINT, w: i32, h: i32, reach: i32, scale: f32) -> bool {
            let (bw, bh) = (w + 2 * reach, h + 2 * reach);
            let pixels = pixels(w, h, reach, RADIUS as f32 * scale, DROP as f32 * scale);
            unsafe {
                let screen = GetDC(std::ptr::null_mut());
                let dc = CreateCompatibleDC(screen);
                let mut info: BITMAPINFO = std::mem::zeroed();
                info.bmiHeader = BITMAPINFOHEADER {
                    biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: bw,
                    // A negative height stores the rows from the top.
                    biHeight: -bh,
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB as u32,
                    biSizeImage: 0,
                    biXPelsPerMeter: 0,
                    biYPelsPerMeter: 0,
                    biClrUsed: 0,
                    biClrImportant: 0,
                };
                let mut bits: *mut core::ffi::c_void = std::ptr::null_mut();
                let bitmap = CreateDIBSection(dc, &info, DIB_RGB_COLORS, &mut bits, std::ptr::null_mut(), 0);
                let mut ok = false;
                if !bitmap.is_null() && !bits.is_null() {
                    std::ptr::copy_nonoverlapping(pixels.as_ptr(), bits.cast::<u32>(), pixels.len());
                    let old = SelectObject(dc, bitmap);
                    let size = SIZE { cx: bw, cy: bh };
                    let origin = POINT { x: 0, y: 0 };
                    let blend = BLENDFUNCTION { BlendOp: AC_SRC_OVER as u8, BlendFlags: 0, SourceConstantAlpha: 255, AlphaFormat: AC_SRC_ALPHA as u8 };
                    ok = UpdateLayeredWindow(shadow, screen, &at, &size, dc, &origin, 0, &blend, ULW_ALPHA) != 0;
                    SelectObject(dc, old);
                    DeleteObject(bitmap);
                }
                DeleteDC(dc);
                ReleaseDC(std::ptr::null_mut(), screen);
                ok
            }
        }

        /// Premultiplied black with an alpha that fades from the window's rounded edge out to `reach`.
        pub(super) fn pixels(w: i32, h: i32, reach: i32, radius: f32, drop: f32) -> Vec<u32> {
            let (bw, bh) = (w + 2 * reach, h + 2 * reach);
            let (half_w, half_h) = (w as f32 / 2.0, h as f32 / 2.0);
            let mut out = Vec::with_capacity((bw * bh) as usize);
            for py in 0..bh {
                let y = py as f32 + 0.5 - reach as f32 - drop - half_h;
                for px in 0..bw {
                    let x = px as f32 + 0.5 - reach as f32 - half_w;
                    // Signed distance to the rounded rectangle: negative inside, positive outside.
                    let qx = x.abs() - (half_w - radius);
                    let qy = y.abs() - (half_h - radius);
                    let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
                    let d = outside + qx.max(qy).min(0.0) - radius;
                    let t = 1.0 - (d / reach as f32).clamp(0.0, 1.0);
                    out.push(((DARKEST * t * t) as u32) << 24);
                }
            }
            out
        }
    }

    pub fn memory() -> (u64, u64) {
        let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
            return (0, 0);
        }
        (status.ullTotalPhys, status.ullAvailPhys)
    }

    pub fn free_space(path: &std::path::Path) -> Option<u64> {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        let mut free = 0u64;
        (unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &mut free, std::ptr::null_mut(), std::ptr::null_mut()) } != 0).then_some(free)
    }

    pub fn kill_if_named(pid: u32, name: &str) -> bool {
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE, FALSE, pid);
            if handle.is_null() {
                return false;
            }
            let mut buf = [0u16; 1024];
            let mut len = buf.len() as u32;
            let ok = QueryFullProcessImageNameW(handle, PROCESS_NAME_WIN32, buf.as_mut_ptr(), &mut len) != 0;
            let image = String::from_utf16_lossy(&buf[..len as usize]).to_lowercase();
            let killed = ok && image.ends_with(&name.to_lowercase()) && TerminateProcess(handle, 1) != 0;
            CloseHandle(handle);
            killed
        }
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use std::process::Command;

    /// macOS keeps a new window inside the visible screen area itself.
    pub fn work_area() -> Option<(f32, f32, f32, f32)> {
        None
    }

    /// Total from `sysctl hw.memsize`, and available as Activity Monitor counts it: everything except wired memory,
    /// the compressor, and app memory. The file cache counts as available, since macOS drops it for a new process.
    pub fn memory() -> (u64, u64) {
        let run = |cmd: &str, args: &[&str]| Command::new(cmd).args(args).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).into_owned());
        let total = run("/usr/sbin/sysctl", &["-n", "hw.memsize"]).and_then(|t| t.trim().parse::<u64>().ok()).unwrap_or(0);
        let Some(stats) = run("/usr/bin/vm_stat", &[]) else { return (total, 0) };
        (total, available_from_vm_stat(&stats, total))
    }

    pub fn available_from_vm_stat(stats: &str, total: u64) -> u64 {
        let page = stats
            .lines()
            .next()
            .and_then(|l| l.split("page size of").nth(1))
            .and_then(|r| r.split_whitespace().next())
            .and_then(|n| n.parse::<u64>().ok())
            .unwrap_or(16384);
        let pages = |key: &str| {
            stats
                .lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split(':').nth(1))
                .and_then(|v| v.trim().trim_end_matches('.').parse::<u64>().ok())
                .unwrap_or(0)
        };
        let used = pages("Pages wired down") + pages("Pages occupied by compressor") + pages("Anonymous pages").saturating_sub(pages("Pages purgeable"));
        let unused = (pages("Pages free") + pages("Pages inactive") + pages("Pages speculative") + pages("Pages purgeable")) * page;
        // Older vm_stat output has no anonymous pages, and then only the unused pages are certain.
        if pages("Anonymous pages") == 0 { unused } else { total.saturating_sub(used * page).max(unused) }
    }

    /// Read from the POSIX df output, whose fourth column is the space available in kilobytes.
    pub fn free_space(path: &std::path::Path) -> Option<u64> {
        let out = Command::new("/bin/df").arg("-Pk").arg(path).output().ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        text.lines().nth(1)?.split_whitespace().nth(3)?.parse::<u64>().ok().map(|kb| kb * 1024)
    }

    /// `ps` gives the full path of the program, which has to end with `name`.
    pub fn kill_if_named(pid: u32, name: &str) -> bool {
        let out = Command::new("/bin/ps").args(["-p", &pid.to_string(), "-o", "comm="]).output();
        let program = out.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
        if program.rsplit('/').next() != Some(name) {
            return false;
        }
        Command::new("/bin/kill").arg(pid.to_string()).status().map(|s| s.success()).unwrap_or(false)
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn reads_available_pages_from_vm_stat() {
            let page = 16384u64;
            let stats = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:                               1000.\nPages active:                             5000.\nPages inactive:                           2000.\nPages speculative:                         300.\nPages wired down:                         4000.\nPages purgeable:                           100.\nFile-backed pages:                        3000.\nAnonymous pages:                          4100.\nPages occupied by compressor:              500.\n";
            // 12,800 pages in all, minus 4,000 wired, 500 compressed, and 4,000 app pages (anonymous less purgeable).
            assert_eq!(super::available_from_vm_stat(stats, 12_800 * page), 4300 * page);
            let (total, available) = super::memory();
            assert!(total > 0 && available > 0 && available <= total);
        }
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
mod imp {
    /// Linux desktops keep a new window inside the work area themselves.
    pub fn work_area() -> Option<(f32, f32, f32, f32)> {
        None
    }

    pub fn memory() -> (u64, u64) {
        let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let field = |key: &str| {
            text.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
                .map(|kb| kb * 1024)
                .unwrap_or(0)
        };
        (field("MemTotal:"), field("MemAvailable:"))
    }

    /// Read from the POSIX df output, whose fourth column is the space available in kilobytes.
    pub fn free_space(path: &std::path::Path) -> Option<u64> {
        let out = std::process::Command::new("df").arg("-Pk").arg(path).output().ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        text.lines().nth(1)?.split_whitespace().nth(3)?.parse::<u64>().ok().map(|kb| kb * 1024)
    }

    pub fn kill_if_named(pid: u32, name: &str) -> bool {
        let comm = std::fs::read_to_string(format!("/proc/{pid}/comm")).unwrap_or_default();
        // The kernel truncates comm to 15 characters.
        let short: String = name.chars().take(15).collect();
        if comm.trim() != short {
            return false;
        }
        std::process::Command::new("kill").arg(pid.to_string()).status().map(|s| s.success()).unwrap_or(false)
    }
}
