//! Memory and process queries that differ between Windows, Linux, and macOS.

/// Physical memory the system can give to a new process without paging, in bytes.
pub fn available_memory() -> u64 {
    imp::memory().1
}

pub fn total_memory() -> u64 {
    imp::memory().0
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
/// them itself when asked. Windows 10 cannot, so the window is clipped to a rounded rectangle, which also hides
/// its shadow. Other systems leave the window as it is.
pub fn round_corners(window: u64, maximized: bool) {
    #[cfg(windows)]
    imp::round_corners(window, maximized);
    #[cfg(not(windows))]
    let _ = (window, maximized);
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
                return;
            }
            let mut r = RECT { left: 0, top: 0, right: 0, bottom: 0 };
            if GetWindowRect(hwnd, &mut r) == 0 {
                return;
            }
            // The 8-pixel radius Windows 11 uses, as a diameter at the window's scale.
            let diameter = (16 * GetDpiForWindow(hwnd).max(96) / 96) as i32;
            let region = CreateRoundRectRgn(0, 0, r.right - r.left + 1, r.bottom - r.top + 1, diameter, diameter);
            // The system owns the region once it is set.
            SetWindowRgn(hwnd, region, 1);
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

    /// Total from `sysctl hw.memsize`, and available as the free, inactive, speculative, and purgeable pages that
    /// `vm_stat` reports, which macOS hands to a new process before it compresses or swaps anything.
    pub fn memory() -> (u64, u64) {
        let run = |cmd: &str, args: &[&str]| Command::new(cmd).args(args).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).into_owned());
        let total = run("/usr/sbin/sysctl", &["-n", "hw.memsize"]).and_then(|t| t.trim().parse::<u64>().ok()).unwrap_or(0);
        let Some(stats) = run("/usr/bin/vm_stat", &[]) else { return (total, 0) };
        (total, available_from_vm_stat(&stats))
    }

    pub fn available_from_vm_stat(stats: &str) -> u64 {
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
        (pages("Pages free") + pages("Pages inactive") + pages("Pages speculative") + pages("Pages purgeable")) * page
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
            let stats = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:                               1000.\nPages active:                             5000.\nPages inactive:                           2000.\nPages speculative:                         300.\nPages purgeable:                           100.\n";
            assert_eq!(super::available_from_vm_stat(stats), 3400 * 16384);
            assert!(super::memory().0 > 0);
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
