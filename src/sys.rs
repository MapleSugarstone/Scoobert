//! Memory and process queries that differ between Windows and Linux.

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

/// The primary screen without the taskbar, as x, y, width, and height in logical pixels.
pub fn work_area() -> Option<(f32, f32, f32, f32)> {
    imp::work_area()
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

#[cfg(not(windows))]
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
