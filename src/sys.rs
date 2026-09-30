//! Memory and process queries that differ between Windows and Linux.

/// Physical memory the system can give to a new process without paging, in bytes.
pub fn available_memory() -> u64 {
    imp::memory().1
}

pub fn total_memory() -> u64 {
    imp::memory().0
}

/// Stops a process left behind by an earlier run, but only when it is still the named program.
pub fn kill_if_named(pid: u32, name: &str) -> bool {
    imp::kill_if_named(pid, name)
}

#[cfg(windows)]
mod imp {
    use windows_sys::Win32::Foundation::{CloseHandle, FALSE};
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_TERMINATE, QueryFullProcessImageNameW,
        TerminateProcess,
    };

    pub fn memory() -> (u64, u64) {
        let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
            return (0, 0);
        }
        (status.ullTotalPhys, status.ullAvailPhys)
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
