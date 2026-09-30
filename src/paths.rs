use std::path::PathBuf;
use std::sync::OnceLock;

use directories::{BaseDirs, ProjectDirs};

/// Folders Scoobert keeps its files in. `SCOOBERT_HOME` puts all of them under one folder, for tests.
pub struct Paths {
    pub config: PathBuf,
    pub data: PathBuf,
    pub cache: PathBuf,
    pub home: PathBuf,
}

pub fn get() -> &'static Paths {
    static PATHS: OnceLock<Paths> = OnceLock::new();
    PATHS.get_or_init(|| {
        let home = BaseDirs::new().map(|b| b.home_dir().to_path_buf()).unwrap_or_default();
        let paths = match std::env::var_os("SCOOBERT_HOME") {
            Some(root) => {
                let root = PathBuf::from(root);
                Paths { config: root.join("config"), data: root.join("data"), cache: root.join("cache"), home }
            }
            None => {
                let dirs = ProjectDirs::from("", "", "Scoobert").expect("the user has a home folder");
                Paths {
                    config: dirs.config_dir().to_path_buf(),
                    data: dirs.data_dir().to_path_buf(),
                    cache: dirs.cache_dir().to_path_buf(),
                    home,
                }
            }
        };
        for dir in [&paths.config, &paths.data, &paths.cache] {
            let _ = std::fs::create_dir_all(dir);
        }
        paths
    })
}

impl Paths {
    pub fn state_file(&self) -> PathBuf {
        self.config.join("state.json")
    }

    pub fn sessions(&self) -> PathBuf {
        self.data.join("sessions")
    }

    pub fn slots(&self) -> PathBuf {
        self.cache.join("slots")
    }

    pub fn server_log(&self) -> PathBuf {
        self.data.join("llama-server.log")
    }

    pub fn pid_file(&self) -> PathBuf {
        self.data.join("llama-server.pid")
    }

    pub fn default_models_dir(&self) -> PathBuf {
        self.home.join("models")
    }
}

/// Folder that holds the bundled llama.cpp build, if this install has one.
pub fn bundled_llama_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let candidates = [dir.join("llama"), dir.join("../lib/scoobert/llama"), dir.join("../share/scoobert/llama")];
    candidates.into_iter().find(|c| c.join(llama_server_name()).is_file())
}

pub fn llama_server_name() -> &'static str {
    if cfg!(windows) { "llama-server.exe" } else { "llama-server" }
}

/// Forward slashes on every platform, for paths shown to the model and the user.
pub fn display(path: &std::path::Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}
