//! API keys for hosted providers, kept in the system credential store rather than in the state file.

const SERVICE: &str = "Scoobert";

fn entry(provider: &str) -> anyhow::Result<keyring::Entry> {
    keyring::Entry::new(SERVICE, provider).map_err(|err| anyhow::anyhow!("The system credential store is unavailable: {err}"))
}

pub fn get(provider: &str) -> Option<String> {
    entry(provider).ok()?.get_password().ok().filter(|k| !k.is_empty())
}

pub fn set(provider: &str, key: &str) -> anyhow::Result<()> {
    entry(provider)?.set_password(key).map_err(|err| anyhow::anyhow!("Could not save the key: {err}"))
}

pub fn remove(provider: &str) -> anyhow::Result<()> {
    match entry(provider)?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(err) => Err(anyhow::anyhow!("Could not remove the key: {err}")),
    }
}
