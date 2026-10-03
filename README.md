# Scoobert

Scoobert is an AI coding assistant for Windows, macOS, and Linux that runs Qwen models on your own computer through [llama.cpp](https://github.com/ggml-org/llama.cpp). Your code and conversations stay on your machine. Each project gets its own conversation history and a folder of linked Markdown notes that Scoobert uses as its memory. If your computer is too slow for a local model, you can use a hosted model with your own API key instead.

Scoobert is a single native program written in Rust. It needs no browser engine or runtime, which leaves more of your memory for the model.

## Download and install

Download one file for your system from the [Releases](../../releases) page.

### Windows

1. Download `Scoobert-Setup-<version>.exe` and open it.
2. If Windows says it protected your PC, select **More info**, then **Run anyway**. The warning appears because the installer is not code-signed.
3. Select **Next**, keep the suggested folder or pick another with **Browse**, and select **Install**. You don't need an administrator password.
4. Open Scoobert from the shortcut on your desktop or in the Start menu.

To remove Scoobert, open **Settings** in Scoobert and select **Uninstall Scoobert**, or find Scoobert under **Installed apps** in Windows Settings. The uninstaller asks whether to also remove your conversations and the downloaded models. The notes in your projects always stay.

**No-install option:** download `Scoobert-<version>-portable-windows.zip`, right-click it, select **Extract All**, and open `scoobert.exe` inside the folder. The portable copy keeps its settings, conversations, and models inside its own folder, so you can put it anywhere, even on a USB drive. To remove it, delete the folder.

### Linux

1. Download `Scoobert-<version>-x86_64.AppImage`.
2. Right-click it, open **Properties**, and turn on **Allow executing file as program** (or run `chmod +x Scoobert-*.AppImage`).
3. Double-click it. On Bazzite and other immutable systems, the Gear Lever app can add it to your app menu.

To remove Scoobert, delete the AppImage file. The `.tar.gz` holds the same program as a portable folder that keeps its data inside it, like the Windows zip.

### macOS

1. Download `Scoobert-<version>-macos-arm64.dmg` and open it.
2. Drag **Scoobert** onto the **Applications** folder in the window that opens.
3. Open Scoobert from Applications. macOS says it could not verify the app, because Scoobert is not notarized by Apple. Select **Done**.
4. Open **System Settings**, then **Privacy & Security**, scroll down, and select **Open Anyway** next to the message about Scoobert. Confirm with your password. After this, Scoobert opens normally.

If **Open Anyway** does not appear, run `xattr -dr com.apple.quarantine /Applications/Scoobert.app` in Terminal, then open Scoobert again.

To remove Scoobert, drag it from Applications to the Trash. Your settings and conversations stay in `~/Library/Application Support/Scoobert`, its saved prompts in `~/Library/Caches/Scoobert`, and the models in `~/models`, until you delete those folders.

On first launch, choose which models to download. Scoobert preselects the ones that fit your computer's memory.

### What your computer needs

- Windows 10 or 11, a 64-bit Linux desktop from 2022 or later, or a Mac with Apple Silicon (M1 or later). On a Mac the models run on the graphics chip, which shares the Mac's memory.
- 16 GB of RAM for the fast model, and 24 GB or more for the larger ones. Scoobert lists each model's memory needs before you download it.
- Free disk space for the models: 6.6 GB for the fast model, and 14 to 83 GB for the larger ones.

On Windows, Scoobert runs shell commands in Git Bash when [Git for Windows](https://git-scm.com/download/win) is installed, and in PowerShell otherwise.

## Models

| Model | Download | Free memory needed | Best for |
|---|---|---|---|
| Qwen3.5 9B | 6.6 GB | About 7 GB | Quick answers and screenshots on any recent laptop |
| Qwen3.8 27B | 14.3 GB | About 15 GB at its default 32K context | Better code on computers with 24 GB of RAM |
| Qwen3.8 27B, high precision | 22.0 GB | About 21 GB | Near full quality on computers with 32 GB of RAM or more |
| Qwen3.8 Flash-Next 125B | 82.9 GB | About 75 GB | The best local Qwen model, for workstations with 96 GB of RAM or more (not yet tested with Scoobert) |
| Qwen3.8 27B, compact | 10.9 GB | About 12 GB | The 27B at 3-bit, which fits entirely on a graphics card with 12 to 16 GB and writes several times faster there |
| Qwen3.6 35B-A3B | 19.1 GB | About 19 GB | A mixture-of-experts model that uses about 3B parameters per word, so it writes fast even on a CPU, and reads images |
| Qwen3.6 35B-A3B, compact | 15.0 GB | About 15 GB | The 35B-A3B at 3-bit, for computers with 16 GB of RAM |
| Qwen3.5 0.8B | 0.5 GB | Under 1 GB | A draft model that speeds up a larger Qwen model through Predict ahead |

The models run on the CPU. On a laptop, the 9B model writes about four words per second and the 27B model about one. For long unattended coding, the 35B-A3B with **Predict ahead** set to its own prediction layers passed as many tasks in Scoobert's coding benchmark as the 27B, in a third of the time. Scoobert checks free memory before it loads a model, and when there isn't enough, it asks you to close other apps instead of letting the system swap to disk.

To add a model later, open **Settings**, then **Models on this computer**. You can also download any GGUF model from Hugging Face there: paste the model's web address, or write `owner/repository`, and add a size after a colon to pick one, as in `unsloth/Qwen3.5-9B-GGUF:Q4_K_M`. When the repository has several sizes and you named none, Scoobert shows every size with its download size so you can pick one. Scoobert checks every downloaded file against the SHA-256 checksum Hugging Face publishes for it.

To make a model write faster, pick a way under **Predict ahead** on its card. The model checks several predicted words in one step and keeps the ones it agrees with, so its replies stay the same. **Its own prediction layers** uses layers that the files of the 27B and the 35B-A3B include. On a laptop processor they made the 27B write 1.4 to 2.3 times as fast and the 35B-A3B 1.45 times as fast. **Text already in the conversation** helps only when the model repeats long stretches of text it has seen, and it made a short edit 5% faster. **Draft with** a smaller model of the same family, such as Qwen3.5 0.8B for the 9B, lets the small model draft while the large one checks. If a model cannot load with the choice, Scoobert runs it without and says so. **Compact context memory** keeps the context at 8 bits, which halves its memory so more of a model fits on the graphics card.

A GGUF model you downloaded yourself works too: select **Choose a model file** under **Add a model file**. Scoobert runs it from its folder, reads the context length it was trained for from the file, and lets it read images when an image projector (a file with `mmproj` in its name) sits beside it. **Remove from list** takes it out of Scoobert without deleting the file.

The trash button on a downloaded model deletes it after asking. That deletes its files, its image projector, the model lab variants made from it, and the prompts Scoobert saved for them, so the disk space comes back at once. Deleting a variant in the model lab works the same way.

### Hosted models

Open **Settings**, then **Hosted models**, pick a provider, paste your API key, and tick the models to add to the Model menu. Scoobert supports OpenRouter, Anthropic, OpenAI, Google Gemini, Qwen on Alibaba Cloud, DeepSeek, Mistral, xAI, Groq, Together AI, Fireworks AI, and Cerebras. You can also add any server that accepts OpenAI-style chat requests by its address.

Hosted models run on the provider's servers, so Scoobert sends them your prompts, the files it reads, and command output, and the provider bills you for use. Scoobert keeps keys in your system's credential store (Credential Manager on Windows, the Secret Service keyring on Linux).

## Features

- **Local assistant:** Scoobert reads files, edits code, runs commands and tests, and fixes what fails, in the project folder you open. It needs no account or internet connection once a model is downloaded.
- **Start without a project:** **New chat** starts a conversation with no project, so you can ask Scoobert anything. These conversations are listed under **Chats**, the first entry in the project list. When a task needs its own files, Scoobert starts a project folder in `Documents\Scoobert` (`~/Documents/Scoobert` on Linux) and moves the conversation into it. The **+** button that appears when you point at a project starts a conversation in it, and the search box above the list finds projects and conversations by name.
- **Projects and conversations:** add any folder as a project. Scoobert reopens the last project and conversation when it starts, and names each conversation after its first task.
- **Past conversations on request:** Scoobert only looks at your other conversations when you ask about them ("what did we decide last time?"). Then it lists the project's conversations and reads the one that matters.
- **Web research (off until you turn it on):** with **Let Scoobert search the web** on in Settings, Scoobert searches DuckDuckGo, skims the promising results, and reads the best pages in full. Pages built by JavaScript are rendered in the browser already on your computer (Edge on Windows, Chrome or Chromium on Linux) with a throwaway profile. When web search is off and you ask for something that needs it, Scoobert offers to turn it on first. No account or key is needed.
- **Three approval modes:** **Ask before changes** shows every edit as a diff and every command before it runs. **Work unattended in the project** is for leaving a task running: edits inside the project run without asking, edits elsewhere are refused, and commands cannot use the internet (see [Unattended work](#unattended-work)). **Allow everything** runs anything without asking.
- **Long tasks:** when a conversation fills the model's context, Scoobert has the model summarize the older messages and continues from the summary. Each summary is also saved in the project's `Notes/Tasks` folder, so you can read how a long task went.
- **Notes as memory:** each project keeps Markdown notes in its `Notes` folder, linked with `[[wikilinks]]`, and Scoobert does the remembering itself rather than relying on the model:
  - Each conversation starts with a short index of the notes (a line about each) and the most recent work from the task log.
  - Each message gets the matching part of the best-matching note attached, found with a keyword search that weighs rare words and note titles. When no note clearly matches, nothing is attached.
  - The model reads any note by name (`[[Auth design]]`), and each note it reads shows where its links lead and which notes link back, so it can follow a chain of links.
  - After a task, Scoobert logs it in that day's note and asks the model for up to three facts labeled as a decision, a convention, or a problem. Scoobert checks them, drops repeats, and files them in the related note or in `Decisions.md`, `Conventions.md`, or `Problems.md`, then shows what it saved.
- **Notes pane:** read and edit notes beside the conversation. The pane shows which notes link to the open note, searches their text, and draws a graph of the links. You can save any reply as a note. The notes are plain Markdown files, so any Markdown note app can open the same folder.
- **Saved prompt cache:** Scoobert saves each conversation's processed prompt to disk, so returning to a long conversation takes seconds instead of the minutes a CPU needs to reread it. It also saves the prompt every new conversation starts with, for Chats and for each project, so a new conversation reads little more than your message. The first time Scoobert runs, and after an update, it loads your default model and builds these saved prompts in the background. The sidebar shows **Preparing** while it works.
- **Testing in a browser:** Scoobert can start a server in the background, such as `npm run dev`, and test the page or game it serves in a hidden browser. It opens the page, reads its text and controls, clicks, types, presses keys (and holds them, for games), runs scripts in the page, reads console errors, and takes screenshots, which appear in the conversation. The browser is the Edge, Chrome, or Chromium already on your computer, with a throwaway profile that has none of your sign-ins, and it closes when the task ends. Pages on your computer and files always open. Other sites open only when web search is on. A server the model leaves running shows above the message box with a **Stop** button, and it stops when you close the conversation.
- **Retry and ratings:** the newest reply has **Retry**, **Bad, retry**, and **Good**. Retry asks again without rerunning the tools before the reply. The other two save the reply as a bad or good example for learning.
- **Learning from ratings:** turn on **Learn from ratings** for a local model in Settings, rate a few replies each way, and select **Apply ratings**. Scoobert compares the good replies with the bad ones the same way the model lab's steering compares two personas, and the model leans toward the good ones from then on, at the strength you pick. It changes tone and style rather than knowledge, and **Reset** forgets the ratings and what was learned. Applying takes a few minutes on the 9B and longer on the 27B, and the model unloads meanwhile.
- **Files and pasting:** the paperclip button attaches any file, and Ctrl+V (Cmd+V on a Mac) in the message box pastes an image or files copied in the file manager. On Linux, pasting images needs `wl-paste` (Wayland) or `xclip` (X11). Other files go in as their path, which the model reads.
- **Images:** attach screenshots and pictures to a message. A model that cannot see images, such as the 27B, gets them described by one that can, such as the 9B, and the same goes for browser screenshots. When memory allows, the describing model runs beside the conversation's model. Otherwise the conversation's model saves what it has read, steps aside while the image is described, and loads again.
- **Model lab:** select **Model lab** on a local model in Settings to see its layers, number formats, and every tensor and setting. You can make a variant there that appears in the Model menu under a name you choose. Scoobert checks for disk space first.
  - **Steering** measures how the model's layers respond to a persona you describe and adds that difference while the variant runs. You can change its strength later. The persona boxes grow as you type, and a few paragraphs with example replies in the voice you want work better than one line.
  - **Adapter** adds a LoRA adapter in GGUF format from Hugging Face or a file.
  - **Layer strength** turns the attention or feed-forward output of chosen layers up or down.
  - **Layer surgery** repeats or removes chosen layers.
  - **Size conversion** writes a copy in a smaller or larger number format.
- **Benchmark:** the model lab also tests the model. It gives the model a normal set of 10 or a hard set of 8 Python coding tasks with the settings the model has now, such as the graphics card and **Predict ahead**, and runs hidden tests on each answer. With more than one try per task, a model whose code fails sees the test output and writes the code again, which shows how well it fixes its own mistakes. Every model's results stay listed together for comparison. Checking answers needs Python 3, and the tests run the code each model writes on your computer. Conversations wait while a benchmark runs.

## Use Scoobert

1. Select **Add project** and choose the folder you want to work in.
2. Type in the **Ask Scoobert** box at the bottom and press Enter. Press Shift+Enter for a new line.
3. Watch Scoobert work. Each file it reads or changes appears as a line you can expand. Press Esc or select **Stop** to interrupt it.

The first reply after you install Scoobert or switch models takes a minute or two, because the model reads its instructions for the first time. Scoobert saves that work to disk, so later conversations start soon after the model loads.

The **Thinking** menu lets the model reason before it answers. Qwen models can only turn thinking on or off, so Scoobert sets the effort by capping how many tokens the model spends thinking:

| Thinking | Local token cap | Extra wait with the 9B | Extra wait with the 27B |
|---|---|---|---|
| Off | 0 | None | None |
| Low | 256 | Up to about 1 minute | Up to about 4 minutes |
| Medium | 1,024 | Up to about 4 minutes | Up to about 15 minutes |
| High | 4,096 | Up to about 15 minutes | Up to about an hour |

## Unattended work

In **Work unattended in the project** mode, Scoobert keeps working without anyone to approve changes:

- Edits and new files inside the project folder go ahead, and anything outside it is refused. The model is told why and carries on without it.
- On Linux, each command runs in a sandbox (bubblewrap) with no internet access, a read-only system, and only the project folder writable. Most desktop Linux systems include bubblewrap, because Flatpak uses it.
- On Windows, where one program cannot be cut off from the internet without administrator rights, Scoobert refuses commands that download or install software, such as `curl`, `git clone`, `pip install`, and `npm install`. Other commands run normally and can still change files outside the project.
- Every command, in every mode, gets a memory cap based on the memory that is free when it starts, so a runaway build cannot freeze the computer.

Tasks that need new packages from the internet fail in this mode, so install a project's dependencies before you leave it running.

## Settings

| Setting | What it does |
|---|---|
| Appearance | Light, dark, or matching the system. |
| When Scoobert makes changes | The approval mode. |
| Notes folder | The folder inside each project where notes live. |
| Log finished tasks | Adds each finished task to that day's note. |
| Ask the model what to remember | After a task, a local model lists up to three facts for the related note. |
| Context, per model | How many tokens of conversation a local model holds. Larger contexts need more memory. |
| System prompt, per model | The instructions every conversation on that model starts with. **Reset to Scoobert's prompt** restores the original. |
| Unload the model after | How long an idle model keeps its memory. |
| Load the model when Scoobert starts | Loads your default model at startup, so the first reply does not wait for it. Off by default, because the model then holds its memory from the start. |
| Where models live | The folder Scoobert looks in for GGUF models. |
| NVIDIA support | Shown on a computer with an NVIDIA card. Downloads llama.cpp's CUDA build and NVIDIA's runtime (about 550 MB), which usually reads prompts much faster than the bundled graphics support. Scoobert falls back to the bundled support if CUDA cannot load a model. |

## Where Scoobert keeps data

| | Windows | Linux |
|---|---|---|
| Settings and project list | `%APPDATA%\Scoobert\config\state.json` | `~/.config/scoobert/state.json` |
| Conversations | `%APPDATA%\Scoobert\data\sessions` | `~/.local/share/scoobert/sessions` |
| Saved prompt caches (up to 4 GB) | `%LOCALAPPDATA%\Scoobert\cache\slots` | `~/.cache/scoobert/slots` |
| Models | `%USERPROFILE%\models` | `~/models` |
| Notes | `Notes` inside each project | `Notes` inside each project |

The uninstaller always removes the saved prompt caches and asks about the settings, conversations, and downloaded models. A portable copy keeps all of these in its own folder (in `data` and `models`) instead.

## Security

- Scoobert treats every project as untrusted. Files in a project are only ever read as data or instructions for the model, never run as code by Scoobert itself.
- The local model server listens only on your computer and requires a random key that changes every time Scoobert starts, so web pages and other programs cannot use it.
- Model downloads are pinned to a tested version and checked against their published checksums.
- Web search is off until you turn it on. Web text reaches the model marked as information rather than instructions, and the page reader refuses addresses on your computer or local network.
- The test browser runs headless with a new, empty profile each task. It opens pages on your computer and files, opens other sites only when web search is on, and never opens other addresses on your local network. While web search is off, the browser's own requests go through a proxy that does not exist, so a page or script cannot reach the internet either. It cannot download files. Running a script in the page asks for approval in **Ask before changes**, like a command.
- Scoobert checks GitHub once a day for a newer release and shows a notice. It downloads an update only when you select **Download and install**, and it deletes the download if it does not match the checksum GitHub publishes for it. You can also check from Settings with **Check for updates**. A portable copy only links to the release page.

## Build from source

You need [Rust](https://rustup.rs) 1.88 or later. On Linux, also install the development packages for `libxkbcommon`, `wayland`, and `dbus`.

```bash
cargo build --release
```

The program looks for `llama-server` next to itself in a `llama` folder, then in the folder set in Settings, then on the `PATH`. To bundle it:

- Windows: `scripts\fetch-llama.ps1` copies the build installed by `winget install ggml.llamacpp`, or downloads a release with `-Release b11193`. Then `scripts\package-windows.ps1` builds the installer with NSIS.
- Linux: `scripts/fetch-llama.sh b11193`, then `scripts/package-linux.sh` builds the AppImage and the tarball.

Pushing a tag such as `v0.1.0` runs `.github/workflows/release.yml`, which builds the Windows installer, the portable zip, the AppImage, and the tarball, and publishes them as a GitHub release. The `repository` field in `Cargo.toml` is where Scoobert checks for newer releases.

## License

MIT. See [LICENSE](LICENSE). The bundled llama.cpp server is also MIT licensed.
