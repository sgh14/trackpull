# Development Container (Devcontainer)

This folder configures a **development container** — a fully-equipped, portable development environment that runs inside Docker.

## How to Use It

### First Time Setup

1. **Install prerequisites:**
   - [VS Code](https://code.visualstudio.com/)
   - [Docker Desktop](https://www.docker.com/products/docker-desktop)
   - [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

2. **Create your `.env` file:**
   ```bash
   cp .devcontainer/.env.example .devcontainer/.env
   # Edit .env with your credentials
   ```

3. **Open the project in VS Code** and click **"Reopen in Container"** when prompted.

4. **Wait for setup** (~5 minutes the first time, ~30 seconds after that).

### Choosing CPU vs GPU

When prompted, choose:

- **CPU** — Works on any computer, good for development and analysis
- **GPU** — Requires NVIDIA GPU

---

## Folder Structure

```
.devcontainer/
├── OVERVIEW.md              # This file
├── .env.example             # Template for environment variables
├── .env                     # Your credentials (not in git)
├── cpu/
│   ├── devcontainer.json
│   ├── Dockerfile
│   ├── post-create.sh       # Runs once after container creation
│   └── post-attach.sh       # Runs each time you connect
└── gpu/
    ├── devcontainer.json
    ├── Dockerfile
    ├── post-create.sh
    └── post-attach.sh
```
