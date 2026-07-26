# Remote GPU Machine — Access Card

> Hand this card to anyone who needs to connect to the **dorm Windows PC** and use its **RTX 4060 GPU**.
> Everything below was verified live on 2026-07-24.

---

## 0. TL;DR — Connect in 3 steps

```bash
# 1. Turn on Tailscale and log into the tailnet (one time)
tailscale up

# 2. SSH in (uses the alias `dorm-gpu` defined in ~/.ssh/config)
ssh dorm-gpu

# 3. Check the GPU is visible
nvidia-smi
```

That's it. You're now on a Linux shell (WSL2) on the Windows PC, with the GPU available.

---

## 1. What the other person needs (prerequisites)

| Need | How |
|------|-----|
| **Tailscale account access** | Must be a member of the tailnet `shiyaol492@gmail.com` (MagicDNS suffix `tail6e08f1.ts.net`). Ask the tailnet owner to invite you. |
| **Tailscale client** | Install from <https://tailscale.com/download> (Windows / macOS / Linux). Run `tailscale up` and log in. |
| **An SSH key on the PC** | The dorm PC must have your **public key** in `~/.ssh/authorized_keys`. Send the owner your `~/.ssh/id_ed25519.pub`. |
| **SSH client** | Built into macOS/Linux. On Windows use OpenSSH or WSL. |

> The owner authorizes two things: (a) adds you to the tailnet, (b) adds your pubkey to the PC.
> Until both are done, you can't log in.

---

## 2. Connection details

| Field | Value |
|-------|-------|
| **SSH alias** | `dorm-gpu` |
| **Tailscale IP** | `100.86.5.128` |
| **MagicDNS name** | `pcli-1.tail6e08f1.ts.net` |
| **Port** | `22` |
| **Login user** | `chrisv` |
| **Shell you land in** | WSL2 `bash` (Linux), working dir `/mnt/c/Users/chrisv` |
| **Identity key (this Mac)** | `~/.ssh/id_dorm_gpu` |

### `~/.ssh/config` entry (paste on the client machine)

```sshconfig
Host dorm-gpu
    HostName 100.86.5.128
    User chrisv
    IdentityFile ~/.ssh/id_ed25519        # <-- use YOUR key path
    StrictHostKeyChecking accept-new
```

If you prefer the DNS name over the IP, replace the `HostName` with `pcli-1.tail6e08f1.ts.net`.

---

## 3. Hardware

| Component | Spec |
|-----------|------|
| **CPU** | Intel Core i7-14700HX, 28 threads |
| **RAM** | 16 GB (15 Gi visible; ~14 Gi free at idle) |
| **GPU** | NVIDIA GeForce RTX 4060 Laptop, 8 GB VRAM |
| **GPU driver** | 566.24 (Windows-side), CUDA compute capability **8.9** (Ada) |
| **Disk `/` (WSL ext4)** | 1.0 TB, ~948 GB free |
| **Disk `C:`** | 301 GB, **only ~8 GB free** — avoid writing here |
| **Disk `D:`** | 652 GB, ~174 GB free |

> ⚠️ **The C: drive is 98% full.** Put big datasets and outputs on `D:` (`/mnt/d/...`) or inside the WSL filesystem (`~/...`), not under `/mnt/c/`.

---

## 4. Software stack

| Tool | Location / version |
|------|--------------------|
| **OS layer** | Windows 11 + WSL2 (kernel `5.15.153.1-microsoft-standard-WSL2`) |
| **WSL hostname** | `PCli` |
| **Python (WSL)** | `/usr/bin/python3` → Python 3.12.3 |
| **Python (Windows)** | `/mnt/c/Python314/` → Python 3.14 |
| **CUDA Toolkit (Windows)** | v11.2 at `/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2` |
| **GPU in WSL** | ✅ works via WSL2 GPU passthrough (`nvidia-smi` shows the 4060) |
| **uv** | ✅ installed in WSL: `~/.local/bin/uv` (v0.11.17). Run `export PATH="$HOME/.local/bin:$PATH"`. |
| **JAX (GPU)** | ✅ **built & verified** — venv at `~/.venv-jaxfem`, jax 0.10.2 + cuda12, `jax.default_backend() == "gpu"`, `[CudaDevice(id=0)]`. See **§9** for usage & how it was built. |

> Note: Windows CUDA toolkit is 11.2, but for JAX/PyTorch in WSL you typically install the **CUDA 12 runtime via pip** (`jax[cuda12]`, `torch`), which does not require the Windows-side toolkit. The Windows driver (566.24) is what matters and it supports CUDA 12.

---

## 5. Recipes — common tasks

### First time: set up a JAX GPU environment in WSL

> **Already done (2026-07-25).** A ready venv exists at `~/.venv-jaxfem`.
> You only need to **activate** it (see §9). Build-from-scratch notes are kept
> in §9.2 for reference / rebuilding.

```bash
ssh dorm-gpu
source ~/.venv-jaxfem/bin/activate     # ready-made JAX GPU env
python -c "import jax; print(jax.devices())"   # should list [CudaDevice(id=0)]
```

### Run a long job in the background (don't lose it when SSH drops)

```bash
ssh dorm-gpu
tmux new -s train          # start a named session; run your job inside
# ...your training command...
# Detach:  Ctrl-b then d      (job keeps running)
# Reattach next time:
ssh dorm-gpu -t 'tmux attach -t train'
```

### Sync this project (JAX-FEM4Geo) to the GPU box

```bash
# from the Mac, in the repo root:
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude 'outputs' \
    ./ dorm-gpu:~/projects/JAX-FEM4Geo/
```

### Pull results back to the Mac

```bash
rsync -avz dorm-gpu:~/projects/JAX-FEM4Geo/outputs/ ./outputs/
```

### Use PowerShell / cmd instead of WSL bash

```bash
ssh dorm-gpu 'powershell.exe -NoProfile -Command "Get-Process | Select -First 3"'
```

---

## 6. Security — verify on first connect

On the **very first** SSH connection you'll be asked to accept the host key. Confirm it matches one of these (SHA256 fingerprints):

| Type | Fingerprint |
|------|-------------|
| **ED25519** (primary) | `SHA256:1m4PFTMFRqOslEzPDK/QzpcpMb7V/JwVY66cifOTwGA` |
| RSA | `SHA256:GKI+ivTREmoqQ1EN2n9qztlLwWLPuvwxcAnf6YGZ6TA` |

**Public key currently authorized** on the PC (this Mac → PC):
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAKZK5ZvBCwKDotp1idxwjAad5nnO7bTXbB1iHuqxFDr chrisv@mac-to-dorm-gpu
```

To add a **new person**: append *their* pubkey to `~/.ssh/authorized_keys` on the PC (`ssh dorm-gpu` then edit that file).

---

## 7. Troubleshooting

**SSH output is garbled with `w s l : …` characters**
That banner is WSL printing a Chinese "localhost relay" notice on stderr. It's harmless. Suppress it:
```bash
ssh dorm-gpu '<command>' 2>/dev/null
# or, permanently, add to ~/.ssh/config under the Host:
#   LogLevel ERROR
```

**`nvidia-smi: command not found` inside WSL**
The WSL path may not include the driver shim. Use the full path:
```bash
/mnt/c/Windows/System32/nvidia-smi.exe
```

**`Permission denied (publickey)`**
Your pubkey isn't on the PC yet. Send your `~/.ssh/id_ed25519.pub` to the owner to add.

**`Could not connect` / timeout**
- Confirm the dorm PC is **powered on** and a user is logged in (or Wake-on-LAN is set up — it is *not* by default).
- Confirm `tailscale status` on your side shows `pcli-1` as **online**, not `offline`.

**Slow / high latency**
~280 ms round-trip is normal for a relayed Tailscale connection. For large file transfers prefer `rsync` with `-z` (compression). If both peers have open NAT you may get a direct connection that's faster.

**GPU shows but JAX says "no GPU devices"**
Make sure you installed the CUDA build (`jax[cuda12]`), not plain `jax`, and that your venv is active. Check `python -c "import jax; print(jax.devices())"`.

**`OSError: libGLU.so.1` / `libXft.so.2` when importing gmsh**
gmsh's Python wheel dlopens X11/OpenGL libraries at import time. Install them once (needs root):
```bash
sudo apt-get install -y libglu1-mesa libopengl0 libxft2
```
(You can verify with `ldd ~/.venv-jaxfem/lib/libgmsh.so.* | grep "not found"` — should be empty when complete.)

---

## 9. JAX-FEM GPU environment (ready-made)

A fully working JAX + jax-fem GPU venv is already built and verified (2026-07-25).
You normally just activate it.

### 9.1 Daily usage

```bash
ssh dorm-gpu
cd /mnt/d/scientific_research/SelfCode/JAX-FEM-geo   # project code lives here
source ~/.venv-jaxfem/bin/activate                    # the GPU venv
python -c "import jax; print(jax.default_backend())"  # -> gpu
```

**What's in the venv** (`~/.venv-jaxfem`, Python 3.12.3, ext4 filesystem):

| Package | Version | Purpose |
|---------|---------|---------|
| jax[cuda12] | 0.10.2 | GPU autodiff core |
| jaxlib | 0.10.2 | native CUDA backend |
| jax-fem | 0.0.11 | FEM (editable install from `jax-fem-main/`) |
| fenics-basix | 0.11.0 | element shape functions (jax_fem.basis) |
| meshio | 5.3.5 | mesh I/O |
| gmsh | 4.15.2 | mesh generation (needs libGLU/libXft, see §7) |
| numpy / scipy / matplotlib | 2.4.6 / 1.17.1 / 3.11.1 | scientific stack |
| pyfiglet | 1.0.4 | jax_fem banner |

> **No PETSc.** The venv uses the `umfpack_solver` (scipy direct) backend, which
> works on GPU just like on macOS. If you ever need PETSc for very large
> problems: `conda install -c conda-forge petsc4py` (heavy, not recommended
> unless needed).

### 9.2 Rebuild from scratch (reference)

If the venv ever needs recreating, this is the exact procedure that worked:

```bash
# 1. Create the venv on EXT4 (WSL home) — NOT on /mnt/d (NTFS corrupts installs, see §10)
ssh dorm-gpu
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 ~/.venv-jaxfem
source ~/.venv-jaxfem/bin/activate

# 2. Install deps (cache is warm at ~/.cache/uv, ~11 GB, so this is fast)
uv pip install "jax[cuda12]==0.10.2" numpy==2.4.6 scipy==1.17.1 matplotlib==3.11.1 \
              meshio==5.3.5 gmsh==4.15.2 fenics-basix==0.11.0 pyfiglet==1.0.4

# 3. Install the patched local jax-fem (editable, from the repo)
cd /mnt/d/scientific_research/SelfCode/JAX-FEM-geo
uv pip install -e jax-fem-main/

# 4. Verify GPU
python -c "import jax; print(jax.default_backend(), jax.devices())"

# 5. Install gmsh's X11 deps (root, one time)
sudo apt-get install -y libglu1-mesa libopengl0 libxft2
```

### 9.3 Known limitation: GPU memory pre-allocation warnings

JAX tries to pre-allocate up to ~6 GB on startup and prints scary-looking
`CUDA_ERROR_OUT_OF_MEMORY` lines when the 4060's 8 GB can't fit the full
request. **These are harmless** — JAX falls back to smaller chunks and the
computation succeeds. To silence them, set before importing jax:
```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# or cap it: os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
```

---

## 10. Pitfalls discovered during setup (read before debugging)

These cost real time. Don't repeat them.

1. **NEVER create a Python venv on `/mnt/d` or `/mnt/c`.** The NTFS mount under
   WSL silently corrupts package installs — `pip/uv` writes the `.dist-info`
   metadata but the package `__init__.py` / `.so` files come out empty, giving
   ghost errors like `module 'jax' has no attribute '__version__'`.
   **Always put venvs in the WSL ext4 filesystem** (`~/.venv-*`). Project code
   can stay on `/mnt/d`; only the venv must be on ext4.

2. **`git diff` shows the entire repo as modified on first checkout.** This is
   CRLF/LF line-ending noise (Windows git defaults `core.autocrlf` flips
   everything). Check with `git diff --ignore-all-space --stat` — if that's
   empty, the changes are phantom and `git checkout .` is safe (loses nothing).
   Prevent recurrence: `git config core.autocrlf input` (per-repo).

3. **`git branch --set-upstream-to` / `git pull` fails with `chmod ... Operation not permitted`.**
   On `/mnt/d` (NTFS) git can't write the `.git/config.lock` with POSIX perms.
   Workaround: don't rely on tracking config; pull explicitly with
   `git fetch origin && git reset --hard origin/<branch>`.

4. **`gmsh` import fails with `libGLU.so.1` even though jax-fem doesn't visually need it.**
   `jax_fem/generate_mesh.py` did `import gmsh` at module top, so merely
   importing `jax_fem` pulled gmsh in. Fixed in this repo by making the gmsh
   import lazy (commit `b9cce5d`); if you fork/upgrade jax-fem, redo that
   change. gmsh then needs `libglu1-mesa libopengl0 libxft2` (root, see §7).

5. **`fenics-basix` is a hard dependency of jax-fem.** `jax_fem/basis.py` uses
   `basix.ElementFamily.P` at module level. It's easy to miss because it's
   listed in `requirements-mac.txt` as `fenics-basix`. Pip-install it into the
   venv (no sudo).

6. **`test_diff_dp.py` (original) needs PETSc; use `test_diff_dp_mac.py` instead.**
   The original test hardcodes `petsc_solver`. The `_mac` variant uses
   `umfpack_solver` and runs identically on this GPU box. Both pass.

---

## 11. Quick reference

```bash
ssh dorm-gpu                    # log in (lands in WSL bash)
ssh dorm-gpu 'nvidia-smi -L'    # list GPUs
ssh dorm-gpu -t 'tmux a -t train' || ssh dorm-gpu -t 'tmux new -s train'  # attach or create
ssh dorm-gpu 'df -h / /mnt/d'   # check free space
exit                            # leave

# GPU compute workflow (once logged in):
cd /mnt/d/scientific_research/SelfCode/JAX-FEM-geo && source ~/.venv-jaxfem/bin/activate
```

---

*Owner: chrisv · Tailnet: shiyaol492@gmail.com · Last verified: 2026-07-25 (gmsh OK, DP differentiability 5/5 on RTX 4060)*
