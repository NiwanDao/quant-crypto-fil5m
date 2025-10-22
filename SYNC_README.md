# Remote Development Sync Setup

## 🚀 Quick Start

### Manual Sync

```bash
# Sync your changes to the server
./sync_to_server.sh
```

### Automatic Sync (File Watcher)

```bash
# Watch for file changes and auto-sync
./watch_and_sync.sh
```

## 📋 SSH Configuration

Your server is configured as `quant-server` in `~/.ssh/config`:

- **Host:** 110.186.66.16
- **User:** shiyi
- **Port:** 2202
- **Path:** /home/shiyi/quant-crypto-fil5m

## 🔧 What Gets Synced

**Included:**

- All Python files
- Configuration files
- Scripts and utilities
- Documentation

**Excluded:**

- `.venv/` (virtual environment)
- `.git/` (git repository)
- `__pycache__/` (Python cache)
- `*.pyc` (compiled Python)
- `.DS_Store` (macOS files)
- `data/raw/` (raw data files)
- `data/processed/` (processed data)
- `models/trained/` (trained models)

## 🎯 Workflow

1. **Edit files locally** in Cursor
2. **Run sync** with `./sync_to_server.sh`
3. **Or use auto-sync** with `./watch_and_sync.sh`
4. **Changes are immediately available** on the server

## 🔍 Testing

```bash
# Test SSH connection
ssh quant-server

# Test sync
./sync_to_server.sh

# Check files on server
ssh quant-server "ls -la /home/shiyi/quant-crypto-fil5m"
```

## 📝 Notes

- The sync script uses `rsync` with `--delete` flag
- This means files deleted locally will be deleted on the server
- Always backup important data before syncing
- Virtual environments are excluded to avoid conflicts
