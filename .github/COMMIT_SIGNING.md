# Commit Signing Guide

This document explains how to set up and use commit signing for CogSynDelta contributions.

## Why Sign Commits?

Commit signing provides cryptographic proof that commits were created by who they claim to be from. This:

- **Prevents impersonation**: Ensures commits truly come from the claimed author
- **Maintains chain of custody**: Creates an auditable trail of authentic contributions
- **Meets compliance requirements**: Required by many security-conscious organizations
- **Enables verified badges**: GitHub shows "Verified" on signed commits

## Setup Options

### Option 1: GPG Signing (Recommended)

GPG provides strong cryptographic signatures that work across all Git platforms.

#### 1. Generate a GPG Key

```bash
# Generate a new GPG key (use RSA 4096-bit)
gpg --full-generate-key

# List your keys to find the key ID
gpg --list-secret-keys --keyid-format=long

# Output example:
# sec   rsa4096/ABCD1234EFGH5678 2024-01-15 [SC]
#       1234567890ABCDEF1234567890ABCDEF12345678
# uid                 [ultimate] Your Name <your.email@example.com>
```

#### 2. Configure Git to Use Your Key

```bash
# Set your signing key (use the key ID after rsa4096/)
git config --global user.signingkey ABCD1234EFGH5678

# Enable signing for all commits
git config --global commit.gpgsign true

# Enable signing for all tags
git config --global tag.gpgsign true
```

#### 3. Add Your Key to GitHub

```bash
# Export your public key
gpg --armor --export ABCD1234EFGH5678

# Copy the output (including BEGIN/END lines) to:
# GitHub → Settings → SSH and GPG keys → New GPG key
```

### Option 2: SSH Signing (Git 2.34+)

SSH signing uses your existing SSH key - simpler setup if you already have SSH configured.

#### 1. Configure Git for SSH Signing

```bash
# Point to your SSH key
git config --global user.signingkey ~/.ssh/id_ed25519.pub

# Set format to SSH
git config --global gpg.format ssh

# Enable signing for all commits
git config --global commit.gpgsign true
```

#### 2. Add Signing Key to GitHub

Your SSH key must be added as both an **Authentication key** AND a **Signing key**:

1. Go to GitHub → Settings → SSH and GPG keys
2. Click "New SSH key"
3. Select "Signing Key" as the key type
4. Paste your public key

### Option 3: S/MIME Signing

For enterprise environments with existing PKI infrastructure:

```bash
# Configure S/MIME signing
git config --global gpg.format x509
git config --global user.signingkey /path/to/certificate.pem
git config --global commit.gpgsign true
```

## Verifying Commits

### Locally

```bash
# Verify the last commit
git verify-commit HEAD

# Verify a specific commit
git verify-commit abc1234

# Show signature in log
git log --show-signature -1

# Verify all commits in a range
git log --show-signature main..HEAD
```

### On GitHub

Signed commits display a "Verified" badge:
- **Verified**: Signature is valid and matches a known key
- **Unverified**: Signature exists but cannot be verified
- **No badge**: Commit is not signed

## Troubleshooting

### "gpg: signing failed: No secret key"

Your signing key ID doesn't match available keys:

```bash
# List available keys
gpg --list-secret-keys --keyid-format=long

# Update your git config with correct key ID
git config --global user.signingkey <correct-key-id>
```

### "gpg: signing failed: Inappropriate ioctl for device"

GPG can't prompt for passphrase. Fix with:

```bash
# Add to ~/.bashrc or ~/.zshrc
export GPG_TTY=$(tty)

# Or configure GPG to use a GUI pinentry
echo "pinentry-program /usr/bin/pinentry-gtk-2" >> ~/.gnupg/gpg-agent.conf
gpgconf --kill gpg-agent
```

### SSH Signing "error: Load key failed"

Ensure your SSH agent has the key loaded:

```bash
# Start SSH agent
eval "$(ssh-agent -s)"

# Add your key
ssh-add ~/.ssh/id_ed25519
```

### Commits Show "Unverified" on GitHub

1. Ensure the email in your commit matches your GitHub account
2. Verify your GPG/SSH key is added to GitHub
3. Check that the key hasn't expired

```bash
# Check your commit email
git config user.email

# Verify it matches GitHub - should be in your GitHub email list
```

## Project Policy

### Current Status

Commit signing is **recommended but not required** for CogSynDelta.

### Future Enforcement

When enabled, the `main` branch will require:
- All commits must be signed
- All commits must have verified signatures

To enable (maintainers only):

```bash
# Enable required signatures via GitHub API
gh api repos/tzervas/CogSynDelta/branches/main/protection/required_signatures \
  -X POST \
  -H "Accept: application/vnd.github+json"
```

## CI Integration

Our CI workflows can verify signatures:

```yaml
# In .github/workflows/verify-signatures.yml
- name: Verify commit signatures
  run: |
    # Verify all commits in PR are signed
    git log --show-signature origin/main..HEAD 2>&1 | grep -q "Good signature" || exit 1
```

## Resources

- [GitHub: Signing commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits)
- [Git: Signing Your Work](https://git-scm.com/book/en/v2/Git-Tools-Signing-Your-Work)
- [GPG Handbook](https://www.gnupg.org/gph/en/manual.html)

---

> **Note**: This guide aligns with [ADR-0005](../docs/adr/0005-conventional-commits-enforcement.md)
> on commit conventions and the project's security standards.
