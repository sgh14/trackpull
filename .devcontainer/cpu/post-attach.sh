#!/bin/bash
# Post-attach script for devcontainer
# This runs every time you attach/reconnect to the container

# Fix SSH config permissions (dev container workaround)
if [ -f /tmp/host-ssh-config ]; then
    cp /tmp/host-ssh-config ~/.ssh/config 2>/dev/null
    chmod 600 ~/.ssh/config 2>/dev/null
fi

cd "$(dirname "$0")/../.."

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📦 Checking Dependencies (CPU)"
echo "════════════════════════════════════════════════════════════════"
echo ""

if uv pip install -e ".[dev]" -q; then
    echo ""
    echo "✅ Dependencies are up to date!"
    uv run pre-commit install --install-hooks >/dev/null 2>&1 && echo "✅ Pre-commit hooks ready"
    echo ""
else
    echo ""
    echo "❌ Failed to sync dependencies"
    echo "   Check pyproject.toml for errors"
    echo ""
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📊 Current Account Status"
echo "════════════════════════════════════════════════════════════════"
echo ""

if gh auth status >/dev/null 2>&1; then
    GITHUB_USER=$(gh api user --jq .login 2>/dev/null || echo "Unknown")
    GITHUB_EMAIL=$(gh api user --jq .email 2>/dev/null || echo "Unknown")
    echo "🐙 GitHub CLI Account:"
    echo "   User:  $GITHUB_USER"
    echo "   Email: $GITHUB_EMAIL"
else
    echo "❌ ERROR: GitHub CLI is not authenticated!"
    echo ""
    echo "Please run: gh auth login"
    echo ""
    exit 1
fi

echo ""

GIT_NAME=$(git config user.name 2>/dev/null || echo "Not set")
GIT_EMAIL=$(git config user.email 2>/dev/null || echo "Not set")

if [ -n "$GIT_USER_NAME" ] && [ -n "$GIT_USER_EMAIL" ]; then
    if [[ "$GIT_NAME" != "$GIT_USER_NAME" ]] || [[ "$GIT_EMAIL" != "$GIT_USER_EMAIL" ]]; then
        echo "🔧 Fixing Git configuration using .env variables..."
        git config user.name "$GIT_USER_NAME"
        git config user.email "$GIT_USER_EMAIL"
        GIT_NAME="$GIT_USER_NAME"
        GIT_EMAIL="$GIT_USER_EMAIL"
        echo "✅ Git configuration updated:"
        echo "   Name:  $GIT_NAME"
        echo "   Email: $GIT_EMAIL"
        echo ""
    fi
fi

GIT_SCOPE=$(git config --show-origin user.email 2>/dev/null | awk '{print $1}' || echo "unknown")

echo "📝 Git Configuration (for this repo):"
echo "   Name:  $GIT_NAME"
echo "   Email: $GIT_EMAIL"

if [[ "$GIT_SCOPE" == *"local"* ]]; then
    echo "   Scope: Repository-specific ✅"
elif [[ "$GIT_SCOPE" == *"global"* ]]; then
    echo "   Scope: Global (container-wide) ⚠️"
fi

echo ""

if [[ "$GIT_NAME" == "Not set" ]] || [[ "$GIT_EMAIL" == "Not set" ]]; then
    echo "⚠️  WARNING: Git user not configured!"
    echo "   Set it with:"
    echo "   git config user.name \"Your Name\""
    echo "   git config user.email \"your@email.com\""
    echo ""
fi

if gh auth status >/dev/null 2>&1 && [[ "$GIT_EMAIL" != "Not set" ]]; then
    if [[ "$GITHUB_EMAIL" != "$GIT_EMAIL" ]]; then
        echo "⚠️  NOTICE: GitHub and Git emails don't match!"
        echo "   This might be intentional, but verify it's correct."
        echo ""
    fi
fi

echo "════════════════════════════════════════════════════════════════"
echo ""
echo "💡 Quick commands:"
echo "   gh auth switch           # Switch GitHub account"
echo "   gh api user              # Check GitHub account details"
echo "   git config user.email    # Check Git email"
echo ""
