#!/bin/bash
# Post-create script for devcontainer
# This runs once when the container is first created

set -e

echo "🚀 Running post-create setup..."
echo ""

# Install Python packages
echo "📦 Installing Python packages..."
cd "$(dirname "$0")/../.."

# Remove existing .venv if it has permission issues (common when mounted from host)
if [ -d ".venv" ]; then
    echo "🗑️  Removing existing .venv (may have incorrect permissions)..."
    rm -rf .venv 2>/dev/null || sudo rm -rf .venv
    uv cache clean  # clear stale cache entries that may cause hardlink failures
fi

uv venv .venv
uv pip install -e ".[dev]"
echo "✅ Python packages installed (including dev tools: ruff, pre-commit)"
echo ""

# Configure Git to require explicit user configuration
echo "⚙️  Configuring Git..."
git config --global user.useConfigOnly true
echo "✅ Git configured to require explicit user.name and user.email"
echo ""

# Install pre-commit hooks (so commits are automatically checked)
echo "🔗 Installing pre-commit hooks..."
uv run pre-commit install
echo "✅ Pre-commit hooks installed (commits will be checked automatically)"
echo ""

# Run GitHub account setup
echo "🔧 GitHub Account Setup"
echo "================================"
echo ""

# Check if gh is authenticated
if ! gh auth status >/dev/null 2>&1; then
    # Try automatic authentication with token from .env
    if [ -n "$GH_TOKEN" ]; then
        echo "🔑 Authenticating with token from .env..."
        if echo "$GH_TOKEN" | gh auth login --with-token 2>/dev/null; then
            echo "✅ GitHub CLI authenticated successfully with token!"
            echo ""
        else
            echo "⚠️  Token authentication failed. Falling back to interactive login."
            echo ""
            GH_TOKEN=""  # Clear invalid token
        fi
    fi

    # If token auth failed or wasn't available, do interactive login
    if ! gh auth status >/dev/null 2>&1; then
        echo "❌ ERROR: GitHub CLI is not authenticated!"
        echo ""
        echo "Please run the following command now:"
        echo ""
        echo "  gh auth login"
        echo ""
        while ! gh auth status >/dev/null 2>&1; do
            read -p "Press Enter to run 'gh auth login' now (or Ctrl+C to cancel): "
            gh auth login || true
            echo ""
            if ! gh auth status >/dev/null 2>&1; then
                echo "⚠️  Authentication failed or incomplete. Please try again."
                echo ""
            fi
        done

        echo "✅ GitHub CLI authenticated successfully!"
        echo ""
    fi
fi

echo "📋 Your authenticated GitHub accounts:"
gh auth status 2>&1 | grep "✓" || echo "None found"
echo ""

echo "🔄 Select which GitHub account to use for this repository:"
echo "(You can change this later with: gh auth switch)"
echo ""

if gh auth switch; then
    echo ""
    CURRENT_USER=$(gh api user --jq .login)
    CURRENT_EMAIL=$(gh api user --jq .email)
    echo "✅ Active account: $CURRENT_USER ($CURRENT_EMAIL)"
    echo ""

    if [ -n "$GIT_USER_NAME" ] && [ -n "$GIT_USER_EMAIL" ]; then
        echo "🔧 Configuring Git using .env variables..."
        git config user.name "$GIT_USER_NAME"
        git config user.email "$GIT_USER_EMAIL"
        echo "✅ Git configuration set:"
        echo "   Name:  $GIT_USER_NAME"
        echo "   Email: $GIT_USER_EMAIL"
        echo ""
    else
        echo "📝 Do you want to configure Git user for this repository?"
        read -p "Use account '$CURRENT_USER' for Git commits? (y/n): " -n 1 -r
        echo ""

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            read -p "Enter your name for Git commits: " GIT_NAME

            GIT_EMAIL=$CURRENT_EMAIL
            read -p "Use email '$GIT_EMAIL'? (y/n): " -n 1 -r
            echo ""

            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                read -p "Enter your email for Git commits: " GIT_EMAIL
            fi

            git config user.name "$GIT_NAME"
            git config user.email "$GIT_EMAIL"

            echo ""
            echo "✅ Git configuration set for this repository:"
            echo "   Name:  $(git config user.name)"
            echo "   Email: $(git config user.email)"
        else
            echo ""
            echo "⚠️  Remember to set your Git user before committing:"
            echo "   git config user.name \"Your Name\""
            echo "   git config user.email \"your@email.com\""
        fi
    fi
else
    echo ""
    echo "⚠️  GitHub account selection skipped or failed."
    echo "   You can set it up later with: gh auth switch"
    echo ""
    echo "⚠️  Remember to also set your Git user before committing:"
    echo "   git config user.name \"Your Name\""
    echo "   git config user.email \"your@email.com\""
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "💡 Quick commands:"
echo "   gh api user              # Check active GitHub account"
echo "   git config user.email    # Check Git email"
echo "   gh auth switch           # Switch GitHub account"
