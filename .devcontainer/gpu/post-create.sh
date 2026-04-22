#!/bin/bash
# Post-create script for GPU devcontainer
# This runs once when the container is first created

set -e

echo "==> Running post-create setup (GPU version)..."
echo ""

# Display CUDA information
echo "CUDA Environment Check"
echo "================================"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
    echo ""
else
    echo "⚠️  WARNING: nvidia-smi not available"
    echo "   GPU may not be accessible in this container"
    echo ""
fi

# Install Python packages
echo "Installing Python packages..."
cd "$(dirname "$0")/../.."

if [ -d ".venv" ]; then
    echo "Removing existing .venv (may have incorrect permissions)..."
    rm -rf .venv 2>/dev/null || sudo rm -rf .venv
    uv cache clean
fi

uv venv .venv
uv pip install -e ".[dev]"
echo "✅ Python packages installed"
echo ""

echo "Configuring Git..."
git config --global user.useConfigOnly true
echo "✅ Git configured to require explicit user.name and user.email"
echo ""

echo "🔗 Installing pre-commit hooks..."
uv run pre-commit install
echo "✅ Pre-commit hooks installed"
echo ""

echo "GitHub Account Setup"
echo "================================"
echo ""

if ! gh auth status >/dev/null 2>&1; then
    if [ -n "$GH_TOKEN" ]; then
        echo "Authenticating with token from .env..."
        if echo "$GH_TOKEN" | gh auth login --with-token 2>/dev/null; then
            echo "✅ GitHub CLI authenticated successfully with token!"
            echo ""
        else
            echo "⚠️  Token authentication failed. Falling back to interactive login."
            echo ""
            GH_TOKEN=""
        fi
    fi

    if ! gh auth status >/dev/null 2>&1; then
        echo "❌ ERROR: GitHub CLI is not authenticated!"
        echo ""
        echo "Please run: gh auth login"
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

echo "Your authenticated GitHub accounts:"
gh auth status 2>&1 | grep "✓" || echo "None found"
echo ""

echo "Select which GitHub account to use for this repository:"
echo "(You can change this later with: gh auth switch)"
echo ""

if gh auth switch; then
    echo ""
    CURRENT_USER=$(gh api user --jq .login)
    CURRENT_EMAIL=$(gh api user --jq .email)
    echo "✅ Active account: $CURRENT_USER ($CURRENT_EMAIL)"
    echo ""

    echo "Do you want to configure Git user for this repository?"
    read -p "Use account '$CURRENT_USER' for Git commits? (y/n): " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -n "$GIT_USER_NAME" ]; then
            GIT_NAME="$GIT_USER_NAME"
            echo "Using name from .env: $GIT_NAME"
        else
            read -p "Enter your name for Git commits: " GIT_NAME
        fi

        if [ -n "$GIT_USER_EMAIL" ]; then
            GIT_EMAIL="$GIT_USER_EMAIL"
            echo "Using email from .env: $GIT_EMAIL"
        else
            GIT_EMAIL=$CURRENT_EMAIL
            read -p "Use email '$GIT_EMAIL'? (y/n): " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                read -p "Enter your email for Git commits: " GIT_EMAIL
            fi
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
else
    echo ""
    echo "⚠️  GitHub account selection skipped or failed."
    echo "   You can set it up later with: gh auth switch"
fi

echo ""
echo "GPU Dev Container Setup Complete!"
echo ""
echo "Quick commands:"
echo "   nvidia-smi               # Check GPU status"
echo "   pytest tests/ -v         # Run tests"
echo "   gh api user              # Check active GitHub account"
echo "   git config user.email    # Check Git email"
echo "   gh auth switch           # Switch GitHub account"
