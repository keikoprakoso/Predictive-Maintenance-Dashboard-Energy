#!/bin/bash

# 🚀 GitHub Upload Helper Script
# Author: Keiko Rafi Ananda Prakoso

echo "🚀 GitHub Upload Helper for Predictive Maintenance Dashboard"
echo "============================================================"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Git repository not initialized. Please run 'git init' first."
    exit 1
fi

# Check if remote is already set
if git remote -v | grep -q "origin"; then
    echo "✅ Remote origin already configured:"
    git remote -v
    echo ""
    echo "To push to GitHub, run:"
    echo "git push -u origin main"
else
    echo "📋 To connect to GitHub, follow these steps:"
    echo ""
    echo "1. Create a new repository on GitHub.com"
    echo "   - Name: predictive-maintenance-dashboard-energy"
    echo "   - Description: 🏭 Predictive Maintenance Dashboard for Geothermal Energy Sector"
    echo "   - Make it Public"
    echo "   - DO NOT initialize with README, .gitignore, or license"
    echo ""
    echo "2. After creating the repository, run these commands:"
    echo ""
    echo "   # Replace YOUR_USERNAME with your GitHub username"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/predictive-maintenance-dashboard-energy.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
    echo "3. Verify the upload by visiting your repository on GitHub"
fi

echo ""
echo "📚 For detailed instructions, see: GITHUB_UPLOAD_GUIDE.md"
echo "🎯 For portfolio tips, see: docs/business/LINKEDIN_POST.md" 