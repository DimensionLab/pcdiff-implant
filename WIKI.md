# Wiki Documentation Setup

This repository includes comprehensive wiki-style documentation for multi-GPU training and other advanced topics.

## 📁 Wiki Files

The following files are structured for GitHub Wiki or standalone documentation:

- **[wiki-home.md](./wiki-home.md)** - Main wiki landing page with complete documentation index
- **[wiki-sidebar.md](./wiki-sidebar.md)** - Navigation sidebar for quick access
- **[pcdiff/distributed-training.md](./pcdiff/distributed-training.md)** - Comprehensive multi-GPU training guide

## 🌐 Setting Up GitHub Wiki (Optional)

To publish this documentation as a GitHub Wiki:

### Option 1: Manual Setup

1. Navigate to your repository's Wiki tab on GitHub
2. Create a new page called "Home"
3. Copy content from `wiki-home.md` into the Home page
4. Create a page called "_Sidebar" 
5. Copy content from `wiki-sidebar.md` into the _Sidebar
6. Create additional pages for each documentation file

### Option 2: Using Git (Advanced)

GitHub wikis are actually separate git repositories:

```bash
# Clone your wiki repository
git clone https://github.com/YOUR_USERNAME/pcdiff-implant.wiki.git

# Copy wiki files
cp wiki-home.md pcdiff-implant.wiki/Home.md
cp wiki-sidebar.md pcdiff-implant.wiki/_Sidebar.md
cp pcdiff/distributed-training.md pcdiff-implant.wiki/distributed-training.md

# Commit and push
cd pcdiff-implant.wiki
git add .
git commit -m "Add comprehensive documentation"
git push origin master
```

## 📖 Using Documentation Standalone

The documentation works perfectly as standalone markdown files in the repository:

- Browse [wiki-home.md](./wiki-home.md) for the complete documentation index
- Access detailed guides directly:
  - [Distributed Training Guide](pcdiff/distributed-training.md)
  - [Training README](pcdiff/README.md)
  - [Installation](INSTALL.md)
  - [Setup](SETUP.md)

## 🔗 Internal Links

All internal links in the documentation use relative paths, so they work both:
- In the repository file browser
- In a GitHub Wiki
- In a local markdown viewer

## 📝 File Naming Convention

This project follows GitHub wiki conventions:

- **README files**: `README.md` (uppercase) - Main documentation in each directory
- **Setup/Install guides**: `INSTALL.md`, `SETUP.md`, `QUICKSTART.md` (uppercase)
- **Wiki/Documentation files**: `distributed-training.md`, `wiki-home.md` (lowercase with hyphens)

## 🎯 Quick Access

### Most Important Documentation

1. **Getting Started**: [INSTALL.md](./INSTALL.md)
2. **Multi-GPU Training**: [pcdiff/distributed-training.md](./pcdiff/distributed-training.md) ⭐
3. **Training Guide**: [pcdiff/README.md](./pcdiff/README.md)
4. **Complete Index**: [wiki-home.md](./wiki-home.md)

### Quick Links to Key Topics

- [Batch Size Scaling](pcdiff/distributed-training.md#understanding-batch-size-distribution)
- [Learning Rate Tuning](pcdiff/distributed-training.md#learning-rate-scaling)
- [Persistent Training Sessions (tmux)](pcdiff/distributed-training.md#5-persistent-training-sessions)
- [Troubleshooting Guide](pcdiff/distributed-training.md#troubleshooting)
- [Performance Optimization](pcdiff/distributed-training.md#performance-expectations)

## 🤝 Contributing to Documentation

When adding new documentation:

1. Use lowercase filenames with hyphens for wiki-style docs: `new-feature-guide.md`
2. Keep README files uppercase: `README.md`
3. Update [wiki-home.md](./wiki-home.md) to link to new content
4. Update [wiki-sidebar.md](./wiki-sidebar.md) if adding major sections
5. Use relative links for internal references

## 📋 Documentation Structure

```
pcdiff-implant/
├── README.md                      # Main project README (uppercase)
├── INSTALL.md                     # Installation guide (uppercase)
├── SETUP.md                       # Setup guide (uppercase)
├── QUICKSTART.md                  # Quick start guide (uppercase)
├── WIKI.md                        # This file
├── wiki-home.md                   # Wiki home page (lowercase)
├── wiki-sidebar.md                # Wiki sidebar (lowercase)
└── pcdiff/
    ├── README.md                  # Training README (uppercase)
    └── distributed-training.md    # Multi-GPU guide (lowercase)
```

## 💡 Tips

- All markdown files render beautifully on GitHub without any wiki setup
- Use the wiki for a more structured documentation experience
- Keep documentation close to code for easier maintenance
- The lowercase naming convention helps distinguish user guides from code READMEs

