# 📤 How to Upload This Repository to GitHub

Follow these steps to create your GitHub repository and upload all the files.

## Option 1: Using GitHub Web Interface (Easiest)

### Step 1: Create Repository on GitHub
1. Go to https://github.com
2. Log in to your account
3. Click the **"+"** icon in the top-right corner
4. Select **"New repository"**
5. Fill in the details:
   - **Repository name**: `Deep-learning-practice-claude`
   - **Description**: "Comprehensive deep learning tutorials for genomics with PyTorch"
   - **Visibility**: Choose Public or Private
   - ✅ Check "Add a README file" (we'll replace it)
   - **Add .gitignore**: None (we have our own)
   - **Choose a license**: MIT or your preference
6. Click **"Create repository"**

### Step 2: Upload Files
1. On your new repository page, click **"Add file"** → **"Upload files"**
2. Download all files from the links below
3. Drag and drop the entire folder structure into the upload area
4. Add a commit message: "Initial commit: Session 1 - Tensor Basics for Genomics"
5. Click **"Commit changes"**

### Files to Upload:
```
Deep-learning-practice-claude/
├── README.md
├── .gitignore
├── Genomics/
│   ├── README.md
│   ├── pytorch_genomics_practice_curriculum.md
│   ├── pytorch_genomics_practice_curriculum.ipynb
│   └── session-1/
│       ├── README.md
│       ├── session_01_tensor_basics.md
│       └── session_01_tensor_basics.ipynb
```

---

## Option 2: Using Git Command Line (Advanced)

### Prerequisites
- Git installed on your computer
- GitHub account created
- All files downloaded to a local folder

### Step 1: Create Repository on GitHub
1. Go to https://github.com
2. Create new repository named `Deep-learning-practice-claude`
3. **DO NOT** initialize with README (we have our own)
4. Copy the repository URL (should be like: `https://github.com/YOUR_USERNAME/Deep-learning-practice-claude.git`)

### Step 2: Initialize and Upload
Open terminal/command prompt and run:

```bash
# Navigate to the downloaded folder
cd /path/to/Deep-learning-practice-claude

# Initialize git repository
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: Session 1 - Tensor Basics for Genomics"

# Add remote repository (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/Deep-learning-practice-claude.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Troubleshooting Git
If you get authentication errors:
```bash
# Use personal access token instead of password
# Generate token at: https://github.com/settings/tokens
```

---

## Option 3: Using GitHub Desktop (User-Friendly)

### Step 1: Install GitHub Desktop
1. Download from https://desktop.github.com/
2. Install and sign in with your GitHub account

### Step 2: Create Repository
1. Click **"File"** → **"New repository"**
2. Name: `Deep-learning-practice-claude`
3. Local path: Choose where to save
4. Click **"Create repository"**

### Step 3: Add Files
1. Copy all downloaded files into the repository folder
2. GitHub Desktop will show the new files
3. Add commit message: "Initial commit: Session 1"
4. Click **"Commit to main"**
5. Click **"Publish repository"**
6. Choose visibility (Public/Private)
7. Click **"Publish repository"**

---

## 📂 Download Instructions

All files are ready in the outputs folder. Download them using the links provided, then follow one of the upload methods above.

### File Structure to Maintain:
```
Deep-learning-practice-claude/          # Root folder
│
├── README.md                            # Main repository README
├── .gitignore                           # Git ignore file
│
└── Genomics/                            # Genomics track folder
    │
    ├── README.md                        # Genomics track README
    ├── pytorch_genomics_practice_curriculum.md
    ├── pytorch_genomics_practice_curriculum.ipynb
    │
    └── session-1/                       # Session 1 folder
        ├── README.md                    # Session 1 README
        ├── session_01_tensor_basics.md
        └── session_01_tensor_basics.ipynb
```

---

## ✅ Verification Checklist

After uploading, verify your repository has:
- [ ] All 8 files uploaded
- [ ] Correct folder structure (3 levels)
- [ ] README.md displays properly on main page
- [ ] .ipynb files preview correctly on GitHub
- [ ] .gitignore file is present
- [ ] All links in README files work

---

## 🎨 Optional: Customize Your Repository

### Add Topics (Tags)
On your repository page:
1. Click **"⚙️ Settings"** (repository settings, not account)
2. In the "About" section, click **"⚙️"**
3. Add topics:
   - `deep-learning`
   - `pytorch`
   - `genomics`
   - `bioinformatics`
   - `machine-learning`
   - `tutorial`
   - `jupyter-notebook`
4. Save changes

### Add Description
In the same "About" section:
- Description: "Deep learning practice materials for genomics with PyTorch - hands-on tutorials from basic tensors to generative AI"
- Website: (optional) Link to your portfolio or blog

### Create Releases
Once you complete more sessions:
1. Click **"Releases"** → **"Create a new release"**
2. Tag version: `v1.0-session-1`
3. Title: "Session 1: Tensor Basics for Genomics"
4. Description: Brief summary
5. Publish release

---

## 🚀 After Upload

### Share Your Work
- Tweet about it with #PyTorch #DeepLearning #Genomics
- Share on LinkedIn
- Post in bioinformatics forums
- Add to your portfolio

### Keep Learning
- Complete Session 1 exercises
- Wait for Session 2 release
- Star useful repositories
- Connect with other learners

### Maintain Repository
As you add more sessions:
```bash
# Pull latest changes (if working from multiple places)
git pull origin main

# Add new files
git add .

# Commit changes
git commit -m "Add Session 2: Autograd and Gradient Descent"

# Push to GitHub
git push origin main
```

---

## 🆘 Common Issues

### Issue: Files too large
**Solution**: GitHub has a 100MB file size limit. For large data files, use Git LFS or host elsewhere.

### Issue: Can't see .gitignore
**Solution**: Hidden files might not show in file explorer. They're there! Use `ls -a` in terminal to verify.

### Issue: Notebook doesn't render
**Solution**: Wait a few minutes. GitHub needs time to render .ipynb files. Refresh the page.

### Issue: Authentication failed
**Solution**: Use a Personal Access Token instead of password. Generate at https://github.com/settings/tokens

---

## 📧 Need Help?

- GitHub Documentation: https://docs.github.com/
- Git Tutorial: https://git-scm.com/doc
- GitHub Community: https://github.community/

---

**Good luck with your repository! 🎉**

Once uploaded, your repository will be live at:
`https://github.com/YOUR_USERNAME/Deep-learning-practice-claude`
