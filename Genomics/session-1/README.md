# PyTorch Genomics Training - Session 01

## 📦 Available Files

### 1. **session_01_tensor_basics.ipynb** (Jupyter Notebook)
   - Interactive notebook format
   - Can be opened in Jupyter Lab, Jupyter Notebook, or Google Colab
   - Contains all exercises with code cells ready to run

### 2. **session_01_tensor_basics.md** (Markdown)
   - Static document format
   - Good for reading and reference
   - Can be viewed in any text editor or Markdown viewer

## 🚀 How to Use

### Option A: Google Colab (Recommended for Beginners)
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Upload `session_01_tensor_basics.ipynb`
4. Run cells sequentially with Shift+Enter
5. PyTorch is pre-installed in Colab!

### Option B: Local Jupyter Notebook
1. Install Jupyter and PyTorch:
   ```bash
   pip install jupyter torch numpy
   ```
2. Navigate to the directory with the notebook
3. Start Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open `session_01_tensor_basics.ipynb` in your browser
5. Run cells with Shift+Enter

### Option C: JupyterLab
1. Install JupyterLab:
   ```bash
   pip install jupyterlab torch numpy
   ```
2. Start JupyterLab:
   ```bash
   jupyter lab
   ```
3. Open the notebook file
4. Run cells sequentially

## 📋 Session Content

### Topics Covered:
1. **DNA Sequence Encoding** - Integer and one-hot encoding
2. **Gene Expression Analysis** - Matrix operations and normalization
3. **K-mer Analysis** - Extracting and analyzing sequence patterns
4. **Position Weight Matrices** - Creating and using PWMs for motif detection
5. **Broadcasting** - Efficient batch operations

### Exercises:
- 5 main exercises with multiple parts
- 2 challenge problems
- All with starter code and expected outputs

## 💡 Tips

1. **Run cells in order** - Each cell may depend on previous ones
2. **Experiment** - Try changing parameters and see what happens
3. **Read error messages** - They're helpful for learning
4. **Use print statements** - Add your own to understand what's happening
5. **Check shapes** - Use `.shape` to verify tensor dimensions

## 🆘 Troubleshooting

### "Module not found" errors
```bash
pip install torch numpy
```

### Google Colab Issues
- Make sure you're running Python 3
- Try Runtime → Restart runtime if things get stuck

### Can't see output
- Make sure you ran all previous cells
- Try Kernel → Restart & Run All

## 📚 Next Steps

After completing Session 01:
- Review the self-assessment checklist
- Try the additional practice ideas
- Move on to Session 02: Autograd and Gradient Descent

## 📖 References

- PyTorch Documentation: https://pytorch.org/docs/
- Tutorial Source: Deep Learning with PyTorch (Second Edition)
- Additional practice datasets available at UCI ML Repository, ENCODE, GTEx

---

**Questions or Issues?**
Feel free to experiment and modify the code - that's the best way to learn!
