# Session 2: Autograd and Gradient Descent

**Learning the Mechanics of Deep Learning**

---

## 📚 Overview

This session teaches you **how models actually learn** from data using PyTorch's automatic differentiation (autograd) and optimization techniques. You'll understand the core mechanics that power all deep learning.

**Duration**: 2-3 hours  
**Difficulty**: Beginner to Intermediate  
**Prerequisites**: Session 1 completed, basic calculus understanding

---

## 🎯 What You'll Learn

### Core Concepts
- **Autograd**: How PyTorch automatically computes derivatives
- **Gradient Descent**: The fundamental optimization algorithm
- **Loss Functions**: Measuring prediction quality
- **Optimizers**: SGD, Adam, and when to use each
- **Training Loops**: The standard pattern for learning

### Genomics Applications
- Predicting gene expression from TF binding
- Classifying genetic variants (pathogenic vs benign)
- Preventing overfitting in genomics models
- Early stopping strategies

---

## 📂 Files

| File | Format | Description |
|------|--------|-------------|
| `session_02_autograd_gradient_descent.ipynb` | Jupyter Notebook | Interactive exercises |
| `session_02_autograd_gradient_descent.md` | Markdown | Reference document |
| `README.md` | Markdown | This file |

---

## 🧪 Exercises Overview

### Exercise 1: Manual Gradient Descent (⭐)
Build understanding by implementing gradient descent from scratch
- Compute gradients manually
- Update parameters step by step
- Visualize convergence

### Exercise 2: PyTorch Autograd (⭐⭐)
Learn how PyTorch automates gradient computation
- `requires_grad` and computation graphs
- `.backward()` and `.grad`
- Compare with manual implementation

### Exercise 3: PyTorch Optimizers (⭐⭐)
Master the `torch.optim` module
- SGD (Stochastic Gradient Descent)
- Adam optimizer
- Compare different optimizers and learning rates

### Exercise 4: Variant Classification (⭐⭐⭐)
Build a real genomics classifier
- Binary classification (pathogenic/benign)
- Logistic regression
- Binary cross-entropy loss
- Accuracy metrics

### Exercise 5: Train/Validation Split (⭐⭐⭐)
Learn proper model evaluation
- Split data correctly
- Monitor both train and validation
- Understand the train/val gap

### Exercise 6: Overfitting Detection (⭐⭐⭐⭐)
Recognize and prevent overfitting
- Visualize overfitting
- Implement early stopping
- Optimal stopping criteria

### Challenge Problems (⭐⭐⭐⭐⭐)
1. Multi-class gene expression classification
2. Learning rate scheduling

---

## 🚀 Getting Started

### Option 1: Google Colab (Recommended)
1. Download `session_02_autograd_gradient_descent.ipynb`
2. Go to https://colab.research.google.com
3. Upload the notebook
4. Run cells with Shift+Enter

### Option 2: Local Jupyter
```bash
# Navigate to session-2 folder
cd Deep-learning-practice-claude/Genomics/session-2

# Start Jupyter
jupyter notebook session_02_autograd_gradient_descent.ipynb
```

### Option 3: Read First
Open `session_02_autograd_gradient_descent.md` to read through all concepts and code before running.

---

## 📊 Session Structure

```
1. Theory Review (15 min)
   ├── What is learning?
   ├── Gradient descent intuition
   └── Autograd overview

2. Manual Gradient Descent (30 min)
   ├── Problem setup
   ├── Manual gradient computation
   └── Training loop

3. PyTorch Autograd (30 min)
   ├── requires_grad
   ├── Computation graphs
   └── Automatic differentiation

4. Optimizers (30 min)
   ├── SGD
   ├── Adam
   └── Comparisons

5. Real Application (45 min)
   ├── Variant classification
   ├── Training/validation split
   └── Overfitting prevention

6. Challenge Problems (30+ min)
   └── Advanced topics
```

---

## 💡 Key Concepts

### The Training Loop Pattern
```python
# Standard PyTorch training pattern
for epoch in range(num_epochs):
    # 1. Forward pass
    predictions = model(inputs)
    loss = loss_function(predictions, targets)
    
    # 2. Backward pass
    optimizer.zero_grad()  # Clear gradients
    loss.backward()         # Compute gradients
    optimizer.step()        # Update parameters
```

### Critical Points
1. **Always call `optimizer.zero_grad()`** before backward pass
2. **Use `torch.no_grad()`** for validation (don't compute gradients)
3. **Monitor validation loss** to detect overfitting
4. **Choose appropriate learning rate** (typically 0.001-0.1)
5. **Split your data** before any training

---

## 🎓 Learning Objectives Checklist

After completing this session, you should be able to:

**Understanding**
- [ ] Explain how autograd builds computation graphs
- [ ] Describe gradient descent algorithm
- [ ] Identify when overfitting occurs

**Implementation**
- [ ] Write training loops with proper gradient handling
- [ ] Use different optimizers (SGD, Adam)
- [ ] Split data into train/validation sets
- [ ] Implement early stopping

**Application**
- [ ] Train genomics classifiers
- [ ] Choose appropriate loss functions
- [ ] Tune hyperparameters (learning rate, epochs)
- [ ] Diagnose training issues

---

## 🔧 Prerequisites Check

Make sure you have from Session 1:
- ✅ Comfortable with PyTorch tensors
- ✅ Can perform tensor operations
- ✅ Understand tensor shapes and indexing
- ✅ Familiar with DNA sequence encoding

New requirements for Session 2:
- 📊 Basic understanding of derivatives (calculus)
- 📈 Familiarity with optimization concepts
- 🎯 Understanding of train/test splits

---

## 🐛 Common Issues & Solutions

### Issue: Gradients are None
**Cause**: Forgot `requires_grad=True` on parameters  
**Solution**: Set `requires_grad=True` when creating parameter tensors

### Issue: Loss is NaN
**Cause**: Learning rate too high or numerical instability  
**Solution**: Reduce learning rate, add epsilon to loss calculations

### Issue: Training loss decreases but validation increases
**Cause**: Overfitting  
**Solution**: Stop training earlier, add regularization, get more data

### Issue: Both losses stuck/not decreasing
**Cause**: Learning rate too small, poor initialization, or bug  
**Solution**: Increase learning rate, check implementation

### Issue: RuntimeError about grad
**Cause**: Forgot to zero gradients  
**Solution**: Call `optimizer.zero_grad()` before `loss.backward()`

---

## 📈 Expected Outcomes

By the end of this session, you should achieve:
- **Exercise 1**: Loss decreases to < 0.05, parameters approach true values
- **Exercise 2**: Identical results to manual implementation
- **Exercise 3**: Understanding of optimizer differences
- **Exercise 4**: ~85-95% accuracy on variant classification
- **Exercise 5**: Small train/val gap (< 0.05 loss difference)
- **Exercise 6**: Successful early stopping implementation

---

## 🔗 Connections

### From Session 1
- Uses tensor operations learned previously
- Builds on data loading and manipulation
- Extends one-hot encoding concepts

### To Session 3
- Training loops used in neural networks
- Optimizers work with any model architecture
- Validation strategies remain the same

---

## 📚 Additional Resources

### PyTorch Documentation
- [Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [Optimizers](https://pytorch.org/docs/stable/optim.html)
- [Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)

### Tutorials
- [PyTorch Autograd Tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [Training Classifier Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

### Further Reading
- Gradient Descent Visualization: https://distill.pub/2017/momentum/
- Understanding Optimizers: https://ruder.io/optimizing-gradient-descent/
- Overfitting and Regularization: https://www.deeplearning.ai/ai-notes/regularization/

---

## 💾 Save Your Work

After completing exercises:
1. **Save notebook** with outputs (File → Download → .ipynb)
2. **Export to HTML** for easy viewing (File → Download as → HTML)
3. **Commit to GitHub** if using version control
4. **Keep notes** on what worked / didn't work

---

## ✅ Session Completion Checklist

Before moving to Session 3:
- [ ] Completed all 6 main exercises
- [ ] Attempted both challenge problems
- [ ] Understand autograd computation graphs
- [ ] Can implement training loops independently
- [ ] Know how to prevent overfitting
- [ ] Reviewed key concepts
- [ ] Saved your work

---

## 🆘 Getting Help

If you're stuck:
1. **Re-read theory section** - Understanding concepts helps with code
2. **Check expected outputs** - Compare your results
3. **Review error messages** - They're usually informative
4. **Try simpler version** - Break problem into smaller pieces
5. **Search PyTorch docs** - Look up specific functions
6. **Compare with Session 1** - Similar patterns used

---

## 🎯 Next Steps

After completing Session 2:
1. ✅ Review self-assessment checklist
2. 📝 Note challenging concepts for review
3. 🔄 Experiment with hyperparameters
4. ➡️ Move to **Session 3: Building Neural Networks**

Session 3 will use everything you learned here to build full neural network architectures!

---

**Ready to learn how machines learn?** Open the notebook and get started! 🚀

---

*Part of the Deep Learning Practice with Claude - Genomics Track*  
*Session 2 of 12 | Beginner-Intermediate Level*
