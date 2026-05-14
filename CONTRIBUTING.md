# Contributing to SmartClinic AI

Thank you for your interest in contributing to SmartClinic AI! This document provides guidelines and instructions for contributing to the project.

## 🤝 Types of Contributions

We welcome all types of contributions:

- **Bug Reports**: Report issues or problems you've found
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit pull requests with fixes or new features
- **Documentation**: Improve README, docstrings, comments, or guides
- **Testing**: Write tests or test edge cases
- **Performance Optimization**: Suggest improvements to speed or efficiency
- **Translations**: Help with Arabic/English localization

## 🐛 Reporting Bugs

When reporting bugs, please include:

1. **Clear description** of the problem
2. **Steps to reproduce** the issue
3. **Expected behavior** vs **actual behavior**
4. **Environment details**:
   - Python version
   - OS (Windows/Linux/Mac)
   - Relevant package versions
5. **Error messages** or stack traces
6. **Screenshots** if applicable

**Example**:
```
Title: M1 predictor fails with missing PKL file

Description:
When running the application without dept_models.pkl, the app crashes instead of 
showing a graceful error.

Steps to Reproduce:
1. Delete m1/dept_models.pkl
2. Start the application
3. Send a symptom query

Error:
FileNotFoundError: [Errno 2] No such file or directory: 'm1/dept_models.pkl'
```

## 💡 Suggesting Features

When suggesting features:

1. **Clear title** describing the feature
2. **Detailed description** of what you want
3. **Why it's needed** (use cases/benefits)
4. **Possible implementation** (if you have ideas)
5. **Related issues** or discussions

**Example**:
```
Title: Add patient history tracking

Description:
Allow users to save and retrieve previous consultations for continuity of care.

Use Cases:
- Track symptom progression over time
- Provide doctor with consultation history
- Improve personalization based on history
```

## 🔧 Development Setup

### Prerequisites
- Python 3.8+
- Git
- Virtual environment tool (venv or conda)

### Setup Steps

```bash
# 1. Fork and clone the repository
git clone https://github.com/yourusername/SmartClinic-AI.git
cd SmartClinic-AI

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies (including dev tools)
pip install -r requirements.txt
pip install pytest black flake8 mypy  # Development tools

# 4. Create .env file
cp .env.example .env
# Edit .env with your OpenRouter API key

# 5. Run tests
pytest test_core.py -v
```

## 📝 Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix-name
```

Use descriptive branch names:
- `feature/add-patient-history` ✅
- `fix/m1-predictor-crash` ✅
- `docs/improve-readme` ✅
- `test` ❌
- `fix-bug` ❌

### 2. Make Your Changes

**Code Style Guidelines**:
- Follow **PEP 8** style guide
- Use **4 spaces** for indentation (not tabs)
- Max line length: **100 characters**
- Use meaningful variable names
- Add docstrings to functions and classes

**Example**:
```python
def calculate_symptom_severity(symptoms: list[str], weights: dict[str, float]) -> float:
    """
    Calculate total symptom severity based on weights.
    
    Args:
        symptoms: List of identified symptoms
        weights: Dictionary mapping symptoms to severity weights
        
    Returns:
        Total weighted severity score (0-100)
    """
    if not symptoms:
        return 0.0
    
    total = sum(weights.get(s, 0) for s in symptoms)
    return min(total / len(symptoms), 100.0)
```

### 3. Commit Your Changes

**Write clear commit messages**:
```bash
# Good ✅
git commit -m "Add patient history tracking feature"
git commit -m "Fix M1 predictor crash when PKL missing"
git commit -m "Improve Arabic text normalization"

# Bad ❌
git commit -m "changes"
git commit -m "fix stuff"
git commit -m "WIP"
```

### 4. Test Your Changes

```bash
# Run existing tests
pytest test_core.py -v

# Run code quality checks
flake8 core.py controller.py
black --check core.py

# Test manually
python controller.py
# Then test endpoints at http://localhost:5000
```

### 5. Update Documentation

- Update README.md if adding features
- Add docstrings to new functions
- Add comments for complex logic
- Update ROADMAP if changing plans

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:

**PR Title**: Clear, descriptive title  
**PR Description**: Include:
- What changes you made
- Why you made them
- Related issues (fixes #123)
- Testing done
- Any breaking changes

**Example**:
```markdown
## Description
Added patient history tracking feature to allow users to save and retrieve previous consultations.

## Changes
- Added history.py module for persistence
- Modified controller.py to add new endpoints
- Updated frontend to display history

## Related Issues
Fixes #45

## Testing
- Tested saving and loading multiple consultations
- Verified history displays correctly in UI
- Checked performance with 100+ records

## Breaking Changes
None
```

## ✅ Code Review Process

1. **Automated Checks**: Tests and linting run automatically
2. **Code Review**: Maintainers review code quality and correctness
3. **Feedback**: We'll request changes if needed
4. **Approval**: Once approved, we'll merge your PR

**Review Criteria**:
- ✅ Code follows style guidelines
- ✅ Tests pass and cover new code
- ✅ Documentation is updated
- ✅ No hardcoded secrets or sensitive data
- ✅ Performance impact is acceptable
- ✅ Backward compatibility maintained (when possible)

## 🧪 Testing Guidelines

### Writing Tests

```python
# test_new_feature.py
import pytest
from new_module import new_function

def test_new_function_success():
    """Test successful case"""
    result = new_function("input")
    assert result == "expected_output"

def test_new_function_empty_input():
    """Test edge case: empty input"""
    result = new_function("")
    assert result is None

def test_new_function_error():
    """Test error handling"""
    with pytest.raises(ValueError):
        new_function(None)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest test_core.py -v

# Run with coverage
pytest --cov=. test_core.py
```

## 📋 Checklist Before Submitting PR

- [ ] Code follows PEP 8 style guide
- [ ] Tests pass locally (`pytest test_*.py`)
- [ ] Code quality checks pass (`flake8`, `black`)
- [ ] No hardcoded secrets or API keys
- [ ] `.env` file not committed
- [ ] Documentation updated (README, docstrings)
- [ ] PR description is clear and complete
- [ ] Related issues are referenced
- [ ] Commit messages are descriptive
- [ ] No breaking changes (or clearly documented)

## 📖 Documentation Standards

### Function Docstrings

```python
def predict_department(symptoms: list[str]) -> dict:
    """
    Predict medical department based on symptoms.
    
    Args:
        symptoms: List of symptom strings
        
    Returns:
        Dictionary with:
            - 'department': Predicted department name
            - 'confidence': Confidence score (0-1)
            - 'alternatives': List of alternative departments
            
    Raises:
        ValueError: If symptoms list is empty
        
    Example:
        >>> result = predict_department(['fever', 'cough'])
        >>> result['department']
        'Respiratory'
    """
```

### Comments

```python
# Good: Explains why, not what
# Use weighted ensemble to improve robustness across different symptom patterns
predictions = ensemble_model.predict(features)

# Bad: Restates obvious code
# Loop through predictions
for pred in predictions:
    ...
```

## 🚀 Performance Considerations

When contributing, consider:

- **Algorithm efficiency**: Don't add O(n²) where O(n) works
- **Memory usage**: Avoid loading entire datasets unnecessarily
- **API calls**: Cache LLM responses when possible
- **UI responsiveness**: Async operations for long tasks

## ♿ Accessibility

- Ensure text is clear and concise
- Provide alt text for images
- Test with different languages (Arabic/English)
- Consider color-blind friendly palettes

## 🌐 Localization

We support Arabic and English. When contributing:

- Add translations for UI strings
- Test both language modes
- Use i18n libraries appropriately
- Don't hardcode language-specific text

## ❓ Questions?

- **GitHub Issues**: Ask publicly to help others
- **GitHub Discussions**: For general questions
- **Email**: team@smartclinic.ai

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to SmartClinic AI! 🎉**
