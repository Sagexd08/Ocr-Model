# 🤝 Contributing to Enterprise OCR Processing System

Thank you for your interest in contributing to the Enterprise OCR Processing System! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Development Standards](#development-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## 📜 Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming environment for everyone.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- 8GB RAM (16GB recommended)
- 10GB free disk space

### Fork and Clone

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Ocr-Model.git
   cd Ocr-Model
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/Sagexd08/Ocr-Model.git
   ```

## 🔧 Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt

# Web interface dependencies
pip install -r requirements_streamlit.txt
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

### 4. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=worker --cov-report=html

# Run specific test file
pytest tests/unit/test_document_processor.py
```

### 5. Start Development Server

```bash
# Start the web interface
python launch_advanced_ocr.py

# Or start individual components
streamlit run advanced_ocr_app.py --server.port 8505
uvicorn api.main:app --reload --port 8000
```

## 📝 Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

- 🐛 **Bug fixes**
- ✨ **New features**
- 📚 **Documentation improvements**
- 🧪 **Tests**
- 🎨 **UI/UX improvements**
- ⚡ **Performance optimizations**
- 🔧 **Refactoring**

### Contribution Workflow

1. **Check existing issues** to avoid duplicates
2. **Create an issue** for new features or bugs
3. **Discuss the approach** before starting work
4. **Create a feature branch** from `main`
5. **Make your changes** following our standards
6. **Write tests** for your changes
7. **Update documentation** as needed
8. **Submit a pull request**

## 🔄 Pull Request Process

### Before Submitting

- [ ] Code follows our style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** in development environment
4. **Approval** from at least one maintainer
5. **Merge** to main branch

## 🐛 Issue Reporting

### Bug Reports

Use the bug report template and include:

- **Environment details** (OS, Python version, dependencies)
- **Steps to reproduce** the issue
- **Expected behavior** vs actual behavior
- **Error messages** and stack traces
- **Sample files** (if applicable)

### Feature Requests

Use the feature request template and include:

- **Problem description** you're trying to solve
- **Proposed solution** or approach
- **Alternative solutions** considered
- **Use cases** and examples

### Issue Labels

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Improvements to docs
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `priority:high` - High priority issue
- `priority:low` - Low priority issue

## 📏 Development Standards

### Code Style

We follow PEP 8 with some modifications:

```python
# Use Black for formatting
black .

# Use isort for imports
isort .

# Use flake8 for linting
flake8 .

# Use mypy for type checking
mypy worker/
```

### Naming Conventions

- **Functions and variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Files and modules**: `snake_case`

### Documentation Strings

```python
def process_document(document_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a document using OCR and extract text.
    
    Args:
        document_path: Path to the document file
        params: Processing parameters
        
    Returns:
        Dictionary containing processing results
        
    Raises:
        ProcessingError: If document processing fails
        FileNotFoundError: If document file doesn't exist
    """
    pass
```

### Type Hints

Use type hints for all function signatures:

```python
from typing import Dict, List, Optional, Union

def extract_text(
    image: np.ndarray,
    confidence_threshold: float = 0.8
) -> List[Dict[str, Union[str, float]]]:
    """Extract text from image with confidence scores."""
    pass
```

## 🧪 Testing

### Test Structure

```
tests/
├── unit/                   # Unit tests
│   ├── test_document_processor.py
│   ├── test_ocr_models.py
│   └── test_pipeline.py
├── integration/            # Integration tests
│   ├── test_api.py
│   └── test_full_pipeline.py
├── fixtures/               # Test data
│   ├── sample_documents/
│   └── expected_results/
└── conftest.py            # Pytest configuration
```

### Writing Tests

```python
import pytest
from worker.document_processor import EnhancedDocumentProcessor

class TestDocumentProcessor:
    @pytest.fixture
    def processor(self):
        return EnhancedDocumentProcessor()
    
    def test_process_pdf_document(self, processor):
        """Test PDF document processing."""
        result = processor.process_document(
            job_id="test",
            document_path="tests/fixtures/sample.pdf",
            params={"mode": "advanced"}
        )
        
        assert result["status"] == "completed"
        assert result["summary"]["page_count"] > 0
        assert result["summary"]["word_count"] > 0
    
    @pytest.mark.parametrize("profile", ["performance", "quality", "balanced"])
    def test_processing_profiles(self, processor, profile):
        """Test different processing profiles."""
        result = processor.process_document(
            job_id="test",
            document_path="tests/fixtures/sample.pdf",
            params={"profile": profile}
        )
        
        assert result["status"] == "completed"
```

### Test Coverage

Maintain test coverage above 80%:

```bash
# Generate coverage report
pytest --cov=worker --cov-report=html --cov-report=term

# View coverage
open htmlcov/index.html
```

## 📚 Documentation

### Documentation Types

1. **API Documentation**: Docstrings and OpenAPI specs
2. **User Guides**: How-to guides and tutorials
3. **Developer Docs**: Architecture and development guides
4. **README**: Project overview and quick start

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add screenshots for UI features
- Keep documentation up to date with code changes

### Documentation Tools

- **Sphinx**: For API documentation
- **MkDocs**: For user guides
- **Mermaid**: For diagrams and flowcharts

## 🏷️ Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## 🎯 Development Roadmap

### Current Priorities

1. **Performance Optimization**: Improve processing speed
2. **Advanced Analytics**: Enhanced reporting and insights
3. **Mobile Support**: Mobile-responsive interface
4. **Cloud Integration**: Native cloud storage support

### How to Contribute to Roadmap

- Review [GitHub Projects](https://github.com/Sagexd08/Ocr-Model/projects)
- Participate in [Discussions](https://github.com/Sagexd08/Ocr-Model/discussions)
- Submit feature requests with detailed use cases

## 🆘 Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Discord**: Real-time chat and support
- **Email**: maintainers@ocr-system.com

### Mentorship

New contributors can request mentorship:

1. Comment on a "good first issue"
2. Tag `@maintainers` for guidance
3. Join our Discord for real-time help

## 🙏 Recognition

Contributors are recognized in:

- **README**: Contributors section
- **CHANGELOG**: Release notes
- **GitHub**: Contributor graphs
- **Discord**: Contributor role

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

<div align="center">

**🤝 Thank you for contributing to Enterprise OCR Processing System!**

*Together, we're building the future of document processing*

</div>
