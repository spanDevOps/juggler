# Contributing to LLM Juggler

Thank you for your interest in contributing to LLM Juggler! 🤹

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/spanDevOps/jugglerr/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - Relevant code snippets

### Suggesting Features

1. Check [Issues](https://github.com/spanDevOps/jugglerr/issues) for existing feature requests
2. Create a new issue with:
   - Clear use case
   - Proposed solution
   - Alternative approaches considered

### Contributing Code

1. **Fork the repository**
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run tests**: `pytest tests/`
6. **Format code**: `black llm_juggler/`
7. **Commit**: `git commit -m "Add feature: description"`
8. **Push**: `git push origin feature/your-feature-name`
9. **Create Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/spanDevOps/jugglerr.git
cd jugglerr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black llm_juggler/

# Lint code
flake8 llm_juggler/
```

### Code Style

- Follow PEP 8
- Use Black for formatting (line length: 120)
- Add type hints where appropriate
- Write docstrings for public functions
- Keep functions focused and small

### Testing

- Write tests for all new features
- Maintain test coverage above 80%
- Use pytest for testing
- Mock external API calls

### Documentation

- Update README.md for user-facing changes
- Update CHANGELOG.md
- Add docstrings to new functions
- Update examples if needed

## Adding New Models

To add a new model to the database:

1. Research model capabilities (use Perplexity or official docs)
2. Add entry to `llm_juggler/models.py`:

```python
'model-name': {
    'power': POWER_SUPER,  # or POWER_REGULAR
    'capabilities': [
        CAP_STREAMING,
        CAP_TOOL_CALLING,
        # ... other capabilities
    ],
    'context_window': CONTEXT_LARGE  # or MEDIUM or SMALL
}
```

3. Add tests
4. Update README.md
5. Submit PR

## Adding New Providers

To add a new provider:

1. Add API endpoint to `API_ENDPOINTS` in `juggler.py`
2. Add provider to `PROVIDER_PRIORITY`
3. Add rate limit header parsing logic (if available)
4. Add models to `models.py`
5. Update documentation
6. Add tests
7. Submit PR

## Questions?

- Open a [Discussion](https://github.com/spanDevOps/jugglerr/discussions)
- Ask in Issues
- Check existing documentation

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make LLM Juggler better! 🎪
