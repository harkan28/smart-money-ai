# Contributing to Smart Money AI

We love your input! We want to make contributing to Smart Money AI as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker](https://github.com/yourusername/smart-money-ai/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/smart-money-ai/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/smart-money-ai.git
cd smart-money-ai
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development dependencies
```

4. Run tests
```bash
python -m pytest tests/
```

## Code Style

* Use Python 3.8+ features
* Follow PEP 8 style guide
* Use type hints where possible
* Write docstrings for all public functions and classes
* Keep line length under 100 characters

## Testing

* Write tests for any new functionality
* Ensure all tests pass before submitting PR
* Aim for high test coverage
* Use meaningful test names and assertions

## Documentation

* Update README.md if needed
* Document new APIs in docstrings
* Update examples if functionality changes
* Keep comments concise and helpful

## License

By contributing, you agree that your contributions will be licensed under its MIT License.