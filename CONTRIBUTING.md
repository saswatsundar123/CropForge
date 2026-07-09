# Contributing to CropForge

First off, thank you for considering contributing to CropForge! It's researchers, agronomists, and developers like you that make this open-source virtual farm runtime a powerful tool for agricultural science.

Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open source project. In return, they should reciprocate that respect in addressing your issue, assessing changes, and helping you finalize your pull requests.

We are an open-source project and we love to receive contributions from our community — you! There are many ways to contribute: writing tutorials or agronomy blog posts, improving the documentation, submitting bug reports, requesting new machinery or crop models, or writing code and GLTF 3D assets which can be incorporated into CropForge itself.

Please, don't use the issue tracker for general agricultural or agronomy support questions that are not related to the CropForge simulation engine. For general questions, Stack Overflow or agricultural science forums are a better fit.

# Ground Rules

### Responsibilities
* Ensure that any code added to core meets all requirements, especially our test coverage (`tests/`).
* Don't add complex abstractions unless absolutely necessary. We prefer explicit, readable, Pythonic code.
* Ensure cross-platform compatibility (Windows, macOS, Linux).
* Create issues for any major features or architectural changes you wish to make. Discuss things transparently and get community feedback before writing significant code.
* Keep feature pull requests as small as possible.
* Be welcoming to newcomers and encourage diverse new contributors from all backgrounds. We adhere to the [Python Community Code of Conduct](https://www.python.org/psf/codeofconduct/).

# Your First Contribution

Unsure where to begin contributing to CropForge? You can start by looking through these issue labels:
* **`good first issue`** - issues which should only require a few lines of code, and a test or two.
* **`help wanted`** - issues which should be a bit more involved, such as adding a new crop plugin or fine-tuning the 3D visualization.

Working on your first Pull Request? You can learn how from this *free* series, [How to Contribute to an Open Source Project on GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github). At this point, you're ready to make your changes! Feel free to ask for help; everyone is a beginner at first :seedling:

# Getting Started

For something that is bigger than a one or two line fix:

1. Create your own fork of the code repository.
2. Clone your fork locally.
3. Install development dependencies using `pip install cropforge".[dev,export]"` (requires Python 3.12+).
4. Create a new branch for your feature.
5. Make your changes in your fork.
6. Run the test suite using `pytest tests/ -q --tb=short` to ensure all tests pass.
7. Send a pull request.

### Obvious Fix Policy

Small contributions such as fixing spelling errors, formatting changes, comment clean up, or minor typo corrections can be submitted as a patch without prior discussion. As a rule of thumb, changes are obvious fixes if they do not introduce any new functionality or creative thinking.

# How to report a bug

### Security Disclosures
If you find a security vulnerability, do **NOT** open a public issue. Please email saswatsundar123@gmail.com directly. 

### Filing a Bug Report
When filing an issue, make sure to answer these five questions:
1. What version of CropForge are you using?
2. What operating system and Python version are you using?
3. What did you do? (Include a minimal code snippet if possible)
4. What did you expect to see?
5. What did you see instead?

# How to suggest a feature or enhancement

The CropForge philosophy is to provide a robust, open-source, code-first virtual farm runtime for agricultural researchers. We believe in high-performance spatial simulation without sacrificing the ease of Python. 

If you find yourself wishing for a feature that doesn't exist in CropForge (e.g., a new crop physiology model, new land preparation techniques, or 3D visual enhancements), you are probably not alone. Open an issue on our GitHub issues list describing the feature you would like to see, why you need it, and how it should work from a user's perspective.

# Community

If you have general questions or want to discuss CropForge development, you can open a Discussion on our GitHub repository. We actively monitor issues and discussions and try our best to respond within a few days.

# Code, commit message and labeling conventions

### Code Style
* We use standard **Python 3.12+ syntax**.
* Use **Type hints** on all public functions.
* Use **Docstrings** (NumPy style) on all public classes and methods.

### Issue Labels
We use standard labels to manage issues:
* `bug`: Something isn't working
* `enhancement`: New feature or request
* `crop-package`: Specific to first-party crop plugins (like StandardWheat)
* `documentation`: Improvements or additions to documentation
