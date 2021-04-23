python setup.py clean sdist

# Upload all files for distribution by using twine
# Check "~/.pypirc" for configuration
twine upload dist/*
