# Method 1: Basic requirements.txt creation
# Make sure your virtual environment is activated, then:
pip freeze > requirements.txt

# Method 2: Create requirements.txt with versions
pip list --format=freeze > requirements.txt

# Method 3: Create requirements with specific packages
pip freeze | grep -E "numpy|pandas|matplotlib|seaborn" > requirements.txt

# Install packages from requirements.txt
pip install -r requirements.txt

# Update all packages in requirements.txt
pip install --upgrade -r requirements.txt

# Check outdated packages
pip list --outdated

# Generate requirements.txt with package versions
pip freeze | sed 's/==/>=/g' > requirements.txt

# Clean up requirements.txt (remove unnecessary dependencies)
pip-compile requirements.in

# Verify installations
pip list --format=columns