# Install pip
python3 -m pip install --user --upgrade pip
# Install virtualenv
python3 -m pip install --user virtualenv
sudo apt-get install -y python3-venv
# Create virtual environment
python3 -m venv env
# Activate virtual environment
source env/bin/activate
# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
