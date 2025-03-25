# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate


#!/bin/bash

# Create installation directory
INSTALL_DIR="$HOME/gromacs"
BUILD_DIR="$HOME/gromacs_build"
mkdir -p $INSTALL_DIR
mkdir -p $BUILD_DIR

# Install dependencies (for Ubuntu/Debian)
if [ -x "$(command -v apt-get)" ]; then
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        cmake \
        libfftw3-dev \
        libopenmpi-dev \
        wget
# For CentOS/RHEL
elif [ -x "$(command -v yum)" ]; then
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y \
        cmake \
        fftw-devel \
        openmpi-devel \
        wget
# For macOS using Homebrew
elif [ -x "$(command -v brew)" ]; then
    brew install \
        cmake \
        fftw \
        open-mpi \
        wget
else
    echo "Unsupported package manager. Please install dependencies manually."
    exit 1
fi

# Download GROMACS 5.1.2
cd $BUILD_DIR
wget http://ftp.gromacs.org/pub/gromacs/gromacs-5.1.2.tar.gz
tar xvf gromacs-5.1.2.tar.gz
cd gromacs-5.1.2

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. \
    -DGMX_BUILD_OWN_FFTW=ON \
    -DREGRESSIONTEST_DOWNLOAD=OFF \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR

# Build and install
make -j$(nproc)
make install

# Add GROMACS to PATH
echo "" >> $HOME/.bashrc
echo "# GROMACS environment variables" >> $HOME/.bashrc
echo "source $INSTALL_DIR/bin/GMXRC" >> $HOME/.bashrc

# Clean up
cd $HOME
rm -rf $BUILD_DIR

echo "GROMACS 5.1.2 has been installed successfully!"
echo "Please run 'source ~/.bashrc' to update your environment variables"
echo "or start a new terminal session."

pip install -r requirements.txt