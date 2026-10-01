#!/usr/bin/env bash
# Install the software environment on Ubuntu 24.04 (Noble).
#
# Steps already satisfied are skipped, so this is safe to re-run.
# ROS installation follows the official instructions:
#   https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html
# If an apt step fails (repository keys and packaging change over time),
# follow that page for ROS, then re-run this script to finish the Python setup.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
say() { echo; echo "==> $*"; }

. /etc/os-release
if [[ "${VERSION_CODENAME:-}" != "noble" ]]; then
  echo "WARNING: tested on Ubuntu 24.04 (noble); this system is ${PRETTY_NAME:-unknown}."
fi

# ------------------------------------------------------------------ ROS 2
if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  say "ROS 2 Jazzy already installed; skipping"
else
  say "Installing ROS 2 Jazzy"
  sudo apt-get update
  sudo apt-get install -y software-properties-common curl
  sudo add-apt-repository -y universe
  ver="$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
         | grep -F '"tag_name"' | awk -F'"' '{print $4}')"
  [[ -n "$ver" ]] || { echo "Could not determine ros-apt-source version; see the URL above."; exit 1; }
  curl -fL -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ver}/ros2-apt-source_${ver}.${VERSION_CODENAME}_all.deb"
  sudo dpkg -i /tmp/ros2-apt-source.deb
  sudo apt-get update
  sudo apt-get install -y ros-jazzy-desktop
fi

# ------------------------------------------------------- Gazebo + bridge
if dpkg -s ros-jazzy-ros-gz >/dev/null 2>&1; then
  say "ros-jazzy-ros-gz (Gazebo Harmonic + bridge) already installed; skipping"
else
  say "Installing Gazebo Harmonic and the ROS-Gazebo bridge"
  sudo apt-get install -y ros-jazzy-ros-gz
fi

# --------------------------------------------------------------- Python
say "Python virtual environment at $REPO/venv"
sudo apt-get install -y python3-venv python3-pip
if [[ ! -d "$REPO/venv" ]]; then
  python3 -m venv "$REPO/venv"
fi
# shellcheck disable=SC1091
. "$REPO/venv/bin/activate"
python -m pip install --upgrade pip
if [[ -f "$REPO/requirements-freeze.txt" ]]; then
  echo "Installing exact versions from requirements-freeze.txt"
  # Freeze files from a venv with system site packages can list ROS packages
  # (installed by apt, absent from PyPI) or local file:// references.
  grep -vE ' @ file://|^-e ' "$REPO/requirements-freeze.txt" > /tmp/praxis-req.txt
  if ! python -m pip install -r /tmp/praxis-req.txt; then
    echo "WARNING: pinned install failed; falling back to requirements.txt."
    echo "         Compare installed versions against requirements-freeze.txt."
    python -m pip install -r "$REPO/requirements.txt"
  fi
else
  echo "requirements-freeze.txt not found; installing unpinned requirements.txt"
  python -m pip install -r "$REPO/requirements.txt"
fi
python -m pip install pyyaml

say "Verifying"
set +u
# shellcheck disable=SC1091
. /opt/ros/jazzy/setup.bash
set -u
python -c "import rclpy, stable_baselines3, gymnasium, torch, pandas, scipy; print('Python imports OK; torch CUDA available:', torch.cuda.is_available())"
gz sim --version | head -1 || echo "WARNING: 'gz sim --version' failed"

echo
echo "Done. Next:"
echo "  source setup_env.sh && ./tests/run_self_tests.sh && ./run_all.sh check"
