# shellcheck shell=bash
# Source this file before any run:   source setup_env.sh
#
# Sets PRAXIS_ROOT, activates the Python virtual environment, sources
# ROS 2 Jazzy, and configures Python and Gazebo resource paths.
#
# Order matters: the virtual environment is activated first and ROS second,
# matching the configuration on which the reported results were produced.
# ROS setup scripts reference unset variables, so nounset is suspended
# while they load and restored afterwards.

_praxis_had_u=0
case $- in *u*) _praxis_had_u=1 ;; esac
set +u

PRAXIS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PRAXIS_ROOT
_praxis_venv="${PRAXIS_VENV:-$PRAXIS_ROOT/venv}"

if [ -f "$_praxis_venv/bin/activate" ]; then
  . "$_praxis_venv/bin/activate"
else
  echo "WARN: no virtual environment at $_praxis_venv" >&2
  echo "      run environment/install_ubuntu2404.sh, or export PRAXIS_VENV=<path>" >&2
fi

if [ "${PRAXIS_NO_ROS:-0}" = 1 ]; then
  :   # self-tests only: keep stub ROS modules ahead of the real ones
elif [ -f /opt/ros/jazzy/setup.bash ]; then
  . /opt/ros/jazzy/setup.bash
else
  echo "WARN: /opt/ros/jazzy/setup.bash not found; simulator stages will not run" >&2
fi

export PYTHONPATH="$PRAXIS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GZ_SIM_RESOURCE_PATH="$PRAXIS_ROOT/worlds:$PRAXIS_ROOT/assets:$PRAXIS_ROOT/assets/ycb/gazebo_models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"

[ "$_praxis_had_u" = 1 ] && set -u
unset _praxis_had_u _praxis_venv
echo "PRAXIS_ROOT=$PRAXIS_ROOT"
