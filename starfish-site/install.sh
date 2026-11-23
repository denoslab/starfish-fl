#!/usr/bin/env bash
# Install starfish-site on Ubuntu or macOS, without admin rights.
#
#   starfish-site/install.sh [--cpu] [--home DIR] [--python PATH] [--no-link]
#
# Makes DIR/venv with Python 3.11, torch, and this checkout's controller and
# starfish-site packages. Both packages are installed editable, so a git pull
# updates them. DIR is ~/.starfish-site unless --home or STARFISH_SITE_HOME
# names another folder. The starfish-site command is linked into ~/.local/bin.
#
# torch comes as a build for CUDA 12.8 on Linux with an NVIDIA driver, which
# covers Ada and Blackwell cards such as the RTX 4070 Ti SUPER and the
# RTX PRO 6000, and as the standard build with Apple MPS on macOS. --cpu
# takes the CPU-only build on Linux.
#
# Python and the packages come through uv, which this script downloads into
# DIR/bin when it is missing, and Python itself goes to DIR/python. Deleting
# DIR and the link removes everything. --python PATH uses that Python 3.10
# or 3.11 with venv and pip in place of uv.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE_HOME="${STARFISH_SITE_HOME:-$HOME/.starfish-site}"
TORCH_SPEC='torch>=2.7,<3'
PYTHON_VERSION=3.11
BUILD=auto
PYTHON=
LINK=1

say() { printf '\n== %s\n' "$*"; }
die() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --cpu) BUILD=cpu ;;
    --home) SITE_HOME="$2"; shift ;;
    --python) PYTHON="$2"; shift ;;
    --no-link) LINK=0 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown option $1, see --help" ;;
  esac
  shift
done

[ -d "$REPO/controller/starfish" ] && [ -d "$REPO/starfish-site/starfish_site" ] ||
  die "run me from a starfish-fl checkout, $REPO has no controller or starfish-site"
VENV="$SITE_HOME/venv"

say "System"
OS="$(uname -s)"
ARCH="$(uname -m)"
TORCH_INDEX=
case "$OS" in
  Linux)
    [ "$ARCH" = x86_64 ] || die "Linux on $ARCH is not supported, only x86_64"
    if [ "$BUILD" = auto ]; then
      if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        BUILD=cu128
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/GPU /'
        DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
        [ "${DRIVER%%.*}" -ge 525 ] ||
          die "NVIDIA driver $DRIVER is older than 525, which CUDA 12 builds need. Update the driver or pass --cpu."
      else
        echo "No NVIDIA GPU found, taking the CPU-only torch build"
        BUILD=cpu
      fi
    fi
    if [ "$BUILD" = cu128 ]; then
      TORCH_INDEX=https://download.pytorch.org/whl/cu128
    else
      TORCH_INDEX=https://download.pytorch.org/whl/cpu
    fi
    ;;
  Darwin)
    if [ "$ARCH" = arm64 ]; then
      BUILD=mps
    else
      echo "An Intel Mac trains on the CPU only"
      BUILD=cpu
    fi
    ;;
  *) die "$OS is not supported, only Linux and macOS" ;;
esac
echo "$OS $ARCH, torch build $BUILD, site folder $SITE_HOME"

say "Python $PYTHON_VERSION"
UV=
if [ -z "$PYTHON" ]; then
  UV="$(command -v uv || true)"
  [ -n "$UV" ] || { [ -x "$SITE_HOME/bin/uv" ] && UV="$SITE_HOME/bin/uv"; } || true
  if [ -z "$UV" ]; then
    command -v curl >/dev/null 2>&1 || die "curl is needed to download uv"
    echo "Downloading uv into $SITE_HOME/bin"
    curl -LsSf https://astral.sh/uv/install.sh |
      env UV_INSTALL_DIR="$SITE_HOME/bin" UV_NO_MODIFY_PATH=1 sh
    UV="$SITE_HOME/bin/uv"
  fi
  # uv's own Python build, kept in the site folder, so no system Python is needed or changed
  export UV_PYTHON_INSTALL_DIR="$SITE_HOME/python"
  [ -x "$VENV/bin/python" ] ||
    "$UV" venv --python "$PYTHON_VERSION" --python-preference only-managed "$VENV"
  pip_install() { "$UV" pip install --python "$VENV/bin/python" "$@"; }
else
  "$PYTHON" -c 'import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] < (3, 12)))' ||
    die "$PYTHON is not Python 3.10 or 3.11"
  [ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  pip_install() { "$VENV/bin/python" -m pip install "$@"; }
fi
"$VENV/bin/python" --version

say "torch"
if [ -n "$TORCH_INDEX" ]; then
  pip_install "$TORCH_SPEC" --index-url "$TORCH_INDEX"
else
  pip_install "$TORCH_SPEC"
fi

say "Starfish controller and starfish-site"
pip_install -e "$REPO/controller" -e "$REPO/starfish-site"

if [ "$LINK" = 1 ]; then
  mkdir -p "$HOME/.local/bin"
  ln -sf "$VENV/bin/starfish-site" "$HOME/.local/bin/starfish-site"
  echo "Linked $HOME/.local/bin/starfish-site"
fi

say "Installed"
"$VENV/bin/python" - <<'EOF'
import torch
mps = getattr(torch.backends, 'mps', None)
print('torch', torch.__version__, '| CUDA build', torch.version.cuda or 'none',
      '| CUDA available', torch.cuda.is_available(),
      '| MPS available', bool(mps and mps.is_available()))
EOF
"$VENV/bin/starfish-site" --version

COMMAND=starfish-site
[ "$SITE_HOME" = "$HOME/.starfish-site" ] || COMMAND="starfish-site --home $SITE_HOME"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) [ "$LINK" = 1 ] && echo "Add ~/.local/bin to PATH, for example: echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.$(basename "${SHELL:-bash}")rc" ;;
esac
cat <<EOF

Next steps
  1. $COMMAND enrol --router <router URL> --code <code> --name "<site name>" --store <sample store>
  2. $COMMAND check
  3. $COMMAND start
EOF
