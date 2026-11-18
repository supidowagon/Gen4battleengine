#!/usr/bin/env bash
# POD SIDE, once per fresh pod (ubuntu:24.04). Installs tools, uv, Python 3.12, runpodctl,
# and pre-clones the decomp. Run detached; log /work/bootstrap.log; ends with BOOTSTRAP_OK.
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
mkdir -p /work/_bin /work/_decomp /work/_uvcache
command -v git >/dev/null && command -v /usr/bin/time >/dev/null || {
  apt-get update -qq; apt-get install -y -qq git curl time xz-utils ca-certificates procps >/dev/null; }
[ -x /root/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
if [ ! -x /work/_bin/runpodctl ]; then
  curl -LsSf -o /work/_bin/runpodctl https://github.com/runpod/runpodctl/releases/download/v3.1.1/runpodctl-linux-amd64
  echo "ae0a2e5ff723c8fceb609e7120dfcaa4d681bf3015adc74141812a21f2bb9486  /work/_bin/runpodctl" | sha256sum -c -
  chmod +x /work/_bin/runpodctl
fi
UV_PYTHON_INSTALL_DIR=/work/_python /root/.local/bin/uv python install 3.12
[ -d /work/_decomp/repo ] || git clone -q https://github.com/pret/pokeplatinum /work/_decomp/repo
echo BOOTSTRAP_OK
