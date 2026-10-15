# cpu-runner image for FlagGems pre-check CI (rule-check + linter/code-style).
#
# These jobs are pure-CPU lint / YAML / registry checks — they never touch the
# Kunlunxin XPU — so the image is based on plain Ubuntu rather than a vendor
# base image. The runner pod happens to be scheduled on a Kunlunxin host, but
# the container needs no XPU driver/runtime; keeping it to Ubuntu avoids
# dragging in the heavy vendor stack.
#
# Everything the jobs need is baked in so the workflows install NOTHING at run
# time:
#   - node20 + jq : GitHub JS actions (checkout@v7, github-script@v9,
#                   changed-files@v47, action-wait-for-workflow) run on the
#                   container's node; jq is used by rule-check's gate step.
#   - git         : actions/checkout and pre-commit hook clones.
#   - python3.10 + pyyaml : pyyaml is the only third-party import in
#                   tools/ci_checks/*.py (pyproject requires-python >=3.10).
#   - pre-commit + prebuilt hook venvs (PRE_COMMIT_HOME) so the ~128s per-job
#     "Installing environment" rebuild (building a venv + pip-installing
#     black/isort/flake8/clang-format for each hook) does not happen. The slow
#     part is building these venvs, not cloning the hook repos; clang-format is
#     a python wheel, so no system clang/llvm is required.
#
# The hook config is inlined at build time (see bottom), so the build needs no
# FlagGems checkout and can run from any directory. After this image ships,
# rule-check.yaml drops its `pip install pyyaml` step and linter.yml drops the
# `|| pip install pre-commit` fallback (both already done in the same change).
# The runner MUST export PRE_COMMIT_HOME=/opt/pre-commit-cache at run time
# (ARC/pod env), or pre-commit falls back to $HOME/.cache and rebuilds per job.

FROM harbor.baai.ac.cn/baai-infra/airs/ubuntu2204-base:1.10.0-build81

ENV DEBIAN_FRONTEND=noninteractive

# --- base tools: node20, jq, git, python3.10 + pip -------------------------
# Purge any distro node first: some bases ship nodejs/libnode-dev 12.x, and the
# nodesource node20 package then fails to unpack ("trying to overwrite
# /usr/include/node/common.gypi, which is also in package libnode-dev").
RUN set -eux; \
    apt-get update; \
    apt-get purge -y nodejs libnode-dev libnode72 npm 2>/dev/null || true; \
    apt-get autoremove -y 2>/dev/null || true; \
    apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg jq git \
        python3 python3-pip python3-venv; \
    mkdir -p /etc/apt/keyrings; \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg; \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends nodejs; \
    node --version; npm --version; jq --version; git --version; \
    apt-get clean; rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# --- python tooling --------------------------------------------------------
RUN set -eux; \
    python3 -m pip install --no-cache-dir --upgrade pip; \
    python3 -m pip install --no-cache-dir pyyaml pre-commit; \
    python3 --version; \
    pre-commit --version; \
    python3 -c "import yaml; print('pyyaml', yaml.__version__)"

# --- pre-warm pre-commit hook environments ---------------------------------
# Build every hook's repo clone + venv at image build time into a FIXED path.
# The runner MUST set PRE_COMMIT_HOME=/opt/pre-commit-cache at run time or this
# baked cache is ignored. Hook repos are gitcode.com mirrors (see
# .pre-commit-config.yaml), so the BUILD machine must reach gitcode.com.
# When .pre-commit-config.yaml hook revs change, pre-commit incrementally
# rebuilds only the changed hooks at run time; rebuild this image on drift.
ENV PRE_COMMIT_HOME=/opt/pre-commit-cache
# Pre-build every hook's isolated venv into a FIXED path so no venv is built at
# run time. `pre-commit install-hooks` only reads a config's repo+rev to build
# the envs (it does NOT run the local hooks' entry scripts), so we inline a
# minimal config here instead of depending on a repo checkout — no git clone of
# the FlagGems repo (the build host cannot reach github.com) and no COPY of a
# repo file (build is context-independent). The hook repos are gitcode.com
# mirrors, which the build host CAN reach.
#
# KEEP THE repo+rev LIST BELOW IN SYNC WITH .pre-commit-config.yaml. If a rev
# drifts, pre-commit incrementally rebuilds only the changed hook at run time
# (correct but slower); rebuild this image to restore the fast path. The
# local-repo hooks (sort-exports/check-*) use the system python + pyyaml already
# installed above, so they need no prebuilt venv and are omitted here.
RUN set -eux; \
    mkdir -p /tmp/hookbuild; \
    cd /tmp/hookbuild; \
    printf '%s\n' \
      'repos:' \
      '- repo: https://gitcode.com/gh_mirrors/pr/pre-commit-hooks.git' \
      '  rev: v2.3.0' \
      '  hooks:' \
      '  - id: check-yaml' \
      '  - id: end-of-file-fixer' \
      '  - id: trailing-whitespace' \
      '  - id: flake8' \
      '- repo: https://gitcode.com/pre-commit-clang/mirrors-clang-format' \
      '  rev: v13.0.0' \
      '  hooks:' \
      '    - id: clang-format' \
      '- repo: https://gitcode.com/GitHub_Trending/is/isort' \
      '  rev: 5.12.0' \
      '  hooks:' \
      '    - id: isort' \
      '- repo: https://gitcode.com/GitHub_Trending/bl/black' \
      '  rev: 26.5.1' \
      '  hooks:' \
      '    - id: black' \
      '    - id: black-jupyter' \
      > .pre-commit-config.yaml; \
    git init -q .; \
    git config --global --add safe.directory /tmp/hookbuild; \
    pre-commit install-hooks; \
    cd /; \
    rm -rf /tmp/hookbuild; \
    chmod -R a+rX "$PRE_COMMIT_HOME"
