%global debug_package %{nil}

# FlagGems Phase 1: pure Python wheel (FLAGGEMS_BUILD_C_EXTENSIONS=OFF).
# C++ extension split (libflaggems + libflaggems-dev) deferred to Phase 2.

Name:           python3-flag-gems
Version:        5.3.0
Release:        1%{?dist}
Summary:        FlagGems — GPU operator library for FlagOS (Phase 1, Python-only)

License:        Apache-2.0
URL:            https://github.com/flagos-ai/FlagGems
Source0:        %{url}/archive/refs/tags/v%{version}.tar.gz#/flag-gems-%{version}.tar.gz
# The upstream build backend is plain setuptools.build_meta since the
# scikit-build retirement; the Phase 1 wheel is py3-none-any, so the
# rpm is noarch (matches Architecture: all on the deb side).
BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  python3-setuptools >= 64
BuildRequires:  python3-wheel
BuildRequires:  python3-pip
BuildRequires:  pyproject-rpm-macros
# setuptools-scm resolves the version (no .git in the source tarball,
# hence the SETUPTOOLS_SCM_PRETEND_VERSION export in %%build)
BuildRequires:  python3-setuptools_scm >= 8

# Filter the auto-generated Requires for: torch + numpy/pyyaml/sqlalchemy/packaging.
# Reason: torch: distro version is CPU-only. numpy/pyyaml/sqlalchemy/packaging: distro has them but FlagGems pyproject uses == pins that distro versions do not match; we Require them below without a version constraint instead.
# See packaging/INSTALL.md (or future flagos-packaging install docs) for the
# user-side pip install incantation.
%global __requires_exclude ^python3(\.[0-9]+)?dist\((torch|numpy|pyyaml|sqlalchemy|packaging)\)$
# Hand-written distro deps (versions left open — distro newer is fine):
Requires:       python3-numpy
Requires:       python3-pyyaml
Requires:       python3-sqlalchemy
Requires:       python3-packaging

# Triton runtime dep — any FlagTree backend satisfies this (libblas3
# pattern, see ADR-002). Not yet active until FlagTree adds the
# Provides: python3-flagtree-backend declaration; leaving plain dep
# names here pending that change.
Recommends:     python3-flagtree-nvidia

%description
FlagGems is an operator library for large language models implemented in
the Triton language, providing a multi-backend interface for diverse AI
hardware platforms.

This Phase 1 RPM ships the pure-Python distribution (the wheel produced
without C++ extensions enabled), matching `pip install flag_gems` default
behavior. Phase 2 will split the C++ operators runtime into libflaggems
and headers into libflaggems-dev.

%prep
%autosetup -n flag-gems-%{version}

# The C++ operators runtime lives in the separate cpp/ tree (its own
# flag-gems-cpp wheel upstream); this Phase 1 rpm builds the pure-Python
# wheel only, aligning with Phase 1 scope.
%build
# The source tarball carries no .git metadata; pin the scm version.
export SETUPTOOLS_SCM_PRETEND_VERSION=%{version}
%pyproject_wheel

%install
%pyproject_install
export SETUPTOOLS_SCM_PRETEND_VERSION=%{version}
%pyproject_save_files flag_gems flaggems_tests flaggems_benchmark

%check
# Smoke find_spec test — verifies module lands at expected sitelib path.
# Doesn't actually import flag_gems because that triggers torch + triton
# imports, neither of which is in the build container (and shouldn't be:
# those are install-time runtime concerns).
# PYTHONSAFEPATH=1 keeps the cwd (the unpacked source tree, which also
# contains flag_gems/) off sys.path, so find_spec resolves against the
# installed copy under PYTHONPATH (sitearch/sitelib), not the source tree.
PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 \
    PYTHONPATH=%{buildroot}%{python3_sitearch}:%{buildroot}%{python3_sitelib} \
    python3 -c "import importlib.util; s = importlib.util.find_spec('flag_gems'); assert s and s.origin, 'flag_gems not findable'; print('OK: flag_gems at', s.origin)"

%files -f %{pyproject_files}
%license LICENSE
%{_bindir}/flaggems-setup

%changelog
* Mon Jul 20 2026 FlagOS Contributors <contact@flagos.io> - 5.3.0-1
- Update to 5.3.0; follow upstream switch to the setuptools build backend
  (noarch wheel, drop scikit-build/pybind11/cmake/ninja build deps).
- Package the bundled flaggems_tests and flaggems_benchmark suites.

* Thu May 21 2026 FlagOS Contributors <contact@flagos.io> - 5.0.2-1
- Initial RPM packaging (Phase 1, Python-only, no C++ extension).
