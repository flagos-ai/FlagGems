%global debug_package %{nil}

# FlagGems Phase 1: pure Python wheel (FLAGGEMS_BUILD_C_EXTENSIONS=OFF).
# C++ extension split (libflaggems + libflaggems-dev) deferred to Phase 2.

Name:           python3-flag-gems
Version:        5.0.2
Release:        1%{?dist}
Summary:        FlagGems — GPU operator library for FlagOS (Phase 1, Python-only)

License:        Apache-2.0
URL:            https://github.com/flagos-ai/FlagGems
Source0:        %{url}/archive/v%{version}/flag-gems-%{version}.tar.gz
# Stay aligned with the wheel's actual arch tag (scikit-build-core tags
# linux_x86_64 even when FLAGGEMS_BUILD_C_EXTENSIONS=OFF) and with the
# Debian `Architecture: amd64`. See packaging/NOTES.md.
BuildArch:      x86_64
BuildRequires:  python3-devel
BuildRequires:  python3-setuptools >= 64
BuildRequires:  python3-wheel
BuildRequires:  python3-pip
BuildRequires:  python3-pybind11
BuildRequires:  pyproject-rpm-macros
# scikit-build-core for the pyproject build-backend
BuildRequires:  python3-scikit-build-core
BuildRequires:  cmake
BuildRequires:  ninja-build

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

%build
%pyproject_wheel

%install
%pyproject_install
%pyproject_save_files flag_gems

%check
# Smoke find_spec test — verifies module lands at expected sitelib path.
# Doesn't actually import flag_gems because that triggers torch + triton
# imports, neither of which is in the build container (and shouldn't be:
# those are install-time runtime concerns).
PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=%{buildroot}%{python3_sitearch}:%{buildroot}%{python3_sitelib} \
    python3 -c "import importlib.util; s = importlib.util.find_spec('flag_gems'); assert s and s.origin, 'flag_gems not findable'; print('OK: flag_gems at', s.origin)"

%files -f %{pyproject_files}
%license LICENSE*

%changelog
* Thu May 21 2026 FlagOS Contributors <contact@flagos.io> - 5.0.2-1
- Initial RPM packaging (Phase 1, Python-only, no C++ extension).
