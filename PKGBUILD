# Maintainer: Your Name <your.email@example.com>
# AUR package for StudyWatchdog
# https://github.com/YOUR_USERNAME/studywatchdog

pkgname=studywatchdog
pkgver=0.1.0
pkgrel=1
pkgdesc="AI-powered study monitor that uses your webcam and SigLIP to detect if you're studying, and rickrolls you when you stop"
arch=('x86_64')
url="https://github.com/YOUR_USERNAME/studywatchdog"
license=('MIT')
depends=(
    'python>=3.12'
    'python-pytorch-cuda'     # torch with CUDA support
    'python-numpy'
    'python-pillow'
    'opencv'                  # opencv with Python bindings
    'python-pydantic'
    'python-pygame'
)
makedepends=(
    'python-build'
    'python-installer'
    'python-hatchling'
)
optdepends=(
    'cuda: GPU acceleration for faster inference'
)
source=("${pkgname}-${pkgver}.tar.gz::${url}/archive/v${pkgver}.tar.gz")
sha256sums=('SKIP')

build() {
    cd "${pkgname}-${pkgver}"
    python -m build --wheel --no-isolation
}

package() {
    cd "${pkgname}-${pkgver}"
    python -m installer --destdir="${pkgdir}" dist/*.whl

    # Install license
    install -Dm644 LICENSE "${pkgdir}/usr/share/licenses/${pkgname}/LICENSE"

    # Install default config as documentation
    install -Dm644 /dev/stdin "${pkgdir}/usr/share/doc/${pkgname}/config.toml.example" <<'CONF'
# Run `studywatchdog --generate-config` to create your personal config at
# ~/.config/studywatchdog/config.toml
# Or copy this file and edit it:
#   cp /usr/share/doc/studywatchdog/config.toml.example ~/.config/studywatchdog/config.toml
CONF
}
