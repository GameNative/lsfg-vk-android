#!/usr/bin/env bash
# Build lsfg-vk for GameNative's Android/Bionic Wine-on-Android runtime.
#
# Produces by default:
#   build-android-${ANDROID_ABI}/liblsfg-vk.so
#   build-android-${ANDROID_ABI}/dist/liblsfg-vk-${ANDROID_ABI}.so (stripped)
#   build-android-${ANDROID_ABI}/dist/VkLayer_LS_frame_generation.json
#
# Requirements:
#   ANDROID_NDK env var pointing at an Android NDK r25+.
#
# Usage:
#   ANDROID_NDK=/path/to/android-ndk-r27d ./scripts/build/android.sh [Release|Debug]
#   ANDROID_ABI=x86_64 ANDROID_NDK=/path/to/android-ndk-r27d ./scripts/build/android.sh

set -euo pipefail

BUILD_TYPE="${1:-Release}"
ABI="${ANDROID_ABI:-arm64-v8a}"
API="${ANDROID_PLATFORM:-android-28}"
GENERATOR="${CMAKE_GENERATOR:-Ninja}"

if [[ -z "${ANDROID_NDK:-}" ]]; then
    echo "error: ANDROID_NDK must be set to your NDK root (e.g. /opt/android-ndk-r27d)" >&2
    exit 1
fi

TOOLCHAIN="${ANDROID_NDK}/build/cmake/android.toolchain.cmake"
if [[ ! -f "${TOOLCHAIN}" ]]; then
    echo "error: toolchain not found at ${TOOLCHAIN}" >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-android-${ABI}}"
DIST_DIR="${DIST_DIR:-${BUILD_DIR}/dist}"

mkdir -p "${BUILD_DIR}" "${DIST_DIR}"

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
    -G "${GENERATOR}" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
    -DANDROID_ABI="${ABI}" \
    -DANDROID_PLATFORM="${API}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DLSFGVK_ANDROID_WINE=ON \
    -DVOLK_STATIC_DEFINES=VK_USE_PLATFORM_ANDROID_KHR \
    -DCMAKE_CXX_FLAGS="-DVK_USE_PLATFORM_ANDROID_KHR" \
    -DCMAKE_C_FLAGS="-DVK_USE_PLATFORM_ANDROID_KHR" \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-z,max-page-size=16384"

cmake --build "${BUILD_DIR}" --parallel

HOST_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
STRIP_BIN="${ANDROID_NDK}/toolchains/llvm/prebuilt/${HOST_TAG:-${HOST_OS}-x86_64}/bin/llvm-strip"
if [[ -x "${STRIP_BIN}" ]]; then
    "${STRIP_BIN}" --strip-unneeded \
        -o "${DIST_DIR}/liblsfg-vk-${ABI}.so" \
        "${BUILD_DIR}/liblsfg-vk.so"
else
    echo "warning: llvm-strip not found at ${STRIP_BIN}; copying unstripped library" >&2
    cp "${BUILD_DIR}/liblsfg-vk.so" "${DIST_DIR}/liblsfg-vk-${ABI}.so"
fi

cp "${REPO_ROOT}/VkLayer_LS_frame_generation.json" "${BUILD_DIR}/"
cp "${REPO_ROOT}/VkLayer_LS_frame_generation.json" "${DIST_DIR}/"

echo ""
echo "Build complete. Artifacts:"
echo "  ${DIST_DIR}/liblsfg-vk-${ABI}.so"
echo "  ${DIST_DIR}/VkLayer_LS_frame_generation.json"
echo ""
echo "For GameNative Android app updates, copy the arm64-v8a shared library to:"
echo "  app/src/main/assets/lsfg_vk/android_arm64_v8a/liblsfg-vk-layer.so"
