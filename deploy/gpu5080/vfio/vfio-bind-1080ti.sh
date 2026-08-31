#!/bin/bash
# Bind GTX 1080 Ti 06:00.0 (+ HDMI audio 06:00.1) to vfio-pci.
# Never bind RTX 5080 01:00.0 / 01:00.1 (host CUDA CI).
set -euo pipefail

GPU=0000:06:00.0
AUD=0000:06:00.1
GPU_WANT=0x1b06
AUD_WANT=0x10ef

die() { echo "vfio-bind-1080ti: $*" >&2; exit 1; }

check_id() {
  local bdf="$1" want="$2"
  local vend dev
  [[ -e "/sys/bus/pci/devices/${bdf}" ]] || die "missing ${bdf}"
  vend=$(cat "/sys/bus/pci/devices/${bdf}/vendor")
  dev=$(cat "/sys/bus/pci/devices/${bdf}/device")
  [[ "$vend" == "0x10de" && "$dev" == "$want" ]] || die "${bdf} is ${vend}:${dev}, want 0x10de:${want}"
  [[ "$dev" != "0x2c02" && "$dev" != "0x22e9" ]] || die "${bdf} is a 5080 function — refuse"
}

# IOMMU group 18 must be 1080 Ti only
for d in /sys/kernel/iommu_groups/18/devices/*; do
  bdf=$(basename "$d")
  dev=$(cat "$d/device")
  [[ "$dev" != "0x2c02" && "$dev" != "0x22e9" ]] || die "5080 ${bdf} in IOMMU group 18"
done

check_id "$GPU" "$GPU_WANT"
check_id "$AUD" "$AUD_WANT"

modprobe vfio
modprobe vfio_iommu_type1
modprobe vfio-pci

unbind_if() {
  local bdf="$1"
  local link="/sys/bus/pci/devices/${bdf}/driver"
  if [[ -e "$link" ]]; then
    local drv
    drv=$(basename "$(readlink "$link")")
    case "$drv" in
      vfio-pci) return 0 ;;
      nvidia|snd_hda_intel|nouveau|nvidia_drm)
        echo "$bdf" > "/sys/bus/pci/devices/${bdf}/driver/unbind"
        ;;
      *) die "unexpected driver ${drv} on ${bdf}" ;;
    esac
  fi
}

unbind_if "$GPU"
unbind_if "$AUD"

echo vfio-pci > "/sys/bus/pci/devices/${GPU}/driver_override"
echo vfio-pci > "/sys/bus/pci/devices/${AUD}/driver_override"

bind_one() {
  local bdf="$1"
  if [[ -e "/sys/bus/pci/devices/${bdf}/driver" ]]; then
    local drv
    drv=$(basename "$(readlink "/sys/bus/pci/devices/${bdf}/driver")")
    [[ "$drv" == "vfio-pci" ]] && return 0
    die "${bdf} still ${drv}"
  fi
  echo "$bdf" > /sys/bus/pci/drivers/vfio-pci/bind
}

bind_one "$GPU"
bind_one "$AUD"
echo "vfio-bind-1080ti: ${GPU}+${AUD} vfio-pci (IOMMU group 18); 5080 untouched"
