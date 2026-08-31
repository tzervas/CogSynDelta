#!/bin/bash
# Create the gpu5080-1080ti-rag Debian guest (VFIO 06:00.0/1 only).
# Disk on NVMe /var/lib/libvirt/images. Never wipe /models. Never VFIO 01:00.0.
set -euo pipefail

DOMAIN=gpu5080-1080ti-rag
UUID=8c1a0b10-1080-4f00-9a18-000000000018
IMAGES=/var/lib/libvirt/images
WORK=/home/tzervas/akula-harness/config/qemu/1080ti-rag
IMG="$IMAGES/${DOMAIN}.qcow2"
SEED="$IMAGES/${DOMAIN}-cidata.iso"
CLOUD_NAME=debian-13-genericcloud-amd64.qcow2
CLOUD="$IMAGES/$CLOUD_NAME"
CLOUD_URL=https://cloud.debian.org/images/cloud/trixie/latest/${CLOUD_NAME}
SUMS_URL=https://cloud.debian.org/images/cloud/trixie/latest/SHA512SUMS
LAN_MAC=52:54:00:10:80:71
NAT_MAC=52:54:00:10:80:72
GPU_NODE=pci_0000_06_00_0
AUD_NODE=pci_0000_06_00_1
FORBID_GPU=pci_0000_01_00_0

die() { echo "provision-1080ti-rag: $*" >&2; exit 1; }

[[ "$(hostname)" == "gpu5080" ]] || die "run on gpu5080 only"
[[ "$(id -u)" -eq 0 ]] || die "run as root (sudo)"

findmnt /models >/dev/null || die "/models not mounted — abort rather than guess"
findmnt -T "$IMAGES" | grep -q gpu5080--vg-root || die "$IMAGES is not on NVMe root LV"
[[ -e /sys/bus/pci/devices/0000:06:00.0 ]] || die "missing 06:00.0"
[[ "$(cat /sys/bus/pci/devices/0000:06:00.0/device)" == "0x1b06" ]] || die "06:00.0 is not GP102"
[[ "$(cat /sys/bus/pci/devices/0000:01:00.0/device)" == "0x2c02" ]] || die "01:00.0 is not the 5080 — unexpected"
drv5080=$(basename "$(readlink -f /sys/bus/pci/devices/0000:01:00.0/driver)")
[[ "$drv5080" == "nvidia" ]] || die "5080 driver is $drv5080, want nvidia — refuse VFIO"
drv1080=$(basename "$(readlink -f /sys/bus/pci/devices/0000:06:00.0/driver)")
[[ "$drv1080" == "vfio-pci" ]] || die "1080 Ti driver is $drv1080, want vfio-pci"

systemctl is-enabled akula-comfyui.service 2>/dev/null | grep -qx masked \
  || echo "WARN: Comfy not masked (will not unmask)" >&2
systemctl is-active akula-comfyui.service >/dev/null 2>&1 \
  && die "Comfy is active — refuse (do not unmask, do not fight it)"

mkdir -p "$WORK" "$IMAGES"

if [[ ! -s "$CLOUD" ]]; then
  echo "downloading $CLOUD_URL"
  curl -fL --retry 3 -o "${CLOUD}.part" "$CLOUD_URL"
  curl -fL --retry 3 -o "${IMAGES}/SHA512SUMS.trixie-cloud" "$SUMS_URL"
  mv "${CLOUD}.part" "$CLOUD"
  (cd "$IMAGES" && sha512sum -c SHA512SUMS.trixie-cloud --ignore-missing) || {
    rm -f "$CLOUD"
    die "cloud image checksum failed"
  }
fi

if [[ ! -s "$IMG" ]]; then
  qemu-img convert -p -O qcow2 "$CLOUD" "$IMG"
  qemu-img resize "$IMG" 40G
fi
chown libvirt-qemu:libvirt-qemu "$IMG"
chmod 0600 "$IMG"

SRC_STEALTH="$WORK/stealth-leds"
[[ -x "$SRC_STEALTH" ]] || SRC_STEALTH=/usr/local/sbin/cabal-stealth-leds
[[ -x "$SRC_STEALTH" ]] || die "stealth-leds missing"
[[ -x /usr/local/lib/cabal/OpenRGB.AppImage ]] || die "OpenRGB.AppImage missing on host"

SEEDDIR=$(mktemp -d /tmp/1080ti-cidata.XXXXXX)
cleanup() { rm -rf "$SEEDDIR"; }
trap cleanup EXIT
cp -a "$WORK/user-data" "$WORK/meta-data" "$WORK/network-config" "$SEEDDIR/"
cp -a "$SRC_STEALTH" "$SEEDDIR/stealth-leds"
chmod 0755 "$SEEDDIR/stealth-leds"
cp -a /usr/local/lib/cabal/OpenRGB.AppImage "$SEEDDIR/OpenRGB.AppImage"
if [[ -f /usr/local/sbin/nvidia-factory-limits.sh ]]; then
  cp -a /usr/local/sbin/nvidia-factory-limits.sh "$SEEDDIR/nvidia-factory-limits.sh"
elif [[ -f "$WORK/nvidia-factory-limits.sh" ]]; then
  cp -a "$WORK/nvidia-factory-limits.sh" "$SEEDDIR/nvidia-factory-limits.sh"
fi
xorriso -as mkisofs -quiet -o "$SEED" -V cidata -r -J "$SEEDDIR"
chown libvirt-qemu:libvirt-qemu "$SEED"

virsh net-info default >/dev/null
virsh net-info lan-enp5s0 >/dev/null
virsh net-start default >/dev/null 2>&1 || true
virsh net-start lan-enp5s0 >/dev/null 2>&1 || true
virsh net-autostart default
virsh net-autostart lan-enp5s0

echo "nodedev $FORBID_GPU is the 5080 — will not pass it"
virsh nodedev-list --cap pci | grep -qx "$GPU_NODE" || die "missing $GPU_NODE"
virsh nodedev-list --cap pci | grep -qx "$AUD_NODE" || die "missing $AUD_NODE"

if virsh dominfo "$DOMAIN" >/dev/null 2>&1; then
  state=$(virsh domstate "$DOMAIN")
  if [[ "$state" != "shut off" && "$state" != "crashed" ]]; then
    die "domain $DOMAIN already $state"
  fi
  virsh undefine "$DOMAIN" --nvram || virsh undefine "$DOMAIN"
fi

virt-install \
  --connect qemu:///system \
  --name "$DOMAIN" \
  --uuid "$UUID" \
  --memory 8192 \
  --vcpus 4 \
  --cpu host-passthrough \
  --machine q35 \
  --boot uefi \
  --osinfo debian13 \
  --import \
  --disk "path=$IMG,format=qcow2,bus=virtio,cache=none,discard=unmap" \
  --disk "path=$SEED,device=cdrom,bus=sata" \
  --network "network=lan-enp5s0,model=virtio,mac=$LAN_MAC" \
  --network "network=default,model=virtio,mac=$NAT_MAC" \
  --hostdev "${GPU_NODE}" \
  --hostdev "${AUD_NODE}" \
  --graphics vnc,listen=127.0.0.1 \
  --video vga \
  --console pty,target_type=serial \
  --serial pty \
  --channel unix,target_type=virtio,name=org.qemu.guest_agent.0 \
  --rng /dev/urandom \
  --features kvm_hidden=on \
  --noautoconsole \
  --autostart \
  --wait 0

python3 - "$DOMAIN" <<'PY'
import subprocess, sys, xml.etree.ElementTree as ET
dom = sys.argv[1]
xml = subprocess.check_output(["virsh", "dumpxml", dom], text=True)
root = ET.fromstring(xml)
seen = set()
for hd in root.findall("./devices/hostdev"):
    src = hd.find("./source/address")
    if src is None:
        continue
    bus, slot, fn = src.get("bus"), src.get("slot"), src.get("function")
    key = (bus, slot, fn)
    seen.add(key)
    if bus == "0x01" and slot == "0x00":
        raise SystemExit(f"REFUSE: hostdev source {bus}:{slot}.{fn} is the 5080")
    if not (bus == "0x06" and slot == "0x00" and fn in ("0x0", "0x1")):
        raise SystemExit(f"REFUSE: unexpected hostdev source {bus}:{slot}.{fn}")
want = {("0x06", "0x00", "0x0"), ("0x06", "0x00", "0x1")}
if seen != want:
    raise SystemExit(f"REFUSE: hostdev sources {seen}, want {want}")
print("hostdev sources OK: 06:00.0/1 only")
PY

virsh dumpxml "$DOMAIN" > /home/tzervas/akula-harness/config/qemu/gpu5080-1080ti-rag.xml
chown tzervas:tzervas /home/tzervas/akula-harness/config/qemu/gpu5080-1080ti-rag.xml
virsh autostart "$DOMAIN"
echo "defined and started $DOMAIN; autostart on"
virsh list --all
echo "5080 still host nvidia:"
lspci -nnk -s 01:00.0 | sed -n '1,8p'
echo "/models still mounted:"
findmnt /models
