# gpu5080 RAID / fstab notes (dual ~3 TB)

Not a drop-in `/etc/fstab`. Do **not** `install` this over the live
file. Captured 2026-08-31 from `tzervas@192.168.1.251`.

gpu5080 LAN is **`192.168.1.251`**. Never `.252`. Preserve `/models`
on `sda2`. Do not use that mount as VFIO guest root. Do not power off
to “fix” the degraded array from autodev.

## Live block devices

| Disk | Size | Role |
|---|---|---|
| `nvme0n1` | 465.8 G | OS (LVM `gpu5080-vg`: root + swap) |
| `sda` | 2.7 T HGST | RAID member + models |
| `sda1` | 1.8 T `linux_raid_member` `gpu5080:bulk` | only leg in md127 |
| `sda2` | 931.5 G ext4 `LABEL=models-hdd` | bind-mounted at `/models` |
| `sdb` | 2.7 T HGST | **not** in the array (`zfs_member` `rpool`) |
| `sdb1` | 2.7 T | inspect mount `/mnt/sdb1-inspect` |

## md127 (`gpu5080:bulk`)

- ARRAY UUID `2fb1b150:fb96fedc:9f569b6b:ddfc2276` — see [mdadm.conf](mdadm.conf)
- Super 1.2 raid1, **degraded** `[2/1] [_U]`, **read-only**, `sda1` only
- Filesystem: ext4 `LABEL=bulk` UUID `ecd4bae8-fadb-42fb-8405-832cb4229595`
- **Live mount:** `/mnt/bulk-old` (ro). fstab intends `LABEL=bulk` → `/bulk`
  (that path is **not** mounted)
- Autodev must not `mdadm --manage --add`, grow, or format `sdb` into
  the array. Operator repair only.

## fstab excerpts (live)

```
LABEL=bulk /bulk ext4 defaults,noatime,nofail 0 2
LABEL=models-hdd /models-hdd ext4 defaults,noatime,nofail 0 2
/models-hdd /models none bind,nofail,x-systemd.requires-mounts-for=/models-hdd 0 0
```

`lsblk` shows `sda2` mounted directly at `/models` (bind of `models-hdd`).
Do not retarget guest disks at this UUID.

## o11y

RAID is a taxonomy **annotation**, not a Prom `path` on `akula-node`
(`node_filesystem` already uses `path=` for mountpoints). See
`docs/program/CSD-O11Y-TAXONOMY.md`.
