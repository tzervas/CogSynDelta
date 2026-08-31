# gpu5080 RAID / fstab notes (dual ~3 TB stripe)

Not a drop-in `/etc/fstab`. Do **not** `install` this over the live
file. Recaptured 2026-08-31T20:25:07Z from `tzervas@192.168.1.251`.

gpu5080 LAN is **`192.168.1.251`**. Never `.252`. Preserve `/models`
(bind of `/bulk/models-hdd`). Do not use that mount as VFIO guest root.
Do not wipe NVMe/SSD OS. Do not reshape a healthy array that already
uses both 3 TB disks.

## Live block devices

| Disk | Size | Role |
|---|---|---|
| `nvme0n1` | 465.8 G Samsung 970 EVO | OS (LVM `gpu5080-vg`: root + swap). **Do not wipe.** |
| `sda` | 2.7 T HGST HDN724030ALE640 | RAID0 member |
| `sda1` | 2.7 T `linux_raid_member` `gpu5080:bulk` | md127 leg 0 |
| `sdb` | 2.7 T HGST HDN724030ALE640 | RAID0 member |
| `sdb1` | 2.7 T `linux_raid_member` `gpu5080:bulk` | md127 leg 1 |

No separate `sda2` models partition on this capture. `/models` is a bind
of `/bulk/models-hdd` on the stripe.

## md127 (`gpu5080:bulk`)

- Live ARRAY UUID `8d85a4cc:d1690f1e:5b147e0f:767d6d3e` (created
  2026-08-31 16:11:07 EDT)
- Super 1.2 **RAID0** stripe, 512k chunks, **clean 2/2**, 0 failed
- Size 5860265984 blocks = **5.46 TiB**
- Filesystem: ext4 `LABEL=bulk` UUID `d942aafc-aa92-4fe4-8a04-732a5eda8809`
- **Live mount:** `/bulk` `rw,noatime,stripe=256` — 5.5 T, ~94 G used
- Old raid1 UUID `2fb1b150:fb96fedc:9f569b6b:ddfc2276` and mount
  `/mnt/bulk-old` are **dead**. Git `mdadm.conf` still has that UUID as
  a historical copy — do **not** install it over live `/etc/mdadm`.
- Autodev must not `mdadm --manage --add`, grow, format, or re-level
  the array. Operator only.

## fstab excerpts (intent)

```
LABEL=bulk /bulk ext4 defaults,noatime,nofail 0 2
```

`/models` follows `/bulk/models-hdd`. Do not retarget guest disks at
this UUID.

## o11y

RAID is a taxonomy **annotation**, not a Prom `path` on `akula-node`
(`node_filesystem` already uses `path=` for mountpoints). See
`docs/program/CSD-O11Y-TAXONOMY.md`.
