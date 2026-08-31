# Homelab send-only MTA

Postfix on homelab for Grafana → `maintainers@vectorweight.com`.

This host is **not** an MX. It cannot be used as a spam relay.

## Attack surface (closed)

- Listens **only** `127.0.0.1:25` and `172.30.0.1:25` (security-net gateway).
  Not `0.0.0.0`, not `192.168.1.170`, not IPv6.
- No submission (`587`) or SMTPS (`465`). Those stay commented in `master.cf`.
- Relays **only** from `127.0.0.1` and Grafana `172.30.0.13`. Other security-net
  containers, LAN, and WAN are rejected at smtpd.
- Envelope sender must be `grafana@vectorweight.com`.
- Envelope recipient must be `maintainers@` or `alerts@vectorweight.com`.
- No SASL, no inbound TLS, VRFY off, unauth pipelining rejected, no local
  mailboxes (`local_transport = error`).
- Rate-limited (20 messages / minute / client). Message size 200 KiB.
- **Do not** port-forward 25/587/465 on the WAN router.

## DNS (do not change)

- MX `vectorweight.com` = `smtp.google.com` (Google Workspace).
- SPF `v=spf1 include:_spf.google.com ~all`.
- Homelab is **not** listed. Direct send from the homelab IP may land in
  junk until Workspace SMTP relay / SPF includes this public IP.

## Grafana

Quadlet drop-in `unifi-grafana.container.d/smtp.conf`:

- `GF_SMTP_HOST=172.30.0.1:25`
- `GF_SMTP_FROM_ADDRESS=grafana@vectorweight.com`
- no SMTP auth (mynetworks + allowlists)

Contact point `maintainers-email` → `maintainers@vectorweight.com`.
Default notification policy uses that receiver.

## Install on homelab

```bash
sudo install -m 644 deploy/mail/postfix-main.cf /etc/postfix/main.cf
sudo install -m 644 deploy/mail/sender_allowlist /etc/postfix/sender_allowlist
sudo install -m 644 deploy/mail/recipient_allowlist /etc/postfix/recipient_allowlist
sudo postmap hash:/etc/postfix/sender_allowlist
sudo postmap hash:/etc/postfix/recipient_allowlist
sudo postfix check && sudo systemctl reload postfix
```

Keep `exim4` masked. Do not enable `inet_interfaces = all`.
