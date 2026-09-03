# csd-lab-console: auth gate and deployment note

As of fix/lab-console-auth-all-routes, every `/api/*` route (GET and POST,
local and proxied) is gated by `check_api_auth()` as the first statement of
`dispatch_api_get()`/`dispatch_api_post()` -- see the docstrings on those
functions and on `check_api_auth()`/`check_apply_auth()` in
`scripts/csd-lab-console` for the mechanism. The static UI page at `/` and
`/lab` is unauthenticated by design; every `/api/*` call the page's JS makes
carries a bearer token the browser prompts for once and then remembers
(`localStorage`).

## Deploying this change

The auth gate needs `CSD_APPLY_TOKEN` (or `CSD_APPLY_TOKEN_FILE`) set in the
unit's environment, or every route on that instance now returns 503
("lab console misconfigured") instead of running -- this is a change in
behavior from before this fix, when an unset token only affected
`POST /api/apply` and every other route ran regardless.

- **akula-prime** (`csd-lab-console.service`): already fine. Its `ExecStart`
  already runs the script under `secret exec ... CSD_APPLY_TOKEN=csd/apply-token
  -- ...`, so the running instance has had a valid token the whole time this
  branch was developed. (Note, unrelated to this fix: the *checked-in*
  `deploy/systemd/csd-lab-console.service` in this repo does not yet have that
  `CSD_APPLY_TOKEN=csd/apply-token` clause -- the live unit on prime was
  updated out of band and is ahead of the repo copy. Worth reconciling
  separately; not part of this change.)

- **homelab** (`csd-lab-console.homelab.service`, currently disabled/inactive):
  its `ExecStart` sets `CSD_LAB_BIND` and `CSD_UPSTREAM` only -- no
  `CSD_APPLY_TOKEN` and no `CSD_APPLY_TOKEN_FILE`. Once this fix ships, that
  unit will return 503 on every `/api/*` call (including the ones it proxies
  to prime) until it gets a token the same way the prime unit does, e.g.:

  ```
  ExecStart=/home/kang/.local/bin/secret exec CSD_APPLY_TOKEN=csd/apply-token -- /usr/bin/python3 /home/kang/code/personal/tzervas/CogSynDelta/scripts/csd-lab-console --http 9118
  ```

  (Same vault entry the prime unit already uses -- `csd/apply-token` -- since
  both instances are meant to accept the same operator bearer token.)

**Operator-visible effect once a token is set on a given instance:** every
browser tab open against that instance's `/lab` page will prompt once for
the bearer token on its next `/api/*` call (the page has no way to know the
token in advance), then remember it. No code change is needed for that --
it is the existing token-prompt flow in the page's `apiFetch()` wrapper,
just now reachable from every route instead of only `/api/apply`.

**This fix does not restart or otherwise touch either systemd unit.** The
operator restarts `csd-lab-console.homelab.service` (after adding the token)
whenever they choose to bring it back up.
