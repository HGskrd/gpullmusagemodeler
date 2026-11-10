# Admin sign-in

The calculator stays public. Production admin access uses the same Tabs sign-in
as Contacts, Upcoming and Scout. Opening `/admin` redirects through
`https://tabs.gskrd.me/sso/gpumodels`; an existing Tabs session completes sign-in
without asking for another password.

Set these values in the GPU Models service's private environment file:

```text
PLANNER_ADMIN_SSO_ENABLED=true
PLANNER_ADMIN_SSO_USERS=reader
SSO_SHARED_SECRET=<same 32+ character secret used by Tabs>
PLANNER_SECRET_KEY=<independent random Flask session signing key>
PLANNER_SECURE_COOKIES=true
PLANNER_BEHIND_PROXY=true
WEB_CONCURRENCY=1
```

`PLANNER_ADMIN_SSO_USERS` is an explicit, comma-separated account allowlist.
Signing into another application does not automatically grant admin permission.
Tabs must expose `/sso/gpumodels` and have `GPUMODELS_URL` set to this application's
HTTPS origin (default `https://gpumodels.gskrd.me`). The optional
`PLANNER_ADMIN_SSO_URL` defaults to the Tabs URL above and requires HTTPS.

Tabs signs a handoff with purpose `gpumodels-sso` and a 60-second expiry. GPU Models
requires a matching browser-bound sign-in state, validates the signature, expiry
and account allowlist, then creates its own host-only Flask session. That admin
session expires after eight hours. Handoffs for Contacts and the other apps are
not accepted. Auth responses disable caching and referrer forwarding.

Password login is disabled whenever SSO is enabled, even if
`PLANNER_ADMIN_PASSWORD` remains configured. Remove that variable from production
after enabling SSO. Password-only local deployments remain supported when SSO is
disabled; they must set both `PLANNER_ADMIN_PASSWORD` and `PLANNER_SECRET_KEY`.

Admin logout clears the local session and visits Tabs logout. Other applications
keep their independent sessions until those sessions expire or are logged out.
Changing the SSO secret, allowed accounts or Flask signing key invalidates
existing GPU Models admin sessions.

Verify the deployed origin and public hostname after restarting:

- `/healthz` and `/` stay available without authentication.
- `/admin/data.json` returns 401 without an admin session.
- `/admin` leads to Tabs login or returns with the existing shared sign-in.
- Only an allowlisted account can view the admin pages and JSON data.
- Posting the old password to `/admin/login` returns 405.

Keep secrets in the service environment file, never in the repository or a URL.
