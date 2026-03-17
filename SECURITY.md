# Security Policy

## Supported versions

This project is pre-1.0. Security fixes are applied to the latest release and to
`main`; older tags are not maintained.

| Version | Supported |
|---------|-----------|
| latest `main` / newest release | ✅ |
| anything older | ❌ |

## Reporting a vulnerability

Please report security issues **privately**, not through public issues or pull
requests.

- Email: **timothy.redaelli@gmail.com** (subject: `llama-preset-proxy security`)
- Alternatively, open a private GitHub Security Advisory on the repository.

Include the affected version/commit, reproduction steps, and the impact you
observed. You will get an acknowledgement; coordinated disclosure is preferred
before any public detail is shared.

## Deployment security model

`llama-preset-proxy` is a thin reverse proxy in front of a `llama-server`
backend. Keep these properties in mind when deploying:

- **No client-facing authentication.** The proxy does not authenticate clients.
  Any client that can reach the listen address can reach the backend through it.
  Bind to `127.0.0.1` (the default) or restrict access with a firewall / private
  network, and put an authenticating, TLS-terminating reverse proxy (nginx,
  Caddy, …) in front if it must be exposed.
- **Client→proxy is plain HTTP.** TLS is only supported on the proxy→backend leg
  (`--backend-url https://…`). Terminate client TLS in a front proxy.
- **Auth headers are forwarded verbatim.** Any `Authorization`/API-key header the
  client sends is passed through to the backend unchanged; the proxy neither
  validates nor stores it. Hop-by-hop headers and `Host` are stripped.
- **Resource limits.** Request bodies are capped (`--max-body-mb`, default 8 MiB
  → `413`). In-flight requests are bounded (`--max-concurrent`, default 256 →
  excess shed with `503`). Idle/connect timeouts are bounded
  (`--request-timeout-secs`, default 300 → `504`).
- **Secrets in logs.** At `debug`/`RUST_LOG` verbosity, request metadata may be
  logged. Avoid debug logging in production if request headers carry secrets.

These are documented in more detail in the README "Security notes" section.
