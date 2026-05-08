# Auth Protocol

The Auth Protocol governs how clients authenticate with the hailstone service
and how the service validates presented credentials.

## Token Lifecycle

All authentication tokens follow a three-phase lifecycle:

| Phase   | Description                            | Duration    |
|---------|----------------------------------------|-------------|
| Issue   | Token minted on successful login       | Instant     |
| Active  | Token accepted for authenticated calls | 15 minutes  |
| Expired | Token rejected; client must re-login   | Permanent   |

Token rotation is triggered automatically when less than 2 minutes of
validity remain and the client presents the expiring token.

## Error Codes

Authentication failures return structured error responses:

| Code | Name                | Description                                    |
|------|---------------------|------------------------------------------------|
| 401  | Unauthorized        | No token or token format invalid               |
| 403  | Forbidden           | Token valid but principal lacks permission     |
| 419  | AuthenticationTimeout | Token expired; re-authenticate             |
| 429  | TooManyRequests     | Rate limit exceeded; back off and retry        |

Clients must treat ``401`` and ``419`` as distinct: ``401`` means no valid
session exists; ``419`` means a session existed but has since expired.
