# Design decisions

Short notes on the main choices, the alternatives considered, and what would
change in production.

## 1. Hash chain plus signed anchors, not a blockchain

A hash chain makes any single edit visible, but it needs no secret: an insider
with database access can edit a record and recompute every later hash. The
first version of this project relied on the chain alone and was vulnerable to
exactly that. Signed anchors fix it: every N decisions the service signs the
chain heads and stores the signature outside the database, so a rewrite of
anything before the latest anchor no longer matches.

A blockchain would add consensus between parties who do not trust each other.
Inside one organisation the same tamper evidence comes from a signing key held
in a KMS/HSM and anchors on write-once storage owned by a separate team, which
is far simpler to run.

## 2. Ed25519 rather than HMAC for anchors

HMAC uses one shared secret to both sign and verify, so anyone who can verify
can also forge. With Ed25519, auditors hold only the public key.

## 3. Outcomes as events, not updates

Repayment outcomes arrive months after a decision and are sometimes corrected.
Writing them back into the decision row would either break the hash or require
leaving the field unhashed (the first version did the latter, so outcomes and
group labels could be edited undetected). Each outcome is instead appended to
its own hash chain; the latest event for a decision is the one used for
monitoring, and earlier events remain as history.

## 4. SQLite with BEGIN IMMEDIATE

SQLite keeps the demo dependency-free. Reading the chain head and inserting
inside one `BEGIN IMMEDIATE` transaction means two writers (the API and the
queue worker run as separate processes) cannot both chain onto the same parent.
A multi-process test covers this. For several application servers, PostgreSQL
with a row lock on the chain head would be the next step. A DuckDB backend
existed in an earlier version but did not work (its table definitions were
SQLite-only) and was removed rather than kept half-built.

## 5. Queue between the model and the ledger

The model service returns its decision immediately and publishes an event to
Redis; the worker writes the ledger. An outage in the audit system therefore
does not block lending decisions, but the gap is visible: `/health` on the model
service reports the queue connection, and failed events go to a dead-letter
list. In production a durable log such as Kafka would replace Redis.

## 6. Sex excluded from the model, kept for monitoring

`personal_status` in German Credit combines sex and marital status. A UK lender
may not base credit decisions on sex, so the field is excluded from the model's
inputs. It is still recorded with each decision, because monitoring outcomes by
group requires knowing the group.

## 7. Fairness metrics without a single verdict

Approval-rate parity and equal opportunity answer different questions and can
disagree. Both are reported per group with sample sizes; groups under 30 cases
are excluded from the gap; thresholds are configurable and labelled
illustrative. The tool's job is to surface the evidence for the model owner and
risk committee, not to decide what is fair.

## 8. Metrics without side effects

GET endpoints never write. Snapshots are stored only by the daily scheduled job
or an explicit `POST /monitoring/snapshot`, so the history reflects the
schedule rather than how often someone opened a dashboard.
