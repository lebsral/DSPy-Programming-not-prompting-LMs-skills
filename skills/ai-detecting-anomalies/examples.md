# Anomaly Detection Examples

## Example 1 - Transaction Fraud Detector

Scores payment events against a user's spending baseline and routes by fraud severity.

```python
import dspy
from typing import Literal

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

SEVERITY_LEVELS = ["normal", "low", "medium", "high", "critical"]

class ScoreTransaction(dspy.Signature):
    """Analyze a payment transaction for fraud signals.
    Consider amount, location, device, time of day, and velocity.
    Compare against the users normal spending baseline."""
    transaction: str = dspy.InputField(
        desc="Transaction details as JSON - amount, merchant, location, device, timestamp"
    )
    user_baseline: str = dspy.InputField(
        desc="Users normal spending patterns - typical amounts, locations, devices, and hours"
    )
    severity: Literal[tuple(SEVERITY_LEVELS)] = dspy.OutputField(
        desc="Fraud severity - normal to critical"
    )
    explanation: str = dspy.OutputField(
        desc="Specific fraud signals citing exact values from the transaction and how they "
             "deviate from the baseline. Example - amount $4,800 is 12x the $400 average, "
             "device is new, and time is 3am outside normal 9am-9pm window."
    )
    anomaly_score: float = dspy.OutputField(
        desc="Fraud confidence from 0.0 (clearly legitimate) to 1.0 (clearly fraudulent)"
    )

scorer = dspy.ChainOfThought(ScoreTransaction)

# User baseline (built from 30-day history)
baseline = """
User spending baseline (30 days):
- Average transaction: $412, typical range $50-$800, max ever: $1,400
- Common merchants: Amazon, Whole Foods, Shell Gas, Netflix
- Locations: New York City, Brooklyn (home), occasional travel to Chicago
- Devices: iPhone 14 (primary), MacBook Pro (web)
- Active hours: 8am-11pm Eastern
- Frequency: 4-7 transactions per day
"""

# Test transactions
test_cases = [
    {
        "amount": 4800,
        "merchant": "Electronics Store",
        "location": "Lagos, Nigeria",
        "device": "Unknown Android",
        "hour_local": 3,
        "day": "Tuesday"
    },
    {
        "amount": 52,
        "merchant": "Whole Foods",
        "location": "Brooklyn, NY",
        "device": "iPhone 14",
        "hour_local": 18,
        "day": "Wednesday"
    },
    {
        "amount": 1100,
        "merchant": "Best Buy",
        "location": "New York City",
        "device": "MacBook Pro",
        "hour_local": 14,
        "day": "Saturday"
    },
]

import json

for tx in test_cases:
    result = scorer(transaction=json.dumps(tx), user_baseline=baseline)
    print(f"\nTransaction: ${tx['amount']} at {tx['merchant']}")
    print(f"Severity: {result.severity} (score: {result.anomaly_score:.2f})")
    print(f"Explanation: {result.explanation}")

    # Route by severity
    if result.severity == "critical":
        print("ACTION - Block transaction, send SMS alert")
    elif result.severity == "high":
        print("ACTION - Alert fraud team, hold for review")
    elif result.severity == "medium":
        print("ACTION - Queue for next analyst review")
    else:
        print("ACTION - Approve")
```

---

## Example 2 - User Behavior Anomaly Detector

Detects account takeover attempts by analyzing login and usage patterns.

```python
import dspy
from typing import Literal
import json

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

SEVERITY_LEVELS = ["normal", "low", "medium", "high", "critical"]

class ScoreUserSession(dspy.Signature):
    """Analyze a user session for account takeover or abuse signals.
    Consider login patterns, device changes, geographic jumps, feature usage,
    and action velocity. Session-level patterns matter as much as individual events."""
    session_events: str = dspy.InputField(
        desc="JSON array of session events in chronological order - logins, actions, API calls"
    )
    user_baseline: str = dspy.InputField(
        desc="This users normal behavior - typical devices, locations, usage patterns, and hours"
    )
    severity: Literal[tuple(SEVERITY_LEVELS)] = dspy.OutputField(
        desc="Anomaly severity - normal to critical"
    )
    explanation: str = dspy.OutputField(
        desc="What specific pattern across this session is anomalous. "
             "Cite the events and values that raised suspicion."
    )
    anomaly_score: float = dspy.OutputField(
        desc="Confidence score from 0.0 to 1.0"
    )
    risk_factors: list[str] = dspy.OutputField(
        desc="List of individual risk factors identified, e.g. ['new device', 'geographic jump', 'high velocity']"
    )

session_scorer = dspy.ChainOfThought(ScoreUserSession)

user_baseline = """
User normal behavior (90 days):
- Devices: MacBook Pro (home), iPhone 13 (mobile)
- Locations: San Francisco, CA (primary); occasional trips to NYC
- Login times: 8am-8pm Pacific, weekdays mostly
- Typical session - check dashboard, run 1-3 reports, update settings occasionally
- API calls per session: 10-50, spread over 20-60 minutes
- Password changes: 0 in 90 days
- 2FA method: authenticator app
"""

# Suspicious session - looks like account takeover
suspicious_session = [
    {"time": "2024-01-15T02:14:00Z", "event": "login_success", "device": "Unknown Windows PC",
     "ip": "185.220.101.42", "country": "Romania", "2fa": "bypass_attempted"},
    {"time": "2024-01-15T02:14:30Z", "event": "password_change", "device": "Unknown Windows PC"},
    {"time": "2024-01-15T02:14:45Z", "event": "email_change", "new_email": "backup9912@tempmail.com"},
    {"time": "2024-01-15T02:15:00Z", "event": "api_key_created", "scope": "full_access"},
    {"time": "2024-01-15T02:15:10Z", "event": "bulk_data_export", "records": 50000},
]

result = session_scorer(
    session_events=json.dumps(suspicious_session, indent=2),
    user_baseline=user_baseline
)

print(f"Severity: {result.severity}")
print(f"Anomaly score: {result.anomaly_score:.2f}")
print(f"Risk factors: {result.risk_factors}")
print(f"\nExplanation:\n{result.explanation}")

# Optimize with BootstrapFewShot given labeled session data
# trainset = [dspy.Example(...).with_inputs("session_events", "user_baseline"), ...]
# optimizer = dspy.BootstrapFewShot(metric=anomaly_metric, max_bootstrapped_demos=4)
# optimized = optimizer.compile(session_scorer, trainset=trainset)
# optimized.save("session_scorer.json")
```

---

## Example 3 - Log Anomaly Detector

Flags unusual error patterns and security events in server logs.

```python
import dspy
from typing import Literal
import json
from collections import defaultdict

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

SEVERITY_LEVELS = ["normal", "low", "medium", "high", "critical"]

class ScoreLogBatch(dspy.Signature):
    """Analyze a batch of server log entries for anomalies.
    Look for unusual error rates, unexpected endpoints, suspicious IPs,
    timing patterns, scanning behavior, and attack signatures."""
    log_batch: str = dspy.InputField(
        desc="JSON array of log entries from a 5-minute window - method, path, status, ip, latency"
    )
    baseline_summary: str = dspy.InputField(
        desc="Normal log patterns - typical error rates, common paths, expected IPs, usual latency"
    )
    severity: Literal[tuple(SEVERITY_LEVELS)] = dspy.OutputField(
        desc="Anomaly severity level"
    )
    explanation: str = dspy.OutputField(
        desc="What in these logs is anomalous. Cite specific paths, IPs, error rates, "
             "or patterns with concrete numbers."
    )
    anomaly_score: float = dspy.OutputField(desc="Confidence from 0.0 to 1.0")
    anomalous_entries: list[str] = dspy.OutputField(
        desc="List of the most suspicious individual log entries or patterns"
    )

log_scorer = dspy.ChainOfThought(ScoreLogBatch)

log_baseline = """
Normal log patterns (7-day baseline):
- Error rate: 0.3% of requests (mostly 404s from typos)
- Common paths: /api/v1/users, /api/v1/products, /health, /static/*
- Typical IPs: known CDN ranges, office IPs (10.0.0.0/8)
- Latency p99: 250ms, average: 45ms
- Request rate: 200-800 req/min during business hours
- No 401/403 spikes, no /admin access from external IPs
"""

# Suspicious log batch - looks like a scanning/injection attempt
suspicious_logs = [
    {"time": "02:31:00", "method": "GET", "path": "/admin/config.php", "status": 404, "ip": "185.220.101.42"},
    {"time": "02:31:01", "method": "GET", "path": "/wp-admin/", "status": 404, "ip": "185.220.101.42"},
    {"time": "02:31:01", "method": "GET", "path": "/.env", "status": 200, "ip": "185.220.101.42"},
    {"time": "02:31:02", "method": "GET", "path": "/api/v1/users?id=1 OR 1=1", "status": 500, "ip": "185.220.101.42"},
    {"time": "02:31:02", "method": "POST", "path": "/api/v1/login", "status": 401, "ip": "185.220.101.42"},
    {"time": "02:31:03", "method": "POST", "path": "/api/v1/login", "status": 401, "ip": "185.220.101.42"},
    {"time": "02:31:03", "method": "POST", "path": "/api/v1/login", "status": 401, "ip": "185.220.101.42"},
    {"time": "02:31:04", "method": "GET", "path": "/api/v1/products", "status": 200, "ip": "10.0.1.5"},
]

result = log_scorer(
    log_batch=json.dumps(suspicious_logs, indent=2),
    baseline_summary=log_baseline
)

print(f"Severity: {result.severity} (score: {result.anomaly_score:.2f})")
print(f"\nAnomalous entries:")
for entry in result.anomalous_entries:
    print(f"  - {entry}")
print(f"\nExplanation:\n{result.explanation}")

# Batch processing pipeline for streaming logs
def process_log_window(log_entries: list[dict], baseline: str, window_size: int = 50):
    """Score a rolling window of log entries."""
    # Group into batches of window_size
    for i in range(0, len(log_entries), window_size):
        batch = log_entries[i:i + window_size]
        result = log_scorer(
            log_batch=json.dumps(batch, indent=2),
            baseline_summary=baseline
        )
        if result.severity not in ("normal", "low"):
            yield {
                "window_start": batch[0]["time"],
                "window_end": batch[-1]["time"],
                "severity": result.severity,
                "score": result.anomaly_score,
                "explanation": result.explanation,
                "anomalous_entries": result.anomalous_entries,
            }
```
