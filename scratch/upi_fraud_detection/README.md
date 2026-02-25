# UPI Fraud Detection — Design Notes

## Why LSTM (Not ANN)

Fraud detection requires temporal context. A single transaction sending Rs. 50,000 to a new payee at midnight looks suspicious in isolation — but the LSTM sees that the previous 14 steps showed a device change followed by 3 rapid-fire small transfers, making the large final transfer unmistakably part of an Account Takeover (ATO) attack. An ANN would evaluate each transaction independently and miss this contextual escalation arc.

## Why Synthetic Data (Not Raw NPCI)

The provided NPCI dataset contains only monthly aggregate statistics (total volume in millions, total value in crores) — not individual transactions, and no fraud labels. Synthetic generation was the only viable path. NPCI calibration parameters (avg transaction ≈ Rs. 1,612, derived from Value/Volume ratio across 30 monthly rows) ensure synthetic amounts are grounded in real market data.

## Sequence Design: The 15-Step ATO Pattern

The 15-step window was designed to cover the complete Account Takeover arc:
- **Steps 1–8:** Normal baseline (low amounts, known payees, business hours, velocity 1–3)
- **Step 9:** Device change — the SIM swap / phone theft signal
- **Steps 10–12:** Velocity spike (5–8 transactions/hour, gaps < 15 min) — attacker testing the account
- **Steps 13–14:** Moderate escalation (3–5× user avg) to new payees at unusual hours
- **Step 15:** Final large transfer (4–9× user avg) — the primary fraud event (label = 1)

This arc is invisible to single-transaction rule engines but clearly learnable by LSTM hidden states that accumulate evidence across all 15 steps.

## Label Noise (7%)

7% of labels are randomly flipped after generation. This prevents the LSTM from memorizing perfectly deterministic synthetic patterns and forces it to learn robust statistical associations — mimicking the real world where some fraud goes unreported and some reports are false alarms.

## pos_weight Cap (5.0)

The raw inverse-class-frequency weight would be ~3.35 given the 23% fraud rate in training data. The cap at 5.0 is a safety measure retained from an earlier design iteration (when fraud rate was 2.3%) to prevent over-aggressive fraud prediction. With the current 23% fraud rate the cap is not binding.

## Feature: cumulative_amount_ratio

This feature (current amount ÷ user's historical average amount) is the single most discriminating signal. In fraud sessions, the final step reaches values of 4–9; in normal sessions it stays near 1.0. This is realistic — transaction monitoring systems routinely flag amount ratios > 5× as high risk.
