# Metric A: Sabotage Detail Report

**Focus TF:** AraC

## 1. Injected False Positives (Goal: Detect as ProbableFalsePos)

| Edge ID | Previous Value | System Prediction | Z-Score | Outcome |
|---|---|---|---|---|
| arac→cbrc | Non-Existent | ProbableFalsePos | 0.00 | ✅ DETECTED |
| arac→argr | Non-Existent | ProbableFalsePos | 0.00 | ✅ DETECTED |
| arac→yefmb | Non-Existent | Not Reviewed | N/A | ⚠️ SKIPPED |
| arac→cyoc | Non-Existent | ProbableFalsePos | 0.54 | ✅ DETECTED |
| arac→ybex | Non-Existent | ProbableFalsePos | 0.00 | ✅ DETECTED |

## 2. Deleted True Edges (Goal: Recover as NovelHypothesis)

| Edge ID | Previous Value | System Prediction | Z-Score | Outcome |
|---|---|---|---|---|
| arac→ygea | Existing (True) | ConditionSilent | 2.59 | ⚠️ ConditionSilent |
| arac→arae | Existing (True) | NovelHypothesis | 4.80 | ✅ RECOVERED |
| arac→yjff | Existing (True) | Not Found (Low Signal) | N/A | ❌ MISSED (Low Stats) |
| arac→ytft | Existing (True) | Not Found (Low Signal) | N/A | ❌ MISSED (Low Stats) |
| arac→arah | Existing (True) | Not Found (Low Signal) | N/A | ❌ MISSED (Low Stats) |