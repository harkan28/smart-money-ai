# Systems Section Scan Verification Report: PII & Vendor Accuracy Audit

**Target Repository:** `harkan28/smart-money-ai`  
**Tool Under Evaluation:** ComplyDP Static Code Scanner (Systems & Data Mapping Module)  
**Audit Scope:** Verification of 15 Scanner-Inferred Personal Identifiable Information (PII) elements, detection of False Negatives (missed PII), and evaluation of third-party vendor discovery.  
**Auditor Methodology:** Direct line-by-line inspection of SQLite database schemas, dataclass models, regex dictionaries, dataset generators, and frontend assets.  
**Report Type:** Quality Assurance (QA) & Data Mapping Accuracy Audit  

---

## Executive Summary of Audit Findings

* **Total Scanner Inferences:** 15
* **True Positives (Verified in Codebase):** 4 (26.7%)
* **False Positives (Scanner Hallucinations / Misidentifications):** 11 (73.3%)
  - *Substring Matching Errors:* 3 (`pran` in "Pranay", `uan` in "quantile/quantity", `abha` in "Sabharwal")
  - *Mock Merchant / Category Keyword Confusion:* 4 (`vehicle_registration`, `esic`, `passport`, `pincode`)
  - *Hardcoded Instructional Text Matching:* 2 (`aadhaar`, `pan` in static setup guide)
  - *Zero Occurrence / Pure Hallucination:* 2 (`ifsc`, `ckyc`)
* **False Negatives (Active PII Missed by Scanner):** 7 active user data fields omitted
* **Third-Party Vendors / SDKs Missed by Scanner:** 6 external services omitted

---

## 1. Methodical Evaluation of the 15 Scanned PII Identifiers

The following table presents an exact audit of each identifier inferred by the ComplyDP scanner in the **Systems** section against the active repository:

| # | Scanner Identifier | Status | Exact Code Reference | Code Snippet / Context | Technical Justification & Recommendation |
|:--|:-------------------|:------:|:---------------------|:-----------------------|:-----------------------------------------|
| **1** | `user.financial.ifsc` | **False Positive** | *None* | `No matching string, model field, or regex detected in codebase` | **Technical Justification:** No IFSC string, column, or regex pattern exists in any source file. The scanner inferred this by generic semantic association with Indian banking features.<br>**ComplyDP Action:** **Reject** |
| **2** | `user.identity.aadhaar` | **False Positive** | `smart_money_ai/intelligence/investment_engine/investment_recommendations_backend.py:735, 746`<br>`budgeting_ml_model/src/pro_plus_dataset_generator.py:215` | `Line 735: "2. Complete KYC with Aadhaar and PAN"`<br>`Line 215: 'Passport Seva', 'Aadhaar Center', 'Jan Aushadhi'` | **Technical Justification:** Static string literal occurrence only. The application outputs instructional advice to complete KYC on external apps (Groww/Zerodha) and includes "Aadhaar Center" as a mock merchant. No 12-digit Aadhaar number is ingested, validated, or stored.<br>**ComplyDP Action:** **Reject** |
| **3** | `user.financial.upi` | **True Positive** | `smart_money_ai/utils/data_manager.py:75, 192`<br>`smart_money_ai/config/bank_patterns.json:193`<br>`smart_money_ai/intelligence/expense_categorizer.py:343` | `Line 75: upi_id TEXT,`<br>`Line 193: "PhonePe: You paid Rs\.?(\d+\.?\d*) to (.+?) UPI ID: (.+?) on ([\d\-]+)"` | **Technical Justification:** Active SQLite database column (`upi_id TEXT` in `transactions` table), parsed directly from SMS transaction strings via regex patterns, and evaluated during expense categorization.<br>**ComplyDP Action:** **Confirm** |
| **4** | `user.financial.bank_account` | **True Positive** | `smart_money_ai/utils/data_manager.py:71`<br>`smart_money_ai/core/sms_parser/enhanced_sms_parser.py:29, 330`<br>`SMS PARSING SYSTEM/sms_parser/core_parser.py:25` | `Line 71: account_number TEXT,`<br>`Line 29: account_number: str`<br>`Line 330: account_number = groups[1]` | **Technical Justification:** Active column in SQLite database, defined as a core dataclass attribute, and extracted from transactional bank SMS alert bodies via regex capture groups.<br>**ComplyDP Action:** **Confirm** |
| **5** | `user.contact.email` | **True Positive** | `smart_money_ai/utils/data_manager.py:49, 127`<br>`smart_money_ai/core/smart_money_ai.py:80, 184` | `Line 49: email TEXT UNIQUE,`<br>`Line 80: email: str`<br>`Line 127: user_data['email'],` | **Technical Justification:** Core user attribute defined as `email TEXT UNIQUE` in the SQLite `users` table, stored on user registration and profile creation.<br>**ComplyDP Action:** **Confirm** |
| **6** | `user.contact.phone` | **True Positive** | `smart_money_ai/utils/data_manager.py:50, 128`<br>`smart_money_ai/core/smart_money_ai.py:81` | `Line 50: phone TEXT,`<br>`Line 81: phone: str`<br>`Line 128: user_data.get('phone', ''),` | **Technical Justification:** Core user profile attribute defined as `phone TEXT` in the SQLite `users` schema, stored to link users with their inbound SMS feeds.<br>**ComplyDP Action:** **Confirm** |
| **7** | `user.contact.pincode` | **False Positive** | `SMS PARSING SYSTEM/sms_parser/core_parser.py:270` | `Line 270: r'\s+\d{6}$',  # pincode`<br>`Line 274: cleaned = re.sub(pattern, '', cleaned)` | **Technical Justification:** The scanner identified a 6-digit regex labeled `# pincode`. However, code analysis reveals this pattern is executed strictly for **data sanitation** (stripping postal codes from merchant store names via `re.sub()`). No residential user pincode is stored.<br>**ComplyDP Action:** **Reject** |
| **8** | `user.financial.pran` | **False Positive** | `SMS PARSING SYSTEM/transactions.csv:183, 213, 357, 518, 587` | `Line 183: ...Pranay Gopal,2575877412@okaxis...`<br>`Line 213: ...Pranay Bala,9058370545@okhdfcbank...` | **Technical Justification:** Substring stemming false-positive. The scanner detected the characters `pran` inside the Indian human given name **"Pranay"** across synthetic CSV transaction records. No National Pension System (NPS) PRAN records exist.<br>**ComplyDP Action:** **Reject** |
| **9** | `user.employment.uan` | **False Positive** | `smart_money_ai/ml_models/savings_budgeting_model.py:472`<br>`smart_money_ai/api/breeze_trading_engine.py:244`<br>`examples/notebooks/advanced_investment_sentiment_model.ipynb:540` | `Line 472: similar_users['savings_rate'].quantile(0.75)`<br>`Line 244: data["quantity"] = quantity`<br>`Line 540: load_in_4bit=True,  # 4-bit quantization` | **Technical Justification:** Substring matching error. The scanner matched the 3-letter sequence `uan` inside common mathematical/programming words: **quantile**, **quantity**, and **quantization**. No EPFO UAN employment records exist.<br>**ComplyDP Action:** **Reject** |
| **10** | `user.financial.ckyc` | **False Positive** | *None* | `No matching string, model field, or regex detected in codebase` | **Technical Justification:** Inferred by generic semantic proximity to the acronym "KYC" in instructional documentation. No 14-digit Central KYC registry identifiers exist in models or payloads.<br>**ComplyDP Action:** **Reject** |
| **11** | `user.health.abha` | **False Positive** | `SMS PARSING SYSTEM/transactions.csv:27, 554, 622, 952` | `Line 27: ...Anika Sabharwal,5259795433@okicici...`<br>`Line 554: ...Raunak Sabharwal,9342292431@okybl...` | **Technical Justification:** Substring matching error. The scanner matched the sequence `abha` inside the Indian surname **"Sabharwal"** (`S-abha-rwal`) in synthetic CSV data. The application has zero medical or Ayushman Bharat health account integrations.<br>**ComplyDP Action:** **Reject** |
| **12** | `user.identity.vehicle_registration` | **False Positive** | `budgeting_ml_model/src/mega_dataset_generator.py:210`<br>`budgeting_ml_model/src/large_dataset_generator.py:61` | `Line 210: 'insurance renewal', 'vehicle registration', 'pollution certificate'` | **Technical Justification:** Inferred from a mock expense subcategory label (`'vehicle registration'`) used to train an expense classifier on vehicle-related expenditures. The application does not collect vehicle RC plate numbers.<br>**ComplyDP Action:** **Reject** |
| **13** | `user.employment.esic` | **False Positive** | `budgeting_ml_model/src/pro_plus_dataset_generator.py:214` | `Line 214: 'Income Tax Department', 'GST Portal', 'MCA Portal', 'EPFO', 'ESIC', 'PF Office'` | **Technical Justification:** Static string literal in an array of government portal merchant names used for synthetic training generation. No 17-digit ESIC employee insurance numbers are collected or processed.<br>**ComplyDP Action:** **Reject** |
| **14** | `user.identity.pan` | **False Positive** | `smart_money_ai/intelligence/investment_engine/investment_recommendations_backend.py:735, 746` | `Line 735: "2. Complete KYC with Aadhaar and PAN"`<br>`Line 746: "1. Keep PAN card and Aadhaar ready"` | **Technical Justification:** Hardcoded instructional UI guidance strings. No 10-character alphanumeric PAN regex pattern (`^[A-Z]{5}[0-9]{4}[A-Z]$`), database field, or document upload endpoint exists.<br>**ComplyDP Action:** **Reject** |
| **15** | `user.identity.passport` | **False Positive** | `budgeting_ml_model/src/mega_dataset_generator.py:181`<br>`budgeting_ml_model/src/pro_plus_dataset_generator.py:215` | `Line 181: 'Passport Fee', 'Visa Fee', 'License Fee'`<br>`Line 215: 'Passport Seva', 'Aadhaar Center', 'Jan Aushadhi'` | **Technical Justification:** String literals in mock transaction categories and government merchant lists. No passport number ingestion, validation, or storage exists.<br>**ComplyDP Action:** **Reject** |

---

## 2. False Negatives: Active PII Missed by the Scanner

While the ComplyDP scanner generated 11 false positives on inactive terms, it completely failed to detect **7 active personal data elements** actively stored in the database models and frontend state:

| # | Unflagged Data Field | Exact File Path & Line Reference | How It Is Handled in Codebase | Classification |
|:--|:---------------------|:---------------------------------|:------------------------------|:--------------:|
| **1** | `user.name` (Full Name) | `smart_money_ai/utils/data_manager.py:48`<br>`smart_money_ai/core/smart_money_ai.py:79`<br>`frontend_static/index.html:40` | Stored as `name TEXT NOT NULL` in the primary SQLite `users` table and rendered as user display name in dashboard greeting. | **False Negative** |
| **2** | `user.demographics.age` (User Age) | `smart_money_ai/utils/data_manager.py:51`<br>`smart_money_ai/core/smart_money_ai.py:82` | Defined as `age INTEGER` in the `users` table; used for financial life-cycle modeling and risk tolerance scoring. | **False Negative** |
| **3** | `user.financial.annual_income` | `smart_money_ai/utils/data_manager.py:52`<br>`smart_money_ai/core/smart_money_ai.py:83` | Defined as `annual_income REAL` in `users` table; primary scalar used to calculate debt-to-income and savings capacity. | **False Negative** |
| **4** | `user.financial.risk_profile` | `smart_money_ai/utils/data_manager.py:53`<br>`smart_money_ai/core/smart_money_ai.py:84` | Defined as `risk_profile TEXT` in `users` schema; stores psychometric behavioral risk classification ("conservative", "moderate", "aggressive"). | **False Negative** |
| **5** | `user.financial.goals` | `smart_money_ai/utils/data_manager.py:54, 55`<br>`smart_money_ai/core/smart_money_ai.py:85, 86` | Stored as `investment_timeline INTEGER` and `financial_goals TEXT` in SQLite database. | **False Negative** |
| **6** | `user.financial.transaction_metadata` | `smart_money_ai/utils/data_manager.py:64-77`<br>`smart_money_ai/core/sms_parser/enhanced_sms_parser.py:28-35` | Stores `amount REAL`, `balance REAL`, `merchant TEXT`, and `category TEXT` linking individual transaction behavior to user IDs. | **False Negative** |
| **7** | `user.identity.avatar_url` | `frontend_static/index.html:33` | Hardcoded avatar endpoint transmitting user display name to a third-party service: `https://ui-avatars.com/api/?name=Demo+User...` | **False Negative** |

---

## 3. Vendor & Third-Party Integration Audit

The scanner's **Systems** catalog failed to detect several external third-party services, CDNs, and API SDKs embedded across the repository:

| Vendor / Service | Purpose | Manifest / Import Reference | Scanner Detection Status | Technical Finding |
|:-----------------|:--------|:----------------------------|:------------------------:|:------------------|
| **UI-Avatars CDN** | User Avatar Generation | `frontend_static/index.html:33` | **False Negative** (Undetected) | Transmits raw user name strings in GET query parameters (`?name=Demo+User`) to an external third-party domain. |
| **Ionicons (Unpkg CDN)** | Frontend UI Icons | `frontend_static/index.html:11, 12` | **False Negative** (Undetected) | Loads client-side scripts directly from `unpkg.com` CDN without Subresource Integrity (SRI) hashes. |
| **Google Fonts** | Typography CDN | `frontend_static/index.html:8` | **False Negative** (Undetected) | Loads external stylesheets and fonts from `fonts.googleapis.com`, exposing user IP/User-Agent to Google. |
| **ICICI Securities Breeze API** | Broker Trading Integration | `smart_money_ai/api/breeze_trading_engine.py:27`<br>`integrations/breeze_trading.py:24` | **False Negative** (Undetected) | Connects to external trading endpoint (`breeze_connect`) to execute orders and fetch demat holdings. |
| **Finnhub Market API** | Financial Market Data | `smart_money_ai/data/finnhub_integration.py:17`<br>`integrations/finnhub_market_data.py:16` | **False Negative** (Undetected) | Outbound REST API calls transmitting stock symbols and receiving financial company telemetry. |
| **Kaggle API** | Training Data Downloader | `test_kaggle_adapter.py:28`<br>`scripts/utilities/kaggle_adapter.py:15` | **False Negative** (Undetected) | Python client communicating with `api.kaggle.com` for dataset retrieval. |
| **SQLite3** | Embedded Relational Database | `smart_money_ai/utils/data_manager.py:12` | **Detected** (Core System) | Local file-based database engine (`smart_money.db`) used for all user and transaction tables. |

---

## 4. Technical Analysis of Scanner Error Patterns

A systematic analysis reveals three distinct architectural flaws in the ComplyDP scanner's inference engine:

```
+--------------------------------------------------------------------------------------------------+
|                              COMPLYDP SCANNER FAILURE MODES BREAKDOWN                             |
+--------------------------+------------------------------------+----------------------------------+
| Error Mode               | Example From Repo                  | Root Cause                       |
+--------------------------+------------------------------------+----------------------------------+
| 1. Substring Stemming    | `pran` in "Pranay"                 | Scanner lacks boundary delimiters|
|                          | `uan` in "quantile", "quantity"    | (`\b`). Tokenizes sub-parts of   |
|                          | `abha` in "Sabharwal"              | natural language words.          |
+--------------------------+------------------------------------+----------------------------------+
| 2. Semantic Role         | Label `'vehicle registration'`     | Scanner flags string literals    |
|    Confusion             | Merchant `'Passport Seva'`         | without distinguishing between   |
|                          | Merchant `'ESIC'`                  | a category label and actual PII. |
+--------------------------+------------------------------------+----------------------------------+
| 3. Sanitization Context  | Regex `r'\s+\d{6}$', # pincode`    | Scanner detects AST regexes but  |
|    Blindness             | inside `re.sub(pattern, '')`       | fails to check if the data is    |
|                          |                                    | being discarded or stored.       |
+--------------------------+------------------------------------+----------------------------------+
| 4. Schema & Dataclass    | `user.name`, `age`, `income`,      | Scanner overlooks explicit SQL   |
|    Omission              | `risk_profile` in SQLite schema    | `CREATE TABLE` statements and    |
|                          | and Python dataclasses             | typed dataclass models.          |
+--------------------------+------------------------------------+----------------------------------+
```

---

## 5. Audit Action Items & Summary Matrix

To calibrate the ComplyDP discovery dashboard for `harkan28/smart-money-ai`, the following actions should be executed in the UI:

```
+--------------------------------------------------------------------------------------------------+
|                                    COMPLYDP UI ACTION MATRIX                                     |
+----+------------------------------------+------------------+---------------+---------------------+
| #  | Scanner Identifier                 | Audit Finding    | Recommended   | Verification State  |
|    |                                    |                  | UI Action     |                     |
+----+------------------------------------+------------------+---------------+---------------------+
| 1  | user.financial.ifsc                | False Positive   | ❌ REJECT     | Zero code references|
| 2  | user.identity.aadhaar              | False Positive   | ❌ REJECT     | Static text only    |
| 3  | user.financial.upi                 | True Positive    | ✅ CONFIRM    | Verified in DB & SMS|
| 4  | user.financial.bank_account        | True Positive    | ✅ CONFIRM    | Verified in DB & SMS|
| 5  | user.contact.email                 | True Positive    | ✅ CONFIRM    | Verified in DB model|
| 6  | user.contact.phone                 | True Positive    | ✅ CONFIRM    | Verified in DB model|
| 7  | user.contact.pincode               | False Positive   | ❌ REJECT     | Sanitization regex  |
| 8  | user.financial.pran                | False Positive   | ❌ REJECT     | Substring error     |
| 9  | user.employment.uan                | False Positive   | ❌ REJECT     | Substring error     |
| 10 | user.financial.ckyc                | False Positive   | ❌ REJECT     | Zero code references|
| 11 | user.health.abha                   | False Positive   | ❌ REJECT     | Substring error     |
| 12 | user.identity.vehicle_registration | False Positive   | ❌ REJECT     | Expense category tag|
| 13 | user.employment.esic               | False Positive   | ❌ REJECT     | Mock merchant string|
| 14 | user.identity.pan                  | False Positive   | ❌ REJECT     | Static text only    |
| 15 | user.identity.passport             | False Positive   | ❌ REJECT     | Mock merchant string|
+----+------------------------------------+------------------+---------------+---------------------+
```

### Final Metrics:
* **Precision Rate of ComplyDP Inferences:** `4 / 15 = 26.67%`
* **False Discovery Rate (FDR):** `11 / 15 = 73.33%`
* **Undetected Critical PII Items (False Negatives):** `7`
* **Undetected Third-Party Vendors:** `6`
