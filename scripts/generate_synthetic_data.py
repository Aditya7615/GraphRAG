"""Synthetic enterprise corpus generator.

Produces eight realistic, cross-referencing enterprise documents as PDFs (plus
optional .txt mirrors) so the whole pipeline can be exercised before any real
data exists.

The content is deliberately built with:
  * numbered section headings      -> exercises section-level citation mapping
  * specific figures and IDs       -> exercises BM25 exact-match retrieval
  * paraphrasable policy prose     -> exercises dense semantic retrieval
  * one deliberate cross-document contradiction (see CONTRADICTION note)
  * facts that are absent on purpose so you can verify the refusal guardrail

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --out data/synthetic --format both
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faker import Faker

from scripts.corpus_extra import EXTRA_BUILDERS

fake = Faker("en_US")
Faker.seed(42)
random.seed(42)

Section = tuple[str, list[str]]
Document = tuple[str, str, str, list[Section]]  # filename, doc_type, title, sections


def _financial_statements() -> Document:
    sections: list[Section] = [
        (
            "1. Consolidated Statement of Operations",
            [
                "Total revenue for the third quarter of fiscal year 2025 was $4,182.6 million, "
                "an increase of 18.4% compared with $3,532.1 million in the third quarter of "
                "fiscal year 2024. Subscription revenue accounted for $3,110.2 million, or 74.4% "
                "of total revenue, while professional services contributed $1,072.4 million.",
                "Cost of revenue was $1,254.8 million, producing a gross profit of $2,927.8 "
                "million and a gross margin of 70.0%, up 140 basis points year over year. The "
                "margin expansion was driven primarily by improved infrastructure utilisation "
                "following the migration described in the Cloud Architecture Guide.",
                "Operating expenses totalled $2,201.5 million, comprising research and "
                "development of $982.3 million, sales and marketing of $874.9 million, and "
                "general and administrative of $344.3 million. Income from operations was "
                "$726.3 million, representing an operating margin of 17.4%.",
                "Net income attributable to common shareholders was $548.9 million, or $2.14 per "
                "diluted share, compared with $401.2 million, or $1.58 per diluted share, in the "
                "prior-year period. The effective tax rate for the quarter was 21.3%.",
            ],
        ),
        (
            "2. Segment Performance",
            [
                "The Enterprise Platform segment generated revenue of $2,644.1 million, up 22.1% "
                "year over year, with a segment operating margin of 24.8%. Net revenue retention "
                "for this segment was 118% on a trailing twelve-month basis.",
                "The Data and Analytics segment generated revenue of $1,048.7 million, up 14.2% "
                "year over year, with a segment operating margin of 11.3%. Growth was moderated "
                "by longer sales cycles in the financial services vertical.",
                "The Emerging Products segment generated revenue of $489.8 million and recorded a "
                "segment operating loss of $62.4 million as the company continued to invest in "
                "the AI assistant product line launched in the second quarter.",
                "By geography, the Americas contributed 58.2% of total revenue, EMEA contributed "
                "27.6%, and APAC contributed 14.2%. APAC was the fastest-growing region at 31.5% "
                "year-over-year growth on a constant-currency basis.",
            ],
        ),
        (
            "3. Balance Sheet and Liquidity",
            [
                "Cash, cash equivalents and short-term investments totalled $6,842.0 million as of "
                "30 September 2025, compared with $5,913.4 million as of 31 December 2024. Total "
                "assets were $18,204.7 million and total liabilities were $7,881.9 million.",
                "The company holds $2,500.0 million of senior unsecured notes maturing in 2029 at "
                "a fixed coupon of 4.25%, and maintains an undrawn $1,000.0 million revolving "
                "credit facility that matures in March 2028. The facility carries a financial "
                "covenant requiring a consolidated leverage ratio of no more than 3.50 to 1.00.",
                "Deferred revenue was $4,116.3 million, of which $3,702.8 million is classified as "
                "current. Remaining performance obligations were $12,940.5 million, approximately "
                "62% of which the company expects to recognise as revenue within twenty-four "
                "months.",
            ],
        ),
        (
            "4. Cash Flow",
            [
                "Net cash provided by operating activities was $1,104.2 million for the quarter, "
                "compared with $861.7 million in the prior-year period. Capital expenditures were "
                "$182.9 million, resulting in free cash flow of $921.3 million.",
                "The company repurchased 3.1 million shares for $412.0 million during the quarter "
                "under the $3,000.0 million repurchase programme authorised in February 2025. "
                "As of 30 September 2025, $1,684.0 million remained available under the programme.",
            ],
        ),
        (
            "5. Outlook",
            [
                "For the fourth quarter of fiscal year 2025, the company expects total revenue "
                "between $4,380 million and $4,440 million, and non-GAAP operating margin between "
                "19.0% and 19.5%. For the full fiscal year 2025, the company expects total revenue "
                "between $16,290 million and $16,350 million.",
                "This outlook assumes no material change in foreign exchange rates from those "
                "prevailing on 30 September 2025 and excludes the impact of any acquisitions not "
                "yet closed.",
            ],
        ),
        (
            "6. Risk Factors Summary",
            [
                "Concentration risk: the ten largest customers accounted for 14.8% of total "
                "revenue in the quarter, down from 16.9% a year earlier. No single customer "
                "accounted for more than 3% of total revenue.",
                "Foreign currency risk: approximately 41.8% of revenue was denominated in "
                "currencies other than the US dollar. The company hedges a rolling twelve-month "
                "window of forecast EUR and GBP exposures using forward contracts.",
            ],
        ),
    ]
    return (
        "Q3_2025_Consolidated_Financial_Statements.pdf",
        "financial",
        "Northwind Systems Inc. — Q3 FY2025 Consolidated Financial Statements (Unaudited)",
        sections,
    )


def _employee_handbook() -> Document:
    sections: list[Section] = [
        (
            "1. Working Hours and Flexibility",
            [
                "Standard working hours are 09:00 to 17:30 local time, Monday through Friday, "
                "totalling 37.5 paid hours per week. Core collaboration hours, during which all "
                "employees are expected to be reachable, are 11:00 to 15:00 in the employee's "
                "registered time zone.",
                "Employees may adjust their start time by up to two hours in either direction "
                "without prior approval, provided core collaboration hours are preserved and the "
                "employee's manager is notified through the team calendar.",
                "The company operates a hybrid working model. Employees assigned to an office "
                "location are expected on site a minimum of eight days per calendar month. Fully "
                "remote status requires written approval from a Vice President or above and is "
                "reviewed annually.",
            ],
        ),
        (
            "2. Paid Time Off",
            [
                "All full-time employees accrue 25 days of paid annual leave per calendar year, "
                "accruing at 2.083 days per completed month of service. Employees with five or "
                "more years of continuous service accrue 30 days per calendar year.",
                "A maximum of 10 unused annual leave days may be carried into the following "
                "calendar year and must be used before 31 March. Days not used by that date are "
                "forfeited without compensation, except where local law requires otherwise.",
                "Annual leave requests of five consecutive days or more require at least three "
                "weeks' notice. Requests are approved by the direct manager and are subject to "
                "team coverage requirements.",
                "The company observes 11 public holidays per year in the United States. Employees "
                "in other jurisdictions observe the statutory holidays of their country of "
                "employment.",
            ],
        ),
        (
            "3. Parental and Family Leave",
            [
                "Primary caregivers receive 20 weeks of fully paid parental leave, which may be "
                "taken in up to three separate blocks within the first 18 months following birth "
                "or placement for adoption. Secondary caregivers receive 12 weeks of fully paid "
                "leave under the same conditions.",
                "Parental leave is available to all employees from their first day of employment; "
                "there is no minimum service requirement. Employees returning from parental leave "
                "are entitled to a phased return of up to four weeks at 80% hours on full pay.",
                "Compassionate leave of up to 10 working days per event is available following the "
                "death of an immediate family member. Additional unpaid leave may be granted at "
                "the discretion of the People Business Partner.",
            ],
        ),
        (
            "4. Expenses and Travel",
            [
                "Expense claims must be submitted within 45 days of the date the expense was "
                "incurred. Claims submitted after 45 days require written approval from a "
                "Director or above and may be declined.",
                "Domestic economy airfare is standard for all employees. Premium economy is "
                "permitted for flights with a scheduled duration exceeding six hours. Business "
                "class requires Senior Vice President approval and is permitted only for flights "
                "exceeding ten hours.",
                "The daily meal allowance is $75 in the United States, EUR 65 in the euro area, "
                "and GBP 55 in the United Kingdom. Receipts are required for any single expense "
                "of $25 or more.",
                "Personal vehicle mileage is reimbursed at the prevailing IRS standard rate. "
                "Ride-hailing and taxi expenses are reimbursable for business travel and for "
                "journeys home after 21:00 following required work.",
            ],
        ),
        (
            "5. Performance and Compensation Review",
            [
                "Performance is reviewed twice per year, in a March cycle and a September cycle. "
                "Compensation adjustments arising from the March cycle take effect on 1 May, and "
                "adjustments from the September cycle take effect on 1 November.",
                "Employees are rated on a five-point scale: Below Expectations, Developing, "
                "Meets Expectations, Exceeds Expectations, and Outstanding. There is no forced "
                "distribution across teams.",
                "The annual bonus target is 10% of base salary for individual contributors, 15% "
                "for managers, and 25% for Directors and above. Bonus payout is a function of "
                "company performance multiplier and individual rating, and is paid in March.",
            ],
        ),
        (
            "6. Code of Conduct and Reporting",
            [
                "All employees must complete annual Code of Conduct training within 30 days of the "
                "training being assigned. Failure to complete mandatory training may result in "
                "withholding of the annual bonus.",
                "Concerns about unethical or unlawful conduct may be raised with a manager, the "
                "People team, or anonymously through the independent ethics hotline. The company "
                "prohibits retaliation against anyone who raises a concern in good faith.",
            ],
        ),
    ]
    return ("Employee_Handbook_2025.pdf", "hr", "Northwind Systems — Global Employee Handbook 2025", sections)


def _cloud_architecture_guide() -> Document:
    sections: list[Section] = [
        (
            "1. Account and Landing Zone Topology",
            [
                "The platform runs on AWS using a multi-account landing zone managed by AWS "
                "Control Tower. Accounts are grouped into four organisational units: Security, "
                "Infrastructure, Workloads-Prod, and Workloads-NonProd. Production workloads are "
                "never deployed into an account that also hosts non-production workloads.",
                "Every workload account is provisioned with a standard VPC using a /16 CIDR from "
                "the 10.0.0.0/8 range, with three private subnets, three public subnets and three "
                "isolated database subnets spread across three Availability Zones in us-east-1.",
                "Cross-account access is granted exclusively through IAM roles assumed via AWS IAM "
                "Identity Center. Long-lived IAM user access keys are prohibited in all accounts; "
                "an SCP named DenyIAMUserCreation enforces this at the organisation level.",
            ],
        ),
        (
            "2. Compute and Container Platform",
            [
                "Stateless services run on Amazon EKS version 1.30 using managed node groups of "
                "m7g.2xlarge Graviton instances, with Karpenter handling just-in-time node "
                "provisioning. Target steady-state cluster utilisation is 65% CPU.",
                "Each service is deployed as a Kubernetes Deployment with a minimum of three "
                "replicas spread across Availability Zones using a topology spread constraint with "
                "maxSkew of 1. Pod Disruption Budgets require at least two available replicas "
                "during voluntary disruptions.",
                "Batch and event-driven workloads run on AWS Lambda with a maximum timeout of 900 "
                "seconds and reserved concurrency set per function. Functions exceeding 900 "
                "seconds of expected runtime must be implemented as AWS Step Functions state "
                "machines instead.",
            ],
        ),
        (
            "3. Data Layer",
            [
                "The primary transactional store is Amazon Aurora PostgreSQL 15 in a Multi-AZ "
                "cluster with one writer and two readers. Automated backups are retained for 35 "
                "days and point-in-time recovery is enabled.",
                "Object storage uses Amazon S3 with SSE-KMS encryption and customer-managed keys. "
                "All buckets have Block Public Access enabled at the account level, versioning "
                "enabled, and a lifecycle policy transitioning objects to S3 Intelligent-Tiering "
                "after 30 days.",
                "Analytical workloads read from an S3-based data lake in Apache Iceberg format, "
                "queried through Amazon Athena. The raw zone retains data for seven years; the "
                "curated zone retains data for three years.",
                "Caching uses Amazon ElastiCache for Redis 7 in cluster mode with encryption in "
                "transit and at rest. Cache keys must carry an explicit TTL; keys without a TTL "
                "are rejected by the shared client library.",
            ],
        ),
        (
            "4. Networking and Edge",
            [
                "Public traffic terminates at Amazon CloudFront with AWS WAF attached. The WAF web "
                "ACL enforces a rate-based rule of 2,000 requests per five minutes per source IP "
                "address, plus the AWS Managed Rules Core Rule Set.",
                "Internal service-to-service traffic uses AWS PrivateLink endpoints. No workload "
                "subnet has a route to an internet gateway; egress is centralised through a "
                "shared inspection VPC running AWS Network Firewall.",
                "The recovery time objective (RTO) for tier-1 services is 15 minutes and the "
                "recovery point objective (RPO) is 5 minutes. Tier-2 services carry an RTO of 4 "
                "hours and an RPO of 1 hour.",
            ],
        ),
        (
            "5. Observability",
            [
                "Metrics are collected by Amazon Managed Service for Prometheus and visualised in "
                "Amazon Managed Grafana. Logs are shipped to Amazon CloudWatch Logs with a 30-day "
                "retention in the log group and a 400-day archive in S3.",
                "Distributed tracing uses AWS X-Ray with a 5% head-based sampling rate for normal "
                "traffic and 100% sampling for requests that return a 5xx status code.",
                "Every tier-1 service must define service level objectives for availability and "
                "latency. The standard availability SLO is 99.95% measured over a rolling 28-day "
                "window, and the standard latency SLO is p99 below 400 milliseconds.",
            ],
        ),
        (
            "6. Cost Governance",
            [
                "All resources must carry the tags CostCenter, Owner, Environment and DataClass. "
                "Untagged resources are flagged daily and are subject to automated termination in "
                "non-production accounts after 14 days.",
                "Compute savings plans cover a committed baseline of 70% of steady-state compute "
                "spend. Spot capacity is used for non-production and for stateless batch "
                "processing, with a maximum spot allocation of 40% of any single node group.",
            ],
        ),
    ]
    return (
        "AWS_Cloud_Architecture_Guide.pdf",
        "engineering",
        "Northwind Systems — AWS Cloud Architecture Guide v4.2",
        sections,
    )


def _security_policy() -> Document:
    sections: list[Section] = [
        (
            "1. Scope and Data Classification",
            [
                "This policy applies to all employees, contractors and third parties who access "
                "Northwind Systems information assets. Data is classified into four levels: "
                "Public, Internal, Confidential, and Restricted.",
                "Restricted data includes customer financial records, authentication credentials, "
                "and personal data subject to GDPR Article 9. Restricted data must never be "
                "stored on endpoint devices and must never be transmitted over unencrypted "
                "channels.",
            ],
        ),
        (
            "2. Access Control",
            [
                "Access is granted on the principle of least privilege and is reviewed quarterly. "
                "Access review campaigns must be completed by the asset owner within 14 days of "
                "being issued.",
                "Multi-factor authentication is mandatory for all systems. Phishing-resistant "
                "FIDO2 hardware keys are required for administrative access to production "
                "environments; time-based one-time passwords are not accepted for these roles.",
                "Privileged access to production is granted just-in-time for a maximum of four "
                "hours per elevation request and requires a linked change ticket or incident "
                "number.",
            ],
        ),
        (
            "3. Encryption Standards",
            [
                "Data at rest must be encrypted using AES-256. Data in transit must use TLS 1.2 or "
                "higher; TLS 1.0 and 1.1 are prohibited. Certificates must use a minimum key size "
                "of RSA 2048 bits or ECDSA P-256.",
                "Encryption keys are managed in AWS KMS with automatic annual rotation enabled. "
                "Key deletion requires a mandatory 30-day waiting period and dual authorisation "
                "from two members of the Security Engineering team.",
            ],
        ),
        (
            "4. Vulnerability Management",
            [
                "Vulnerabilities are remediated according to severity: Critical within 7 calendar "
                "days, High within 30 calendar days, Medium within 90 calendar days, and Low "
                "within 180 calendar days. The clock starts when the vulnerability is first "
                "reported by the scanning platform.",
                "All container images are scanned at build time and again daily in the registry. "
                "Images with unresolved Critical findings are blocked from deployment to "
                "production by an admission controller policy.",
                "Penetration testing of internet-facing systems is performed by an independent "
                "third party at least annually, and after any material architectural change.",
            ],
        ),
        (
            "5. Incident Response",
            [
                "Security incidents are classified SEV-1 through SEV-4. A SEV-1 incident requires "
                "notification of the Chief Information Security Officer within 15 minutes of "
                "declaration and activation of the incident bridge.",
                "Where a personal data breach is confirmed, the Data Protection Officer must "
                "notify the lead supervisory authority within 72 hours of the company becoming "
                "aware of the breach, in accordance with GDPR Article 33.",
                "A written post-incident review is required for every SEV-1 and SEV-2 incident and "
                "must be published within 10 business days of incident resolution.",
            ],
        ),
        (
            "6. Third-Party and Vendor Risk",
            [
                "Vendors processing Confidential or Restricted data must complete a security "
                "assessment before contract signature and annually thereafter. Vendors handling "
                "Restricted data must hold a current SOC 2 Type II report or ISO/IEC 27001 "
                "certification.",
                "All vendor contracts involving personal data must include a data processing "
                "addendum with the standard contractual clauses where transfers leave the "
                "European Economic Area.",
            ],
        ),
    ]
    return (
        "Information_Security_Policy.pdf",
        "compliance",
        "Northwind Systems — Information Security Policy (ISP-2025.3)",
        sections,
    )


def _api_reference() -> Document:
    sections: list[Section] = [
        (
            "1. Authentication",
            [
                "All requests to the Payments API must include an Authorization header carrying a "
                "bearer token: Authorization: Bearer <access_token>. Tokens are issued by the "
                "OAuth 2.0 token endpoint at https://auth.northwind.example/oauth2/token using "
                "the client_credentials grant.",
                "Access tokens expire after 3600 seconds. Clients should refresh a token when it "
                "is within 300 seconds of expiry. Requests with an expired token return HTTP 401 "
                "with error code token_expired.",
            ],
        ),
        (
            "2. Rate Limits",
            [
                "The default rate limit is 1,000 requests per minute per API key for read "
                "operations and 300 requests per minute for write operations. Exceeding the limit "
                "returns HTTP 429 with a Retry-After header expressed in seconds.",
                "Rate limit state is reported on every response through the X-RateLimit-Limit, "
                "X-RateLimit-Remaining and X-RateLimit-Reset headers. Enterprise-tier customers "
                "may request a limit of up to 10,000 requests per minute.",
            ],
        ),
        (
            "3. Create Payment",
            [
                "POST /v2/payments creates a payment intent. Required fields are amount (integer, "
                "minor units), currency (ISO 4217, lowercase), and destination_account_id. The "
                "optional idempotency_key field accepts a string of up to 255 characters.",
                "Idempotency keys are retained for 24 hours. Replaying a request with the same "
                "idempotency key within that window returns the original response with HTTP 200 "
                "instead of creating a duplicate payment.",
                "A successful creation returns HTTP 201 with a payment object whose status is "
                "pending. Terminal statuses are succeeded, failed and cancelled. Payments remain "
                "in processing for a maximum of 72 hours before being automatically cancelled.",
            ],
        ),
        (
            "4. Error Codes",
            [
                "insufficient_funds (HTTP 402): the source account balance is lower than the "
                "requested amount. This error is not retryable without a balance change.",
                "invalid_destination (HTTP 400): the destination_account_id does not exist or is "
                "closed. account_frozen (HTTP 403): the account is under compliance review.",
                "rate_limited (HTTP 429): the client exceeded its quota; retry after the interval "
                "in the Retry-After header. upstream_timeout (HTTP 504): the downstream banking "
                "partner did not respond within 30 seconds; this error is safe to retry with the "
                "same idempotency key.",
            ],
        ),
        (
            "5. Webhooks",
            [
                "Webhook deliveries are signed with an HMAC-SHA256 signature in the "
                "X-Northwind-Signature header. Consumers must verify the signature using their "
                "webhook secret and reject any request whose timestamp is more than 300 seconds "
                "old to prevent replay attacks.",
                "Failed deliveries are retried with exponential backoff at 1, 5, 25 and 125 "
                "minutes, for a maximum of 8 attempts over 24 hours. After the final attempt the "
                "event is moved to the dead letter queue and surfaced in the developer dashboard.",
            ],
        ),
        (
            "6. Versioning and Deprecation",
            [
                "The API version is pinned per API key and may be overridden per request with the "
                "Northwind-Version header. Breaking changes are only introduced in a new major "
                "version.",
                "Deprecated versions are supported for a minimum of 12 months from the "
                "deprecation announcement. Version v1 was deprecated on 15 January 2025 and will "
                "reach end of life on 15 January 2026.",
            ],
        ),
    ]
    return ("Payments_API_Reference_v2.pdf", "api", "Northwind Payments API — Developer Reference v2", sections)


def _incident_postmortem() -> Document:
    sections: list[Section] = [
        (
            "1. Incident Summary",
            [
                "Incident INC-2291 was declared at 14:07 UTC on 12 August 2025 and resolved at "
                "17:42 UTC the same day, for a total customer-impacting duration of 3 hours and "
                "35 minutes. The incident was classified SEV-1.",
                "Approximately 34% of payment creation requests returned HTTP 504 during the "
                "impact window. An estimated 128,400 payment attempts failed. No data was lost "
                "and no unauthorised access occurred.",
            ],
        ),
        (
            "2. Timeline",
            [
                "14:02 UTC — A configuration change increased the Aurora PostgreSQL connection "
                "pool maximum from 200 to 400 per service instance, deployed as change CHG-8841.",
                "14:07 UTC — Automated alerting fired on p99 latency breaching the 400 ms SLO. The "
                "on-call engineer declared SEV-1 four minutes later.",
                "14:55 UTC — The team identified connection exhaustion on the Aurora writer "
                "instance, which reached its 5,000 connection ceiling.",
                "15:40 UTC — Change CHG-8841 was rolled back. Error rates began declining but did "
                "not fully recover because stale connections were retained by the pooler.",
                "17:15 UTC — Rolling restart of the payment service completed. 17:42 UTC — Error "
                "rates returned to baseline and the incident was resolved.",
            ],
        ),
        (
            "3. Root Cause",
            [
                "The direct cause was connection pool exhaustion on the Aurora writer. The pool "
                "size increase was applied to all 40 service instances simultaneously, producing "
                "a theoretical maximum of 16,000 connections against a database ceiling of 5,000.",
                "The contributing cause was the absence of a pre-deployment validation check "
                "comparing aggregate pool capacity against the database max_connections "
                "parameter. The change was approved without this calculation being performed.",
                "A secondary contributing factor was that the connection pooler did not evict "
                "stale connections on configuration rollback, extending recovery by roughly 95 "
                "minutes beyond the rollback.",
            ],
        ),
        (
            "4. What Went Well",
            [
                "Alerting fired within 5 minutes of impact beginning, and the incident bridge was "
                "staffed within 8 minutes of declaration. Customer status page updates were "
                "published every 30 minutes throughout the incident.",
            ],
        ),
        (
            "5. Action Items",
            [
                "ACT-1: Add an automated pre-deployment check that validates aggregate connection "
                "pool capacity against database max_connections. Owner: Platform Engineering. Due "
                "15 September 2025. Status: completed.",
                "ACT-2: Require progressive rollout for all database configuration changes, "
                "beginning at 10% of instances with a 30-minute soak. Owner: Release Engineering. "
                "Due 30 September 2025. Status: completed.",
                "ACT-3: Upgrade the connection pooler to a version that evicts stale connections "
                "on config change. Owner: Data Platform. Due 31 October 2025. Status: in progress.",
                "ACT-4: Introduce a database connection saturation dashboard with an alert at 80% "
                "of max_connections. Owner: Observability. Due 20 September 2025. Status: "
                "completed.",
            ],
        ),
    ]
    return (
        "Incident_Postmortem_INC-2291.pdf",
        "operations",
        "Post-Incident Review — INC-2291 Payment API Degradation",
        sections,
    )


def _vendor_contract() -> Document:
    sections: list[Section] = [
        (
            "1. Term and Renewal",
            [
                "This Master Services Agreement (MSA-2025-0147) is effective from 1 April 2025 and "
                "continues for an initial term of 36 months, expiring 31 March 2028. The agreement "
                "renews automatically for successive 12-month terms unless either party gives "
                "written notice of non-renewal at least 90 days before the end of the then-current "
                "term.",
            ],
        ),
        (
            "2. Fees and Payment Terms",
            [
                "The annual platform subscription fee is $1,850,000, invoiced annually in advance. "
                "Professional services are charged at a blended rate of $215 per hour against a "
                "pre-purchased pool of 4,000 hours per contract year.",
                "Invoices are payable within 45 days of the invoice date. Undisputed amounts "
                "outstanding beyond 45 days accrue interest at 1.0% per month or the maximum rate "
                "permitted by law, whichever is lower.",
                "Fees may increase at each renewal by no more than the lesser of 5% or the "
                "increase in the US Consumer Price Index for All Urban Consumers over the "
                "preceding 12 months.",
            ],
        ),
        (
            "3. Service Levels",
            [
                "The vendor commits to a monthly uptime percentage of 99.9% for the production "
                "service. Uptime excludes scheduled maintenance notified at least 5 business days "
                "in advance, capped at 4 hours per calendar month.",
                "Service credits apply as follows: below 99.9% but at or above 99.0%, a credit of "
                "10% of the monthly fee; below 99.0% but at or above 95.0%, a credit of 25%; below "
                "95.0%, a credit of 50%. Credits must be requested within 30 days of the end of "
                "the affected month and are the sole remedy for availability failures.",
                "Support response targets are 30 minutes for Priority 1, 4 hours for Priority 2, "
                "and 1 business day for Priority 3.",
            ],
        ),
        (
            "4. Liability and Indemnity",
            [
                "Each party's aggregate liability under this agreement is limited to the fees paid "
                "or payable in the 12 months preceding the event giving rise to the claim. This "
                "cap does not apply to breaches of confidentiality, indemnification obligations, "
                "or a party's gross negligence or wilful misconduct.",
                "The vendor indemnifies the customer against third-party claims that the service "
                "infringes any patent, copyright or trade secret, provided the customer notifies "
                "the vendor promptly and grants control of the defence.",
            ],
        ),
        (
            "5. Data Protection and Security",
            [
                "The vendor acts as a data processor and shall process personal data only on the "
                "documented instructions of the customer. The vendor must notify the customer of "
                "any personal data breach without undue delay and in any event within 24 hours of "
                "becoming aware of it.",
                "The vendor shall maintain a SOC 2 Type II report and provide a copy annually. The "
                "customer may audit the vendor's security controls once per contract year on 30 "
                "days' written notice.",
            ],
        ),
        (
            "6. Termination",
            [
                "Either party may terminate for material breach if the breach remains uncured 30 "
                "days after written notice. The customer may terminate for convenience on 120 "
                "days' written notice, subject to payment of fees for the remainder of the "
                "then-current contract year.",
                "On termination the vendor shall return or securely delete all customer data "
                "within 60 days and certify the deletion in writing. Data export is provided in "
                "JSON or CSV format at no additional charge.",
            ],
        ),
    ]
    return (
        "Vendor_MSA_Contoso_Analytics.pdf",
        "legal",
        "Master Services Agreement MSA-2025-0147 — Contoso Analytics Ltd.",
        sections,
    )


def _investment_mandate() -> Document:
    sections: list[Section] = [
        (
            "1. Investment Objective",
            [
                "The Global Equity Growth Fund seeks long-term capital appreciation by investing "
                "primarily in listed equity securities of companies domiciled in developed and "
                "emerging markets. The fund's benchmark is the MSCI All Country World Index "
                "measured in US dollars on a net total return basis.",
                "The fund targets annualised outperformance of the benchmark by 200 to 300 basis "
                "points gross of fees over a rolling five-year period, with an ex-ante tracking "
                "error maintained between 4% and 7%.",
            ],
        ),
        (
            "2. Eligible Instruments",
            [
                "The fund may invest in common stock, preferred stock, depositary receipts, and "
                "listed equity index futures for efficient portfolio management. Investment in "
                "unlisted securities is limited to 5% of net asset value.",
                "The fund may not employ leverage. Gross exposure shall not exceed 100% of net "
                "asset value, and short selling of individual securities is prohibited.",
                "Derivatives may be used solely for hedging and efficient portfolio management. "
                "Total derivative notional exposure is limited to 20% of net asset value.",
            ],
        ),
        (
            "3. Concentration and Diversification Limits",
            [
                "No single issuer may exceed 5% of net asset value at the time of purchase, and no "
                "single issuer may exceed 7% of net asset value on a passive basis following "
                "market appreciation.",
                "Sector exposure may deviate from the benchmark by no more than plus or minus 8 "
                "percentage points. Country exposure may deviate by no more than plus or minus 10 "
                "percentage points, and emerging market exposure is capped at 30% of net asset "
                "value.",
                "The portfolio shall hold a minimum of 45 and a maximum of 90 positions. Cash and "
                "cash equivalents shall not exceed 10% of net asset value other than during "
                "subscription or redemption settlement periods.",
            ],
        ),
        (
            "4. Risk Management",
            [
                "Ex-ante tracking error is measured daily using a multi-factor risk model. A "
                "breach of the 7% upper bound must be remediated within 10 business days and "
                "reported to the Investment Risk Committee at its next scheduled meeting.",
                "The maximum permitted drawdown before mandatory Investment Risk Committee review "
                "is 20% peak to trough over any rolling 12-month period. Value at risk is "
                "monitored at the 95% confidence level over a one-day horizon.",
                "Liquidity is monitored such that at least 85% of the portfolio can be liquidated "
                "within five trading days assuming 20% of average daily volume participation.",
            ],
        ),
        (
            "5. Fees and Reporting",
            [
                "The management fee is 65 basis points per annum of net asset value, accrued daily "
                "and paid quarterly in arrears. A performance fee of 15% of returns above the "
                "benchmark applies, subject to a high water mark and measured over a rolling "
                "three-year period.",
                "Portfolio holdings are reported to the client monthly within 10 business days of "
                "month end. Performance attribution and risk reporting are delivered quarterly "
                "within 20 business days of quarter end.",
            ],
        ),
        (
            "6. Restrictions and Exclusions",
            [
                "The fund excludes issuers deriving more than 10% of revenue from thermal coal "
                "extraction, tobacco production, or controversial weapons. Exclusion screening is "
                "refreshed quarterly using the client's designated ESG data provider.",
                "Securities of the investment manager's parent company and its affiliates may not "
                "be held. Any breach of the restrictions in this section must be remedied within "
                "three business days of identification.",
            ],
        ),
    ]
    return (
        "Global_Equity_Growth_Fund_Investment_Mandate.pdf",
        "investment",
        "Global Equity Growth Fund — Investment Management Mandate (IMM-2025-04)",
        sections,
    )


def _it_onboarding_runbook() -> Document:
    # CONTRADICTION note: section 3 below states a 21-day access review window,
    # while Information_Security_Policy section 2 states 14 days. This is
    # intentional - it lets you verify that the model surfaces conflicting
    # sources instead of silently picking one.
    sections: list[Section] = [
        (
            "1. Day One Provisioning",
            [
                "New joiner accounts are created in the identity provider 3 business days before "
                "the start date. The hiring manager must submit the onboarding request at least "
                "10 business days before the start date for hardware to arrive on time.",
                "Standard issue hardware for engineering roles is a 16-inch laptop with 36 GB of "
                "memory and 1 TB of storage. Non-engineering roles receive a 14-inch laptop with "
                "18 GB of memory and 512 GB of storage.",
            ],
        ),
        (
            "2. Baseline Access Bundle",
            [
                "Every new employee receives the baseline bundle: email and calendar, the "
                "corporate chat workspace, the HR self-service portal, the expense system, and "
                "read access to the internal documentation wiki.",
                "Access to production systems, customer data, or financial systems is never part "
                "of the baseline bundle and must be requested individually with manager approval "
                "and a documented business justification.",
            ],
        ),
        (
            "3. Access Reviews",
            [
                "Asset owners receive an access review campaign each quarter and are expected to "
                "complete the review within 21 days of receipt. Uncompleted campaigns are "
                "escalated to the owner's manager.",
                "Accounts with no successful authentication in 45 consecutive days are "
                "automatically disabled. Disabled accounts are deleted after a further 90 days.",
            ],
        ),
        (
            "4. Offboarding",
            [
                "All access is revoked within 2 hours of the termination effective time. Hardware "
                "must be returned within 5 business days; the final payslip may be withheld "
                "pending return where local law permits.",
                "Mailbox contents are retained for 90 days and delegated to the departing "
                "employee's manager, after which the mailbox is permanently deleted unless a "
                "litigation hold applies.",
            ],
        ),
    ]
    return (
        "IT_Onboarding_and_Access_Runbook.pdf",
        "operations",
        "Northwind Systems — IT Onboarding and Access Management Runbook",
        sections,
    )


BUILDERS = (
    _financial_statements,
    _employee_handbook,
    _cloud_architecture_guide,
    _security_policy,
    _api_reference,
    _incident_postmortem,
    _vendor_contract,
    _investment_mandate,
    _it_onboarding_runbook,
) + EXTRA_BUILDERS


# --------------------------------------------------------------------------
# Appendices
#
# Real enterprise documents carry long reference tables, and those tables are
# what makes parent-child chunking worth having: each document grows past the
# 2,000-token parent size and splits into several parents, while the dense
# reference entries give BM25 plenty of exact identifiers to match on.
# --------------------------------------------------------------------------
def _paragraphize(entries: list[str], per_paragraph: int = 4) -> list[str]:
    return [
        " ".join(entries[i : i + per_paragraph]) for i in range(0, len(entries), per_paragraph)
    ]


def _appendix_financial() -> list[Section]:
    quarters = [f"Q{q} FY{y}" for y in (2023, 2024, 2025) for q in (1, 2, 3, 4)][:11]
    rows = []
    base = {"Enterprise Platform": 1720.0, "Data and Analytics": 742.0, "Emerging Products": 210.0}
    for i, quarter in enumerate(quarters):
        parts = []
        for segment, start in base.items():
            value = start * (1.0 + 0.048 * i)
            parts.append(f"{segment} ${value:,.1f} million")
        rows.append(f"{quarter}: " + ", ".join(parts) + ".")

    recon = [
        f"{label}: GAAP ${g:,.1f} million, adjustments ${a:,.1f} million, non-GAAP ${g + a:,.1f} million."
        for label, g, a in (
            ("Gross profit", 2927.8, 41.2),
            ("Research and development", 982.3, -118.6),
            ("Sales and marketing", 874.9, -96.4),
            ("General and administrative", 344.3, -52.1),
            ("Income from operations", 726.3, 308.3),
            ("Net income", 548.9, 241.7),
        )
    ]
    return [
        (
            "7. Appendix A — Quarterly Segment Revenue History",
            ["Unaudited quarterly revenue by reportable segment, presented in US dollars."]
            + _paragraphize(rows, 3),
        ),
        (
            "8. Appendix B — GAAP to Non-GAAP Reconciliation",
            [
                "Adjustments consist of stock-based compensation, amortisation of acquired "
                "intangibles, and acquisition-related costs."
            ]
            + _paragraphize(recon, 2),
        ),
    ]


def _appendix_hr() -> list[Section]:
    faqs = [
        ("Can I carry unused annual leave into the next year?",
         "Yes, up to 10 days, and they must be used by 31 March or they are forfeited."),
        ("How much notice is required for a two-week holiday?",
         "At least three weeks' notice, because the request exceeds five consecutive days."),
        ("Do I accrue leave while on parental leave?",
         "Yes, annual leave continues to accrue at the normal rate throughout parental leave."),
        ("Is there a minimum service period before I can take parental leave?",
         "No, parental leave is available from the first day of employment."),
        ("Can parental leave be split into separate blocks?",
         "Yes, primary caregivers may take leave in up to three blocks within 18 months."),
        ("What happens if I submit an expense claim after 45 days?",
         "It requires written approval from a Director or above and may be declined."),
        ("Do I need a receipt for a $20 taxi fare?",
         "No, receipts are required only for single expenses of $25 or more."),
        ("When can I book premium economy?",
         "On flights with a scheduled duration exceeding six hours."),
        ("Who approves business class travel?",
         "A Senior Vice President, and only for flights exceeding ten hours."),
        ("How often is performance reviewed?",
         "Twice per year, in a March cycle and a September cycle."),
        ("When do March-cycle salary increases take effect?",
         "On 1 May of the same year."),
        ("What is the bonus target for a manager?",
         "15% of base salary, compared with 10% for individual contributors."),
        ("Is there a forced distribution of performance ratings?",
         "No, there is no forced distribution across teams."),
        ("How many days per month must office-assigned staff attend on site?",
         "A minimum of eight days per calendar month."),
        ("Who can approve fully remote status?",
         "A Vice President or above, in writing, reviewed annually."),
        ("Can I change my start time without approval?",
         "Yes, by up to two hours, provided core hours of 11:00 to 15:00 are preserved."),
        ("How many public holidays are observed in the United States?",
         "Eleven per year; other jurisdictions follow local statutory holidays."),
        ("What is the compassionate leave entitlement?",
         "Up to 10 working days per event following the death of an immediate family member."),
        ("What happens if I do not complete Code of Conduct training?",
         "The annual bonus may be withheld until the training is completed."),
        ("Can I report an ethics concern anonymously?",
         "Yes, through the independent ethics hotline, and retaliation is prohibited."),
        ("How much leave do I accrue after six years of service?",
         "30 days per calendar year, up from 25 days below five years of service."),
        ("What is the phased return arrangement after parental leave?",
         "Up to four weeks at 80% hours on full pay."),
        ("What is the daily meal allowance in the United Kingdom?",
         "GBP 55 per day."),
        ("Are late-night taxis home reimbursable?",
         "Yes, for journeys home after 21:00 following required work."),
    ]
    entries = [f"Q: {q} A: {a}" for q, a in faqs]
    return [("7. Appendix A — Frequently Asked Questions", _paragraphize(entries, 3))]


def _appendix_aws() -> list[Section]:
    services = [
        ("payments-api", "tier-1", "Payments Engineering", "eks", "99.95%", "p99 400 ms"),
        ("ledger-service", "tier-1", "Payments Engineering", "eks", "99.99%", "p99 250 ms"),
        ("identity-service", "tier-1", "Platform Security", "eks", "99.99%", "p99 150 ms"),
        ("notification-dispatch", "tier-2", "Growth Engineering", "lambda", "99.9%", "p99 900 ms"),
        ("reporting-api", "tier-2", "Data Platform", "eks", "99.9%", "p99 1200 ms"),
        ("ingest-worker", "tier-2", "Data Platform", "eks", "99.9%", "p99 2000 ms"),
        ("fraud-scoring", "tier-1", "Risk Engineering", "eks", "99.95%", "p99 300 ms"),
        ("webhook-relay", "tier-2", "Payments Engineering", "lambda", "99.9%", "p99 800 ms"),
        ("settlement-batch", "tier-2", "Payments Engineering", "step-functions", "99.5%", "n/a"),
        ("kyc-orchestrator", "tier-1", "Compliance Engineering", "step-functions", "99.95%", "n/a"),
        ("audit-log-sink", "tier-1", "Platform Security", "lambda", "99.99%", "p99 500 ms"),
        ("search-indexer", "tier-3", "Data Platform", "eks", "99.0%", "p99 3000 ms"),
        ("billing-reconciler", "tier-2", "Finance Systems", "step-functions", "99.9%", "n/a"),
        ("customer-portal", "tier-1", "Growth Engineering", "eks", "99.95%", "p99 600 ms"),
        ("admin-console", "tier-3", "Internal Tools", "eks", "99.0%", "p99 1500 ms"),
        ("export-service", "tier-3", "Data Platform", "lambda", "99.0%", "p99 5000 ms"),
    ]
    entries = [
        f"Service {name} is classified {tier}, owned by {owner}, deployed on {platform}, "
        f"with an availability SLO of {slo} and a latency SLO of {lat}."
        for name, tier, owner, platform, slo, lat in services
    ]

    tags = [
        f"Tag {tag}: {desc} Permitted values: {values}."
        for tag, desc, values in (
            ("CostCenter", "Identifies the finance cost centre charged for the resource.", "CC-1000 to CC-9999"),
            ("Owner", "Email alias of the owning team, not an individual.", "any @northwind.example group alias"),
            ("Environment", "Deployment stage of the resource.", "prod, staging, dev, sandbox"),
            ("DataClass", "Highest classification of data the resource may hold.", "public, internal, confidential, restricted"),
            ("Compliance", "Regulatory regime the resource falls under, when applicable.", "pci, sox, gdpr, none"),
            ("Backup", "Backup policy applied to the resource.", "daily-35d, weekly-90d, none"),
        )
    ]
    return [
        (
            "7. Appendix A — Service Inventory and Ownership",
            ["Tier assignment determines the applicable RTO, RPO and on-call rotation."]
            + _paragraphize(entries, 3),
        ),
        (
            "8. Appendix B — Mandatory Resource Tags",
            ["All four core tags are enforced by an AWS Config rule in every account."]
            + _paragraphize(tags, 2),
        ),
    ]


def _appendix_security() -> list[Section]:
    controls = [
        ("SEC-001", "Access Control", "Unique user IDs are required; shared accounts are prohibited.", "quarterly"),
        ("SEC-002", "Access Control", "Privileged access is granted just-in-time for a maximum of four hours.", "monthly"),
        ("SEC-003", "Access Control", "FIDO2 hardware keys are required for production administrative access.", "quarterly"),
        ("SEC-004", "Access Control", "Access review campaigns are completed within 14 days of issue.", "quarterly"),
        ("SEC-005", "Access Control", "Terminated user access is revoked within two hours.", "monthly"),
        ("SEC-006", "Cryptography", "Data at rest is encrypted with AES-256.", "annual"),
        ("SEC-007", "Cryptography", "TLS 1.2 is the minimum permitted transport protocol version.", "quarterly"),
        ("SEC-008", "Cryptography", "KMS keys rotate automatically on an annual schedule.", "annual"),
        ("SEC-009", "Cryptography", "Key deletion requires dual authorisation and a 30-day wait.", "annual"),
        ("SEC-010", "Vulnerability", "Critical vulnerabilities are remediated within 7 calendar days.", "weekly"),
        ("SEC-011", "Vulnerability", "High vulnerabilities are remediated within 30 calendar days.", "weekly"),
        ("SEC-012", "Vulnerability", "Container images are scanned at build and daily in the registry.", "continuous"),
        ("SEC-013", "Vulnerability", "Images with Critical findings are blocked from production deployment.", "continuous"),
        ("SEC-014", "Vulnerability", "External penetration testing is performed at least annually.", "annual"),
        ("SEC-015", "Logging", "Security-relevant events are retained for 400 days in immutable storage.", "quarterly"),
        ("SEC-016", "Logging", "Log integrity is protected by object lock and cross-account replication.", "quarterly"),
        ("SEC-017", "Incident", "SEV-1 incidents are escalated to the CISO within 15 minutes.", "per event"),
        ("SEC-018", "Incident", "Personal data breaches are notified to the supervisory authority within 72 hours.", "per event"),
        ("SEC-019", "Incident", "Post-incident reviews are published within 10 business days.", "per event"),
        ("SEC-020", "Incident", "Incident response plans are tested by tabletop exercise twice per year.", "semi-annual"),
        ("SEC-021", "Vendor", "Vendors handling Restricted data hold SOC 2 Type II or ISO/IEC 27001.", "annual"),
        ("SEC-022", "Vendor", "Security assessments precede contract signature for Confidential data.", "per vendor"),
        ("SEC-023", "Vendor", "Data processing addenda include standard contractual clauses for EEA transfers.", "annual"),
        ("SEC-024", "Endpoint", "Full disk encryption is enforced on all managed endpoints.", "continuous"),
        ("SEC-025", "Endpoint", "Endpoint detection and response agents are mandatory and tamper-protected.", "continuous"),
        ("SEC-026", "Endpoint", "Restricted data may not be stored on endpoint devices.", "quarterly"),
        ("SEC-027", "Network", "Workload subnets have no route to an internet gateway.", "quarterly"),
        ("SEC-028", "Network", "Egress traffic is inspected by AWS Network Firewall in a shared VPC.", "quarterly"),
        ("SEC-029", "Network", "WAF rate limiting is set to 2,000 requests per five minutes per source IP.", "quarterly"),
        ("SEC-030", "Awareness", "Security awareness training is completed annually by all staff.", "annual"),
    ]
    entries = [
        f"Control {cid} ({domain}): {desc} Test frequency: {freq}."
        for cid, domain, desc, freq in controls
    ]
    return [
        (
            "7. Appendix A — Control Catalogue",
            [
                "Each control is mapped to SOC 2 Trust Services Criteria and ISO/IEC 27001 "
                "Annex A. Control owners are recorded in the governance risk and compliance "
                "platform."
            ]
            + _paragraphize(entries, 3),
        )
    ]


def _appendix_api() -> list[Section]:
    endpoints = [
        ("POST", "/v2/payments", "Create a payment intent.", "201", "300/min"),
        ("GET", "/v2/payments/{id}", "Retrieve a single payment by identifier.", "200", "1000/min"),
        ("GET", "/v2/payments", "List payments with cursor pagination, 100 per page maximum.", "200", "1000/min"),
        ("POST", "/v2/payments/{id}/cancel", "Cancel a payment that has not reached a terminal status.", "200", "300/min"),
        ("POST", "/v2/payments/{id}/capture", "Capture a previously authorised payment.", "200", "300/min"),
        ("POST", "/v2/refunds", "Create a full or partial refund against a succeeded payment.", "201", "300/min"),
        ("GET", "/v2/refunds/{id}", "Retrieve a refund by identifier.", "200", "1000/min"),
        ("POST", "/v2/accounts", "Create a destination account.", "201", "100/min"),
        ("GET", "/v2/accounts/{id}", "Retrieve account details and current status.", "200", "1000/min"),
        ("PATCH", "/v2/accounts/{id}", "Update mutable account metadata.", "200", "300/min"),
        ("GET", "/v2/accounts/{id}/balance", "Retrieve available and pending balance in minor units.", "200", "1000/min"),
        ("POST", "/v2/payouts", "Initiate a payout to a linked bank account.", "201", "100/min"),
        ("GET", "/v2/payouts/{id}", "Retrieve a payout by identifier.", "200", "1000/min"),
        ("POST", "/v2/webhooks", "Register a webhook endpoint and receive its signing secret.", "201", "60/min"),
        ("DELETE", "/v2/webhooks/{id}", "Deregister a webhook endpoint.", "204", "60/min"),
        ("GET", "/v2/events", "List webhook events from the last 30 days.", "200", "1000/min"),
        ("POST", "/v2/events/{id}/replay", "Replay a single webhook event delivery.", "202", "60/min"),
        ("GET", "/v2/disputes", "List open disputes and their evidence deadlines.", "200", "1000/min"),
        ("POST", "/v2/disputes/{id}/evidence", "Submit dispute evidence before the deadline.", "200", "60/min"),
        ("GET", "/v2/reports/settlement", "Download a settlement report as CSV for a given date.", "200", "30/min"),
    ]
    entries = [
        f"{method} {path} — {desc} Success status {status}. Rate limit {limit}."
        for method, path, desc, status, limit in endpoints
    ]

    fields = [
        f"Field {name} ({ftype}, {req}): {desc}"
        for name, ftype, req, desc in (
            ("amount", "integer", "required", "Charge amount in the minor unit of the currency."),
            ("currency", "string", "required", "Lowercase ISO 4217 code, for example usd or eur."),
            ("destination_account_id", "string", "required", "Identifier of an active destination account."),
            ("idempotency_key", "string", "optional", "Up to 255 characters, retained for 24 hours."),
            ("description", "string", "optional", "Free-text description shown on statements, 140 characters maximum."),
            ("metadata", "object", "optional", "Up to 20 key-value pairs, 500 characters per value."),
            ("capture_method", "string", "optional", "Either automatic or manual; defaults to automatic."),
            ("statement_descriptor", "string", "optional", "22 characters maximum, alphanumeric and spaces only."),
        )
    ]
    return [
        (
            "7. Appendix A — Endpoint Catalogue",
            ["All endpoints are served from https://api.northwind.example and require TLS 1.2 or higher."]
            + _paragraphize(entries, 3),
        ),
        (
            "8. Appendix B — Payment Object Field Reference",
            ["Unknown fields are rejected with HTTP 400 and error code unknown_parameter."]
            + _paragraphize(fields, 2),
        ),
    ]


def _appendix_incident() -> list[Section]:
    log = [
        ("13:58", "Change CHG-8841 approved by the release manager without a connection capacity calculation."),
        ("14:02", "Deployment of CHG-8841 begins across all 40 payment service instances simultaneously."),
        ("14:04", "Aurora writer active connections rise from 1,850 to 3,400 within 90 seconds."),
        ("14:06", "First HTTP 504 responses observed on POST /v2/payments."),
        ("14:07", "Latency SLO alert fires; incident INC-2291 auto-created at severity SEV-3."),
        ("14:11", "On-call engineer acknowledges and escalates to SEV-1; incident bridge opened."),
        ("14:15", "Incident commander assigned; customer status page updated to 'investigating'."),
        ("14:22", "Aurora writer reaches the 5,000 connection ceiling; new connections rejected."),
        ("14:28", "Error rate peaks at 34% of payment creation requests."),
        ("14:35", "Read replicas confirmed healthy, ruling out a general database failure."),
        ("14:47", "Team correlates the error onset with the CHG-8841 deployment window."),
        ("14:55", "Connection exhaustion on the writer confirmed as the direct cause."),
        ("15:02", "Decision taken to roll back CHG-8841 rather than scale the database."),
        ("15:12", "Rollback begins; status page updated to 'identified'."),
        ("15:40", "Rollback completes across all instances; error rate begins to decline."),
        ("16:05", "Error rate plateaus at 11%; stale pooled connections identified as the cause."),
        ("16:20", "Decision taken to perform a rolling restart of the payment service."),
        ("16:30", "Rolling restart begins at 10% of instances with a 5-minute soak."),
        ("17:15", "Rolling restart completes across all 40 instances."),
        ("17:28", "Error rate returns to the pre-incident baseline of 0.02%."),
        ("17:42", "Incident declared resolved; status page updated to 'resolved'."),
        ("18:30", "Customer communications sent to 412 affected merchant accounts."),
    ]
    entries = [f"{time} UTC — {event}" for time, event in log]

    impact = [
        f"{metric}: {value}"
        for metric, value in (
            ("Failed payment attempts", "128,400 across 412 merchant accounts."),
            ("Peak error rate", "34% of POST /v2/payments requests."),
            ("Customer-impacting duration", "3 hours and 35 minutes, from 14:07 to 17:42 UTC."),
            ("Error budget consumed", "62% of the 28-day availability error budget for payments-api."),
            ("Data loss", "None. No payment was double-charged and no record was corrupted."),
            ("Security impact", "None. No unauthorised access occurred and no data was exfiltrated."),
            ("Financial impact", "Estimated $184,000 in delayed settlement volume, fully recovered within 24 hours."),
            ("Regulatory reporting", "Not required; the incident did not involve personal data."),
        )
    ]
    return [
        ("6. Appendix A — Detailed Event Log", _paragraphize(entries, 4)),
        (
            "7. Appendix B — Impact Assessment",
            ["Figures below were confirmed by the payments data team on 14 August 2025."]
            + _paragraphize(impact, 2),
        ),
    ]


def _appendix_contract() -> list[Section]:
    deliverables = [
        ("D-01", "Production tenant provisioning", "within 15 business days of the effective date"),
        ("D-02", "Single sign-on integration via SAML 2.0", "within 30 business days"),
        ("D-03", "Historical data migration of up to 5 TB", "within 60 business days"),
        ("D-04", "Administrator training for up to 25 named users", "within 45 business days"),
        ("D-05", "Disaster recovery test report", "annually, by 31 January"),
        ("D-06", "SOC 2 Type II report delivery", "annually, within 30 days of issue"),
        ("D-07", "Quarterly business review", "within 20 business days of each quarter end"),
        ("D-08", "Penetration test summary letter", "annually, by 30 June"),
        ("D-09", "Roadmap briefing", "semi-annually"),
        ("D-10", "Uptime and service credit statement", "monthly, within 10 business days of month end"),
    ]
    entries = [f"Deliverable {did}: {desc}, due {due}." for did, desc, due in deliverables]

    rates = [
        f"{role}: ${rate} per hour, minimum engagement {minimum}."
        for role, rate, minimum in (
            ("Solution Architect", 285, "40 hours"),
            ("Senior Implementation Consultant", 240, "40 hours"),
            ("Implementation Consultant", 195, "20 hours"),
            ("Data Migration Engineer", 225, "40 hours"),
            ("Technical Account Manager", 210, "monthly retainer of 20 hours"),
            ("Training Specialist", 165, "8 hours"),
            ("Project Manager", 200, "20 hours"),
        )
    ]
    return [
        (
            "7. Appendix A — Schedule of Deliverables",
            ["Delivery dates run from the effective date of 1 April 2025 unless stated otherwise."]
            + _paragraphize(entries, 3),
        ),
        (
            "8. Appendix B — Professional Services Rate Card",
            [
                "Rates below apply to work outside the pre-purchased pool of 4,000 hours and are "
                "fixed for the initial term."
            ]
            + _paragraphize(rates, 2),
        ),
    ]


def _appendix_mandate() -> list[Section]:
    markets = [
        ("United States", "developed", 62.0, "no restriction"),
        ("Japan", "developed", 6.5, "no restriction"),
        ("United Kingdom", "developed", 4.0, "no restriction"),
        ("Canada", "developed", 3.2, "no restriction"),
        ("France", "developed", 3.0, "no restriction"),
        ("Germany", "developed", 2.4, "no restriction"),
        ("Switzerland", "developed", 2.3, "no restriction"),
        ("Australia", "developed", 1.8, "no restriction"),
        ("Netherlands", "developed", 1.3, "no restriction"),
        ("China", "emerging", 3.1, "H-shares and ADRs only; A-shares require prior client consent"),
        ("India", "emerging", 2.2, "no restriction"),
        ("Taiwan", "emerging", 1.9, "no restriction"),
        ("South Korea", "emerging", 1.4, "no restriction"),
        ("Brazil", "emerging", 0.7, "no restriction"),
        ("Saudi Arabia", "emerging", 0.5, "prior client consent required"),
        ("Mexico", "emerging", 0.4, "no restriction"),
        ("South Africa", "emerging", 0.3, "no restriction"),
        ("Turkey", "emerging", 0.1, "prohibited without Investment Risk Committee approval"),
    ]
    entries = [
        f"{country} ({classification}, benchmark weight {weight}%): {note}."
        for country, classification, weight, note in markets
    ]

    breaches = [
        f"Breach type {code}: {desc} Remediation window: {window}."
        for code, desc, window in (
            ("BR-01", "Single issuer exceeds 5% of net asset value at purchase.", "same business day"),
            ("BR-02", "Single issuer exceeds 7% of net asset value passively.", "three business days"),
            ("BR-03", "Sector deviation exceeds plus or minus 8 percentage points.", "five business days"),
            ("BR-04", "Country deviation exceeds plus or minus 10 percentage points.", "five business days"),
            ("BR-05", "Emerging market exposure exceeds 30% of net asset value.", "five business days"),
            ("BR-06", "Ex-ante tracking error exceeds the 7% upper bound.", "ten business days"),
            ("BR-07", "Cash exceeds 10% of net asset value outside settlement periods.", "three business days"),
            ("BR-08", "Position count falls below 45 or exceeds 90.", "ten business days"),
            ("BR-09", "A holding breaches the ESG exclusion screen.", "three business days"),
            ("BR-10", "Derivative notional exposure exceeds 20% of net asset value.", "same business day"),
        )
    ]
    return [
        (
            "7. Appendix A — Permitted Markets",
            ["Benchmark weights are as of 30 September 2025 and are restated quarterly."]
            + _paragraphize(entries, 3),
        ),
        (
            "8. Appendix B — Breach Classification and Remediation",
            ["All breaches are reported to the Investment Risk Committee regardless of remediation speed."]
            + _paragraphize(breaches, 2),
        ),
    ]


def _appendix_runbook() -> list[Section]:
    apps = [
        ("APP-001", "Email and calendar", "baseline", "automatic"),
        ("APP-002", "Corporate chat workspace", "baseline", "automatic"),
        ("APP-003", "HR self-service portal", "baseline", "automatic"),
        ("APP-004", "Expense management system", "baseline", "automatic"),
        ("APP-005", "Documentation wiki (read)", "baseline", "automatic"),
        ("APP-006", "Documentation wiki (write)", "standard", "manager approval"),
        ("APP-007", "Source control", "engineering", "manager approval"),
        ("APP-008", "CI/CD pipelines", "engineering", "manager approval"),
        ("APP-009", "Non-production AWS accounts", "engineering", "manager approval"),
        ("APP-010", "Production AWS accounts (read-only)", "restricted", "director approval"),
        ("APP-011", "Production AWS accounts (write)", "restricted", "director approval and security review"),
        ("APP-012", "Customer data warehouse", "restricted", "data owner approval"),
        ("APP-013", "Payments admin console", "restricted", "director approval and security review"),
        ("APP-014", "Financial ledger system", "restricted", "finance controller approval"),
        ("APP-015", "Procurement system", "standard", "manager approval"),
        ("APP-016", "CRM", "standard", "manager approval"),
        ("APP-017", "Support ticketing", "standard", "manager approval"),
        ("APP-018", "Business intelligence dashboards", "standard", "manager approval"),
        ("APP-019", "Secrets manager", "restricted", "security engineering approval"),
        ("APP-020", "Identity provider administration", "restricted", "CISO approval"),
    ]
    entries = [
        f"{aid} {name} — bundle: {bundle}; provisioning: {approval}."
        for aid, name, bundle, approval in apps
    ]
    return [
        (
            "5. Appendix A — Application Access Matrix",
            [
                "Restricted applications additionally require a completed access request "
                "ticket and are re-certified at each quarterly review."
            ]
            + _paragraphize(entries, 4),
        )
    ]


def _glossary_section(number: int, intro: str, terms: list[tuple[str, str]]) -> Section:
    entries = [f"{term}: {definition}" for term, definition in terms]
    return (f"{number}. Appendix — Glossary of Terms", [intro] + _paragraphize(entries, 3))


APPENDICES = {
    "Q3_2025_Consolidated_Financial_Statements": _appendix_financial,
    "Employee_Handbook_2025": _appendix_hr,
    "AWS_Cloud_Architecture_Guide": _appendix_aws,
    "Information_Security_Policy": _appendix_security,
    "Payments_API_Reference_v2": _appendix_api,
    "Incident_Postmortem_INC-2291": _appendix_incident,
    "Vendor_MSA_Contoso_Analytics": _appendix_contract,
    "Global_Equity_Growth_Fund_Investment_Mandate": _appendix_mandate,
    "IT_Onboarding_and_Access_Runbook": _appendix_runbook,
}

GLOSSARIES: dict[str, tuple[int, str, list[tuple[str, str]]]] = {
    "Q3_2025_Consolidated_Financial_Statements": (
        9,
        "Definitions used throughout this statement and in the earnings release.",
        [
            ("Annual recurring revenue", "The annualised value of subscription contracts in effect on the last day of the period."),
            ("Net revenue retention", "Revenue from the prior-year cohort in the current period divided by that cohort's prior-year revenue."),
            ("Remaining performance obligation", "Contracted revenue not yet recognised, including both billed and unbilled amounts."),
            ("Free cash flow", "Net cash provided by operating activities less purchases of property and equipment."),
            ("Constant currency", "Growth recalculated using prior-period foreign exchange rates to remove currency movement."),
            ("Non-GAAP operating margin", "Income from operations excluding stock-based compensation and acquisition-related items, divided by revenue."),
            ("Basis point", "One hundredth of one percentage point."),
            ("Deferred revenue", "Amounts invoiced to customers in advance of revenue recognition."),
            ("Consolidated leverage ratio", "Total funded debt divided by trailing twelve-month consolidated EBITDA."),
            ("Effective tax rate", "Provision for income taxes divided by income before income taxes."),
            ("Segment operating margin", "Segment income from operations divided by segment revenue, excluding unallocated corporate costs."),
            ("Diluted share count", "Weighted average shares outstanding including the dilutive effect of equity awards."),
        ],
    ),
    "Employee_Handbook_2025": (
        8,
        "Terms used in this handbook have the meanings given below.",
        [
            ("Core collaboration hours", "The window of 11:00 to 15:00 local time during which employees must be reachable."),
            ("Primary caregiver", "The employee who has the main day-to-day responsibility for the care of a child."),
            ("Secondary caregiver", "An employee who shares caring responsibility but is not the primary caregiver."),
            ("Continuous service", "Unbroken employment with the company, including approved periods of paid and unpaid leave."),
            ("People Business Partner", "The assigned HR contact responsible for an organisational unit."),
            ("Phased return", "A temporary reduction to 80% working hours on full pay following extended leave."),
            ("Hybrid working model", "The arrangement requiring office-assigned employees on site a minimum of eight days per month."),
            ("Carry-over", "Unused annual leave transferred into the following calendar year, capped at 10 days."),
            ("Company performance multiplier", "The company-wide factor applied to individual bonus targets at payout."),
            ("Forced distribution", "A ratings quota imposed across a team; the company does not use one."),
            ("Ethics hotline", "The independently operated channel for raising conduct concerns, including anonymously."),
            ("Statutory holiday", "A public holiday required by the law of the employee's country of employment."),
        ],
    ),
    "AWS_Cloud_Architecture_Guide": (
        9,
        "Terminology used in this guide and in the associated runbooks.",
        [
            ("Landing zone", "The standardised multi-account AWS environment provisioned by AWS Control Tower."),
            ("Organisational unit", "A grouping of AWS accounts to which service control policies are applied."),
            ("Service control policy", "An organisation-level guardrail that sets the maximum available permissions in an account."),
            ("Topology spread constraint", "A Kubernetes scheduling rule distributing pods evenly across failure domains."),
            ("Pod Disruption Budget", "The minimum number of pods that must remain available during voluntary disruption."),
            ("Karpenter", "The just-in-time node provisioner used in place of static autoscaling groups."),
            ("Recovery time objective", "The maximum acceptable time to restore service after an outage."),
            ("Recovery point objective", "The maximum acceptable amount of data loss measured in time."),
            ("Service level objective", "The internal reliability target a service commits to, measured over a rolling window."),
            ("Error budget", "The permitted amount of unreliability implied by an SLO over the measurement window."),
            ("Head-based sampling", "A tracing strategy that decides whether to record a trace at request start."),
            ("Inspection VPC", "The shared virtual private cloud through which all egress traffic is routed and filtered."),
            ("Tier-1 service", "A service whose failure directly prevents customers from transacting."),
            ("Savings plan", "A committed-spend discount applied to a baseline level of compute usage."),
        ],
    ),
    "Information_Security_Policy": (
        8,
        "Defined terms used in this policy and in supporting standards.",
        [
            ("Restricted data", "The highest classification, covering credentials, customer financial records and GDPR Article 9 data."),
            ("Confidential data", "Non-public information whose disclosure would harm the company or its customers."),
            ("Least privilege", "Granting only the access necessary to perform an assigned task, for only as long as required."),
            ("Just-in-time access", "Temporary privilege elevation granted for a bounded period against a ticket."),
            ("Phishing-resistant authentication", "An authentication factor bound to the origin, such as a FIDO2 hardware key."),
            ("Access review campaign", "The periodic attestation in which an asset owner confirms who should retain access."),
            ("Admission controller", "The Kubernetes component that blocks non-compliant workloads from being scheduled."),
            ("Dual authorisation", "A control requiring two distinct approvers before an action takes effect."),
            ("Supervisory authority", "The data protection regulator competent for the company's main establishment."),
            ("Data processing addendum", "The contractual annex governing a processor's handling of personal data."),
            ("Standard contractual clauses", "The European Commission approved terms permitting personal data transfer outside the EEA."),
            ("SEV-1", "An incident causing severe, widespread customer impact or a confirmed data breach."),
        ],
    ),
    "Payments_API_Reference_v2": (
        9,
        "Terms used in this reference and in the client SDKs.",
        [
            ("Idempotency key", "A client-supplied string that makes a retried create request safe, retained for 24 hours."),
            ("Minor unit", "The smallest denomination of a currency, such as cents for USD."),
            ("Payment intent", "The object tracking a payment from creation through to a terminal status."),
            ("Terminal status", "A status from which a payment cannot transition further: succeeded, failed or cancelled."),
            ("Capture", "The step that moves an authorised payment to settlement."),
            ("Client credentials grant", "The OAuth 2.0 flow used for server-to-server authentication without a user."),
            ("Cursor pagination", "Paging by an opaque cursor token rather than by numeric offset."),
            ("Dead letter queue", "Where webhook events are placed after the final failed delivery attempt."),
            ("Replay attack", "Resubmission of a captured request, mitigated by the 300-second signature timestamp window."),
            ("Rate limit window", "The rolling one-minute period over which request quota is measured."),
            ("Dispute", "A cardholder challenge to a settled payment, requiring evidence before a deadline."),
            ("Statement descriptor", "The text shown on the payer's bank statement, 22 characters maximum."),
        ],
    ),
    "Incident_Postmortem_INC-2291": (
        8,
        "Terminology used in this review and in the incident management process.",
        [
            ("Direct cause", "The immediate technical failure that produced customer impact."),
            ("Contributing cause", "A condition that allowed the direct cause to occur or to persist."),
            ("Incident commander", "The single person accountable for coordinating the response, not for fixing the issue."),
            ("Connection pool", "The set of reusable database connections maintained by each service instance."),
            ("max_connections", "The PostgreSQL parameter setting the hard ceiling on concurrent connections."),
            ("Progressive rollout", "Deploying to a small percentage of instances first and observing before proceeding."),
            ("Soak period", "The observation window after a partial rollout before the rollout continues."),
            ("Rolling restart", "Restarting instances in batches so capacity is preserved throughout."),
            ("Error budget", "The allowance for unreliability implied by the availability SLO."),
            ("Blameless review", "A post-incident analysis focused on systemic causes rather than individual fault."),
            ("Action item", "A tracked remediation with a named owner and a due date."),
            ("Status page", "The public channel used to communicate incident state to customers."),
        ],
    ),
    "Vendor_MSA_Contoso_Analytics": (
        9,
        "Capitalised terms used in this agreement have the meanings set out below.",
        [
            ("Agreement", "This Master Services Agreement together with all schedules and appendices."),
            ("Effective Date", "1 April 2025, being the date from which the initial term runs."),
            ("Initial Term", "The 36-month period ending 31 March 2028."),
            ("Renewal Term", "Each successive 12-month period following the Initial Term."),
            ("Monthly Uptime Percentage", "Total minutes in a month less unavailable minutes, divided by total minutes."),
            ("Scheduled Maintenance", "Maintenance notified at least five business days in advance, capped at four hours monthly."),
            ("Service Credit", "The percentage of monthly fees credited when uptime falls below the commitment."),
            ("Priority 1", "A complete loss of production service or a material security incident."),
            ("Confidential Information", "Non-public information disclosed by either party and marked or reasonably understood as confidential."),
            ("Data Processor", "The party processing personal data on the documented instructions of the other party."),
            ("Material Breach", "A breach that substantially deprives the other party of the benefit of this Agreement."),
            ("Termination for Convenience", "Termination without cause on 120 days' written notice, subject to the fee obligation."),
        ],
    ),
    "Global_Equity_Growth_Fund_Investment_Mandate": (
        9,
        "Definitions applying to this mandate and to the associated reporting.",
        [
            ("Net asset value", "The total value of fund assets less liabilities, calculated at each valuation point."),
            ("Benchmark", "The MSCI All Country World Index, net total return, in US dollars."),
            ("Ex-ante tracking error", "The forecast annualised standard deviation of returns relative to the benchmark."),
            ("Active share", "The proportion of the portfolio that differs from the benchmark by holding weight."),
            ("High water mark", "The highest cumulative performance level on which a performance fee has previously been paid."),
            ("Passive breach", "A limit breach caused by market movement rather than by a transaction."),
            ("Drawdown", "The peak-to-trough decline in value over a specified period."),
            ("Value at risk", "The loss threshold not expected to be exceeded at a stated confidence level and horizon."),
            ("Participation rate", "The assumed share of average daily trading volume used in liquidity modelling."),
            ("Efficient portfolio management", "Transactions intended to reduce risk or cost rather than to take active positions."),
            ("Depositary receipt", "A negotiable instrument representing shares in a foreign company."),
            ("Exclusion screen", "The rules-based filter removing issuers on ESG grounds before portfolio construction."),
        ],
    ),
    "IT_Onboarding_and_Access_Runbook": (
        6,
        "Terms used in this runbook and in the access request workflow.",
        [
            ("Baseline bundle", "The set of applications provisioned automatically to every new employee."),
            ("Standard bundle", "Applications available on manager approval without further review."),
            ("Restricted application", "An application requiring director or security approval and quarterly recertification."),
            ("Identity provider", "The system of record for authentication and account lifecycle."),
            ("Joiner-mover-leaver", "The lifecycle process governing account creation, change and removal."),
            ("Recertification", "Periodic confirmation by an owner that granted access remains appropriate."),
            ("Dormant account", "An account with no successful authentication in 45 consecutive days."),
            ("Litigation hold", "A legal instruction suspending routine deletion of an employee's data."),
            ("Access request ticket", "The auditable record capturing justification and approval for an access grant."),
            ("Termination effective time", "The timestamp from which the two-hour access revocation window is measured."),
        ],
    ),
}


def write_pdf(path: Path, title: str, sections: list[Section]) -> None:
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Title"], fontSize=17, leading=21, spaceAfter=18
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12.5,
        leading=16,
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=10, leading=15, alignment=TA_JUSTIFY, spaceAfter=8
    )
    meta_style = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=8.5, textColor="#666666")

    doc = SimpleDocTemplate(
        str(path),
        pagesize=LETTER,
        title=title,
        author="Northwind Systems Inc.",
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
    )

    flow = [
        Paragraph(title, title_style),
        Paragraph(
            "Northwind Systems Inc. &nbsp;|&nbsp; Internal — Confidential &nbsp;|&nbsp; "
            f"Document owner: {fake.name()} &nbsp;|&nbsp; Last revised: "
            f"{fake.date_between(start_date='-8m', end_date='today').isoformat()}",
            meta_style,
        ),
        Spacer(1, 14),
    ]
    for i, (heading, paragraphs) in enumerate(sections):
        # Force a page break every three sections so PDFs are genuinely
        # multi-page and page-level citations are meaningful.
        if i and i % 3 == 0:
            flow.append(PageBreak())
        flow.append(Paragraph(heading, heading_style))
        flow.extend(Paragraph(p, body_style) for p in paragraphs)

    doc.build(flow)


def write_txt(path: Path, title: str, sections: list[Section]) -> None:
    lines = [title, "=" * len(title), ""]
    for heading, paragraphs in sections:
        lines.append(heading)
        lines.append("")
        lines.extend(f"{p}\n" for p in paragraphs)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic enterprise corpus.")
    parser.add_argument("--out", default="data/synthetic", help="Output directory.")
    parser.add_argument(
        "--format", choices=["pdf", "txt", "both"], default="pdf", help="Output format."
    )
    parser.add_argument("--clean", action="store_true", help="Delete existing files first.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for existing in out_dir.glob("*"):
            if existing.is_file():
                existing.unlink()

    written: list[str] = []
    for build in BUILDERS:
        filename, _doc_type, title, sections = build()
        stem = Path(filename).stem
        if appendix := APPENDICES.get(stem):
            sections = sections + appendix()
        if glossary := GLOSSARIES.get(stem):
            sections = sections + [_glossary_section(*glossary)]
        if args.format in ("pdf", "both"):
            pdf_path = out_dir / f"{stem}.pdf"
            write_pdf(pdf_path, title, sections)
            written.append(pdf_path.name)
        if args.format in ("txt", "both"):
            txt_path = out_dir / f"{stem}.txt"
            write_txt(txt_path, title, sections)
            written.append(txt_path.name)

    print(f"Generated {len(written)} files in {out_dir.resolve()}:")
    for name in written:
        size_kb = (out_dir / name).stat().st_size / 1024
        print(f"  - {name} ({size_kb:.1f} KB)")
    print("\nNext: python scripts/ingest.py --dir", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
