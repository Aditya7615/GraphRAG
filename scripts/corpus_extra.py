"""Additional synthetic enterprise documents.

Kept separate from `generate_synthetic_data.py` purely to stop that file from
turning into a two-thousand-line wall of prose. Each entry here is a complete,
ready-to-render document: body sections, a reference appendix and a glossary.

The same design rules apply as in the base corpus:
  * exact identifiers (BCP-04, RSK-118, TIER-2) give BM25 something to bite on
  * paraphrasable policy prose exercises the dense embeddings
  * documents cross-reference each other, and two of them deliberately conflict
    with the base corpus so the "conflicting sources" behaviour is testable
"""

from __future__ import annotations

Section = tuple[str, list[str]]
Document = tuple[str, str, str, list[Section]]


def _paragraphize(entries: list[str], per_paragraph: int = 3) -> list[str]:
    return [
        " ".join(entries[i : i + per_paragraph]) for i in range(0, len(entries), per_paragraph)
    ]


def _glossary(number: int, intro: str, terms: list[tuple[str, str]]) -> Section:
    return (
        f"{number}. Appendix — Glossary of Terms",
        [intro] + _paragraphize([f"{t}: {d}" for t, d in terms], 3),
    )


# ---------------------------------------------------------------------------
def _data_privacy() -> Document:
    sections: list[Section] = [
        (
            "1. Lawful Basis and Purpose Limitation",
            [
                "Every processing activity must be recorded in the Record of Processing "
                "Activities (RoPA) before processing begins, and must identify a lawful basis "
                "under GDPR Article 6. Where special category data is involved, an additional "
                "Article 9 condition must be documented.",
                "Personal data collected for one purpose may not be reused for an incompatible "
                "purpose. A compatibility assessment must be completed and approved by the Data "
                "Protection Officer before any secondary use, including model training.",
                "Marketing communications rely on consent, which must be freely given, specific, "
                "informed and unambiguous. Pre-ticked boxes and bundled consent are prohibited. "
                "Consent records must be retained for the life of the processing plus three years.",
            ],
        ),
        (
            "2. Data Subject Rights",
            [
                "Data subject access requests must be fulfilled within one calendar month of "
                "receipt. The period may be extended by a further two months for complex or "
                "numerous requests, provided the data subject is informed within the first month.",
                "Requests for erasure are assessed against the retention schedule and any "
                "overriding legal obligation. Where erasure is refused, the reason must be "
                "communicated in writing within the same one-month window.",
                "Data portability applies only to data provided by the data subject and processed "
                "by automated means on the basis of consent or contract. Exports are provided in "
                "JSON format within one calendar month.",
                "Objections to processing based on legitimate interests suspend that processing "
                "until the balancing test has been re-performed and documented.",
            ],
        ),
        (
            "3. Retention Schedule",
            [
                "Customer account records are retained for the duration of the contract plus "
                "seven years. Payment transaction records are retained for seven years to satisfy "
                "financial regulation. Marketing engagement data is retained for 24 months from "
                "the last interaction.",
                "Recruitment records for unsuccessful candidates are retained for 12 months from "
                "the hiring decision. Employee records are retained for six years after the end "
                "of employment. CCTV footage is retained for 31 days.",
                "Application and infrastructure logs containing personal data are retained for "
                "400 days, aligned to the Information Security Policy, and are then irreversibly "
                "deleted rather than archived.",
            ],
        ),
        (
            "4. International Transfers",
            [
                "Transfers of personal data outside the European Economic Area require an "
                "adequacy decision, standard contractual clauses, or binding corporate rules. A "
                "transfer impact assessment must be completed for every new transfer route.",
                "The company operates primary processing in AWS eu-west-1 for EEA personal data. "
                "Transfer to us-east-1 is permitted only for aggregated, pseudonymised telemetry "
                "and requires DPO sign-off recorded in the RoPA.",
            ],
        ),
        (
            "5. Breach Notification",
            [
                "A personal data breach must be reported internally to the DPO within 12 hours of "
                "discovery. The DPO notifies the lead supervisory authority within 72 hours of the "
                "company becoming aware, consistent with the Information Security Policy.",
                "Where a breach is likely to result in a high risk to the rights and freedoms of "
                "data subjects, affected individuals must be notified without undue delay. "
                "Notification may be omitted where the data was encrypted to the approved "
                "standard and the keys were not compromised.",
            ],
        ),
        (
            "6. Privacy by Design",
            [
                "A Data Protection Impact Assessment is mandatory for any processing involving "
                "systematic monitoring, large-scale special category data, or automated "
                "decision-making with legal effect. The DPIA must be completed before development "
                "begins, not before launch.",
                "Production personal data may not be copied into non-production environments. "
                "Test data must be synthetic or irreversibly anonymised, and anonymisation must be "
                "validated against re-identification risk before first use.",
            ],
        ),
        (
            "7. Appendix A — Processing Activity Register",
            ["Extract from the Record of Processing Activities as at 30 September 2025."]
            + _paragraphize(
                [
                    f"{rid} — {activity}. Lawful basis: {basis}. Retention: {retention}."
                    for rid, activity, basis, retention in (
                        ("ROPA-001", "Customer account administration", "contract", "contract plus 7 years"),
                        ("ROPA-002", "Payment processing and settlement", "legal obligation", "7 years"),
                        ("ROPA-003", "Fraud detection and scoring", "legitimate interests", "5 years"),
                        ("ROPA-004", "Know-your-customer verification", "legal obligation", "5 years after relationship ends"),
                        ("ROPA-005", "Marketing email campaigns", "consent", "24 months from last interaction"),
                        ("ROPA-006", "Product analytics and telemetry", "legitimate interests", "13 months"),
                        ("ROPA-007", "Customer support ticketing", "contract", "3 years"),
                        ("ROPA-008", "Recruitment and applicant tracking", "legitimate interests", "12 months"),
                        ("ROPA-009", "Employee HR administration", "contract", "6 years after employment"),
                        ("ROPA-010", "Payroll and benefits", "legal obligation", "7 years"),
                        ("ROPA-011", "CCTV at office premises", "legitimate interests", "31 days"),
                        ("ROPA-012", "Security event logging", "legitimate interests", "400 days"),
                        ("ROPA-013", "Vendor contact management", "contract", "contract plus 3 years"),
                        ("ROPA-014", "Investor relations mailing list", "consent", "until withdrawal"),
                        ("ROPA-015", "Website cookie analytics", "consent", "13 months"),
                    )
                ],
                3,
            ),
        ),
        _glossary(
            8,
            "Terms in this manual carry the meanings given in the GDPR unless stated otherwise.",
            [
                ("Personal data", "Any information relating to an identified or identifiable natural person."),
                ("Special category data", "Data revealing racial origin, political opinions, health, biometrics or sexual orientation."),
                ("Controller", "The party determining the purposes and means of processing."),
                ("Processor", "The party processing personal data on behalf of the controller."),
                ("RoPA", "The Record of Processing Activities required by GDPR Article 30."),
                ("DPIA", "Data Protection Impact Assessment, required for high-risk processing under Article 35."),
                ("Transfer impact assessment", "The analysis of destination-country law required before an international transfer."),
                ("Pseudonymisation", "Processing such that data can no longer be attributed to a person without separate information."),
                ("Anonymisation", "Irreversible processing after which a person is no longer identifiable; anonymised data falls outside the GDPR."),
                ("Lead supervisory authority", "The regulator for the company's main establishment in the EEA."),
                ("Legitimate interests", "A lawful basis requiring a documented three-part balancing test."),
                ("Automated decision-making", "A decision with legal or similarly significant effect taken without meaningful human involvement."),
            ],
        ),
    ]
    return (
        "Data_Privacy_and_GDPR_Compliance_Manual.pdf",
        "compliance",
        "Northwind Systems — Data Privacy and GDPR Compliance Manual (DPM-2025.2)",
        sections,
    )


def _business_continuity() -> Document:
    sections: list[Section] = [
        (
            "1. Scope and Governance",
            [
                "This plan covers all tier-1 and tier-2 services and the business functions that "
                "depend on them. It is owned by the Head of Resilience and approved annually by "
                "the Operating Committee.",
                "Business continuity exercises are conducted quarterly. At least one exercise per "
                "year must be an unannounced failover of a tier-1 service in production.",
            ],
        ),
        (
            "2. Business Impact Analysis",
            [
                "Each business process is assigned a maximum tolerable period of disruption "
                "(MTPD). Payment authorisation has an MTPD of 30 minutes, settlement has an MTPD "
                "of 8 hours, and management reporting has an MTPD of 5 business days.",
                "Recovery objectives are derived from the MTPD and must be strictly shorter. "
                "Tier-1 services carry an RTO of 15 minutes and an RPO of 5 minutes; tier-2 "
                "services carry an RTO of 4 hours and an RPO of 1 hour, consistent with the Cloud "
                "Architecture Guide.",
                "Any service whose measured recovery time exceeds its RTO in two consecutive "
                "exercises must be escalated to the Operating Committee with a remediation plan "
                "within 15 business days.",
            ],
        ),
        (
            "3. Recovery Strategies",
            [
                "Tier-1 services run active-active across three Availability Zones in us-east-1, "
                "with a warm standby in us-west-2 maintained by continuous Aurora global database "
                "replication. Regional failover is a documented, rehearsed runbook procedure, not "
                "an improvisation.",
                "Regional failover is authorised by the incident commander after consultation "
                "with the Head of Resilience. Failover must not be initiated during an active "
                "data-corruption incident, because replication would propagate the corruption.",
                "Backups are tested by restore, not by inspection. A full restore of the "
                "production Aurora cluster into an isolated account is performed monthly, and the "
                "restore time is recorded against the RTO.",
            ],
        ),
        (
            "4. Crisis Management Structure",
            [
                "The crisis management team comprises the Head of Resilience as chair, the "
                "incident commander, a communications lead, a legal representative, and the "
                "service owner for the affected system.",
                "The team convenes within 30 minutes of a SEV-1 declaration that has lasted more "
                "than one hour, or immediately for any event affecting more than one region.",
                "Internal communications are issued every 30 minutes during an active crisis. "
                "External customer communications are issued every 60 minutes and must be "
                "approved by the communications lead and the legal representative.",
            ],
        ),
        (
            "5. Dependency and Supplier Resilience",
            [
                "Critical third-party suppliers must provide evidence of their own continuity "
                "testing annually. Where a supplier has no viable substitute, an exit and "
                "contingency plan must be documented and reviewed every 12 months.",
                "Single points of failure are recorded in the resilience register with a named "
                "owner and a target elimination date. New single points of failure may not be "
                "introduced without Operating Committee approval.",
            ],
        ),
        (
            "6. Appendix A — Recovery Procedures Register",
            ["Each procedure is rehearsed at the stated cadence and version-controlled in the runbook repository."]
            + _paragraphize(
                [
                    f"{pid} — {name}. Target duration: {duration}. Rehearsal cadence: {cadence}."
                    for pid, name, duration, cadence in (
                        ("BCP-01", "Single Availability Zone loss, automatic", "no manual action", "quarterly"),
                        ("BCP-02", "Aurora writer failover within region", "under 120 seconds", "monthly"),
                        ("BCP-03", "Full regional failover to us-west-2", "under 45 minutes", "semi-annual"),
                        ("BCP-04", "Point-in-time restore of the payments database", "under 90 minutes", "monthly"),
                        ("BCP-05", "EKS cluster rebuild from infrastructure code", "under 3 hours", "quarterly"),
                        ("BCP-06", "Certificate authority compromise response", "under 4 hours", "annual"),
                        ("BCP-07", "Loss of the primary identity provider", "under 60 minutes", "semi-annual"),
                        ("BCP-08", "Ransomware containment and clean-room recovery", "under 24 hours", "annual"),
                        ("BCP-09", "Loss of the settlement banking partner", "under 8 hours", "annual"),
                        ("BCP-10", "Office inaccessibility and full remote operation", "under 2 hours", "annual"),
                        ("BCP-11", "Key personnel unavailability", "immediate deputy activation", "semi-annual"),
                        ("BCP-12", "Corrupted deployment rollback across all regions", "under 30 minutes", "quarterly"),
                    )
                ],
                3,
            ),
        ),
        _glossary(
            7,
            "Resilience terminology used across this plan and the associated runbooks.",
            [
                ("MTPD", "Maximum tolerable period of disruption before unacceptable business harm occurs."),
                ("RTO", "Recovery time objective: the target elapsed time to restore a service."),
                ("RPO", "Recovery point objective: the maximum tolerable data loss measured in time."),
                ("Active-active", "An architecture in which all sites serve traffic simultaneously."),
                ("Warm standby", "A scaled-down but running environment that can absorb traffic after promotion."),
                ("Clean-room recovery", "Rebuilding into a known-good isolated environment after a destructive compromise."),
                ("Failback", "The controlled return of traffic to the primary region after recovery."),
                ("Resilience register", "The inventory of single points of failure and their remediation owners."),
                ("Crisis management team", "The cross-functional group accountable for decisions during a major incident."),
                ("Exit plan", "The documented route to migrate away from a critical supplier."),
            ],
        ),
    ]
    return (
        "Business_Continuity_and_Disaster_Recovery_Plan.pdf",
        "operations",
        "Northwind Systems — Business Continuity and Disaster Recovery Plan (BCP-2025)",
        sections,
    )


def _sales_compensation() -> Document:
    sections: list[Section] = [
        (
            "1. Plan Structure",
            [
                "On-target earnings (OTE) for an Account Executive are split 50% base salary and "
                "50% variable commission. For Solutions Engineers the split is 75% base and 25% "
                "variable. For Customer Success Managers the split is 80% base and 20% variable.",
                "The plan year runs from 1 February 2026 to 31 January 2027. Quotas are assigned "
                "in writing within 30 days of the plan year start, or within 30 days of hire for "
                "new joiners.",
                "Commission is earned on booked annual contract value (ACV) recognised at contract "
                "signature, and is paid in the month following the close of the quarter in which "
                "the deal was booked.",
            ],
        ),
        (
            "2. Accelerators and Caps",
            [
                "Attainment between 0% and 100% of quota is paid at the base commission rate of "
                "8% of ACV. Attainment between 100% and 150% is paid at an accelerated rate of "
                "12%. Attainment above 150% is paid at 16%.",
                "There is no cap on commission earnings. However, any single transaction with an "
                "ACV above $2,000,000 requires the commission treatment to be pre-approved in "
                "writing by the Chief Revenue Officer and the Chief Financial Officer.",
                "A decelerator applies below 40% attainment: commission on the portion below 40% "
                "is paid at 4% rather than 8%.",
            ],
        ),
        (
            "3. Multi-Year and Multi-Product Deals",
            [
                "Multi-year contracts are credited at the first-year ACV, plus 25% of the ACV of "
                "each subsequent committed year. Uncommitted optional years carry no credit until "
                "exercised.",
                "New logo business carries a 1.25x credit multiplier. Expansion within an existing "
                "account carries a 1.0x multiplier, and renewals at flat value carry a 0.5x "
                "multiplier.",
                "Professional services attached to a software deal are credited at 30% of services "
                "value. Standalone services sales carry no commission credit.",
            ],
        ),
        (
            "4. Clawback and Adjustments",
            [
                "Commission is subject to clawback where a customer cancels within 90 days of "
                "contract signature, or where the first invoice remains unpaid 120 days after its "
                "due date. Clawback is recovered against future commission payments.",
                "Where a deal is credited to more than one representative, the split must be "
                "agreed in writing before the deal closes. Retroactive split disputes raised more "
                "than 30 days after payment will not be adjusted.",
            ],
        ),
        (
            "5. Leaver Provisions",
            [
                "A representative who resigns is paid commission on deals booked before their last "
                "working day, provided the payment date falls within 60 days of departure. "
                "Commission on deals booked after notice is given but not yet closed is forfeited.",
                "A representative terminated without cause is paid all earned commission plus a "
                "pro-rata share of the current quarter at actual attainment. Termination for cause "
                "forfeits all unpaid commission.",
            ],
        ),
        (
            "6. Appendix A — Quota and OTE Bands",
            ["Bands are set annually by Sales Operations and reviewed against market benchmarks each October."]
            + _paragraphize(
                [
                    f"{role} ({level}): OTE ${ote:,}, annual quota ${quota:,}, quota-to-OTE ratio {quota / ote:.1f}x."
                    for role, level, ote, quota in (
                        ("Account Executive", "AE1", 180000, 900000),
                        ("Account Executive", "AE2", 220000, 1200000),
                        ("Account Executive", "AE3", 260000, 1560000),
                        ("Strategic Account Executive", "SAE1", 320000, 2240000),
                        ("Strategic Account Executive", "SAE2", 380000, 2850000),
                        ("Solutions Engineer", "SE1", 165000, 900000),
                        ("Solutions Engineer", "SE2", 195000, 1200000),
                        ("Customer Success Manager", "CSM1", 140000, 2000000),
                        ("Customer Success Manager", "CSM2", 170000, 3000000),
                        ("Sales Manager", "M1", 300000, 5400000),
                        ("Sales Director", "D1", 380000, 9500000),
                        ("Regional Vice President", "VP1", 460000, 18400000),
                    )
                ],
                3,
            ),
        ),
        _glossary(
            7,
            "Definitions used in this plan and in the commission statements.",
            [
                ("OTE", "On-target earnings: base salary plus variable pay at 100% quota attainment."),
                ("ACV", "Annual contract value: the committed subscription value for a twelve-month period."),
                ("Booking", "A signed, countersigned contract recorded in the order management system."),
                ("Attainment", "Credited bookings divided by assigned quota, expressed as a percentage."),
                ("Accelerator", "An increased commission rate applied to attainment above 100% of quota."),
                ("Decelerator", "A reduced commission rate applied to attainment below 40% of quota."),
                ("New logo", "A customer with no prior paid contract, carrying a 1.25x credit multiplier."),
                ("Clawback", "Recovery of paid commission following early cancellation or non-payment."),
                ("Split credit", "The agreed allocation of a single deal's credit between representatives."),
                ("Ramp", "The reduced quota applied to a new hire during their first two quarters."),
            ],
        ),
    ]
    return (
        "Sales_Compensation_Plan_FY2026.pdf",
        "hr",
        "Northwind Systems — Sales Compensation Plan FY2026",
        sections,
    )


def _procurement() -> Document:
    sections: list[Section] = [
        (
            "1. Purchase Approval Thresholds",
            [
                "Purchases below $10,000 per year may be approved by a Manager. Purchases between "
                "$10,000 and $50,000 require Director approval. Purchases between $50,000 and "
                "$250,000 require Vice President approval and a competitive quotation from at "
                "least three suppliers.",
                "Purchases above $250,000 require Chief Financial Officer approval and a formal "
                "request for proposal. Purchases above $1,000,000 additionally require Operating "
                "Committee approval.",
                "Splitting a purchase to avoid an approval threshold is a disciplinary matter. "
                "Related purchases from the same supplier within a rolling 12-month period are "
                "aggregated for threshold purposes.",
            ],
        ),
        (
            "2. Supplier Onboarding",
            [
                "No purchase order may be raised against a supplier that has not completed "
                "onboarding. Onboarding comprises sanctions screening, beneficial ownership "
                "verification, financial health assessment, and a security assessment where the "
                "supplier will access company data.",
                "Suppliers accessing Confidential or Restricted data must hold a current SOC 2 "
                "Type II report or ISO/IEC 27001 certification, consistent with the Information "
                "Security Policy. Evidence must be dated within the preceding 12 months.",
            ],
        ),
        (
            "3. Third-Party Risk Tiering",
            [
                "Suppliers are tiered from TIER-1 (critical) to TIER-4 (low). Tiering considers "
                "data classification accessed, business criticality, substitutability, and annual "
                "spend. Tiering is reassessed annually or on material change.",
                "TIER-1 suppliers require an annual on-site or virtual audit, a documented exit "
                "plan, and continuity evidence. TIER-2 suppliers require an annual questionnaire "
                "and evidence review. TIER-3 and TIER-4 suppliers require attestation only.",
                "Concentration risk is monitored quarterly. No single supplier may account for "
                "more than 15% of total third-party spend without Operating Committee "
                "acknowledgement.",
            ],
        ),
        (
            "4. Contracting Standards",
            [
                "All agreements must be executed on company paper or on supplier paper amended by "
                "the company's standard rider. Contracts involving personal data require a data "
                "processing addendum before any data is shared.",
                "Auto-renewal clauses longer than 12 months are prohibited. Every contract must "
                "record its notice period in the contract register at signature, and the register "
                "generates a renewal alert 120 days before the notice deadline.",
                "Liability caps below 12 months of fees require Legal approval. Uncapped liability "
                "for the company is prohibited without General Counsel approval.",
            ],
        ),
        (
            "5. Payment and Invoicing",
            [
                "Standard payment terms are 45 days from a valid invoice date, matching the terms "
                "the company extends to its own customers. Terms shorter than 30 days require "
                "Treasury approval.",
                "Three-way matching between purchase order, goods receipt and invoice is mandatory "
                "for all purchases above $10,000. Invoices without a valid purchase order number "
                "are returned unpaid.",
            ],
        ),
        (
            "6. Appendix A — Supplier Risk Register Extract",
            ["Extract as at 30 September 2025. Full register is maintained in the procurement system."]
            + _paragraphize(
                [
                    f"{sid} {supplier} — tier {tier}, category {category}, annual spend ${spend:,}, "
                    f"next review {review}."
                    for sid, supplier, tier, category, spend, review in (
                        ("SUP-1001", "Contoso Analytics Ltd.", "TIER-1", "data platform", 1850000, "March 2026"),
                        ("SUP-1002", "Fabrikam Cloud Services", "TIER-1", "infrastructure", 6420000, "January 2026"),
                        ("SUP-1003", "Litware Identity", "TIER-1", "identity and access", 740000, "February 2026"),
                        ("SUP-1004", "Adventure Works Payments", "TIER-1", "banking partner", 2100000, "April 2026"),
                        ("SUP-1005", "Proseware Monitoring", "TIER-2", "observability", 415000, "May 2026"),
                        ("SUP-1006", "Tailspin Security", "TIER-2", "penetration testing", 180000, "June 2026"),
                        ("SUP-1007", "Wingtip Legal Services", "TIER-3", "professional services", 95000, "August 2026"),
                        ("SUP-1008", "Coho Recruitment", "TIER-3", "talent acquisition", 260000, "July 2026"),
                        ("SUP-1009", "Lucerne Facilities", "TIER-4", "facilities", 88000, "October 2026"),
                        ("SUP-1010", "Trey Research Data", "TIER-2", "market data", 520000, "September 2026"),
                        ("SUP-1011", "Woodgrove Translation", "TIER-4", "localisation", 41000, "November 2026"),
                        ("SUP-1012", "Alpine Ski House Travel", "TIER-4", "corporate travel", 310000, "December 2026"),
                    )
                ],
                2,
            ),
        ),
        _glossary(
            7,
            "Procurement terminology used in this standard and in the contract register.",
            [
                ("Purchase order", "The authorised commitment document raised before goods or services are received."),
                ("Three-way match", "Reconciliation of purchase order, goods receipt and supplier invoice before payment."),
                ("Beneficial ownership", "The natural persons who ultimately own or control a supplier entity."),
                ("Sanctions screening", "Checking a counterparty against applicable government restricted-party lists."),
                ("Concentration risk", "Excessive dependence on a single supplier or a small group of suppliers."),
                ("Exit plan", "The documented route to migrate away from a supplier without unacceptable disruption."),
                ("Standard rider", "The company's pre-approved amendment applied to supplier paper."),
                ("Auto-renewal", "A contract term that extends automatically absent notice, capped at 12 months."),
                ("Contract register", "The system of record for executed agreements, terms and notice deadlines."),
                ("Tiering", "Classification of suppliers from TIER-1 to TIER-4 by criticality and data access."),
            ],
        ),
    ]
    return (
        "Procurement_and_Third_Party_Risk_Standard.pdf",
        "legal",
        "Northwind Systems — Procurement and Third-Party Risk Standard (PRC-2025.1)",
        sections,
    )


def _support_playbook() -> Document:
    sections: list[Section] = [
        (
            "1. Severity Definitions",
            [
                "Priority 1 (P1) means production is completely unavailable, or a security "
                "incident is suspected, with no workaround. Priority 2 (P2) means major "
                "functionality is degraded with no acceptable workaround. Priority 3 (P3) means "
                "minor functionality is impaired or a workaround exists. Priority 4 (P4) covers "
                "questions and feature requests.",
                "The customer proposes the priority; the support engineer may adjust it with a "
                "written justification in the ticket. Disputed priorities are escalated to the "
                "Support Duty Manager, whose decision is final.",
            ],
        ),
        (
            "2. Response and Resolution Targets",
            [
                "First response targets are 15 minutes for P1, 1 hour for P2, 4 business hours for "
                "P3, and 1 business day for P4. These targets apply to Enterprise-tier customers.",
                "Business-tier customers receive first response targets of 1 hour for P1, 4 hours "
                "for P2, 1 business day for P3, and 3 business days for P4. Standard-tier "
                "customers receive best-effort response during business hours only.",
                "Resolution targets are 4 hours for P1 and 2 business days for P2. Where a defect "
                "requires an engineering fix, the target applies to delivery of a workaround "
                "rather than to the permanent fix.",
                "Note that these support response targets are distinct from the availability "
                "commitments in customer contracts; service credits are governed solely by the "
                "contractual uptime terms.",
            ],
        ),
        (
            "3. Escalation Path",
            [
                "A P1 ticket unresolved after 60 minutes escalates automatically to the Support "
                "Duty Manager. Unresolved after 2 hours, it escalates to the Director of Support "
                "and the owning engineering manager.",
                "Unresolved after 4 hours, a P1 escalates to the Vice President of Customer "
                "Experience, and a formal incident is declared under the incident management "
                "process if customer impact is systemic rather than account-specific.",
                "Customers may request a management escalation at any time through their account "
                "team. Executive escalations are acknowledged within 2 hours by a Director or "
                "above.",
            ],
        ),
        (
            "4. Handover and Follow-the-Sun",
            [
                "Support operates three regional hubs providing 24-hour coverage on weekdays: "
                "Sydney, Dublin and Austin. Weekend coverage for P1 and P2 is provided by an "
                "on-call rotation in each hub.",
                "Every open P1 and P2 ticket must be verbally handed over at shift change, and the "
                "handover summary recorded in the ticket. A ticket may not be left unassigned "
                "across a hub transition.",
            ],
        ),
        (
            "5. Quality and Measurement",
            [
                "Customer satisfaction is surveyed on ticket closure. The target CSAT score is 4.5 "
                "or above on a 5-point scale, measured monthly. The target first-contact "
                "resolution rate is 65% for P3 and P4 tickets.",
                "A sample of 5% of closed tickets is reviewed monthly for quality against a "
                "published rubric covering accuracy, tone, and adherence to this playbook.",
            ],
        ),
        (
            "6. Appendix A — Common Issue Runbook Index",
            ["Each entry links to a full runbook in the support knowledge base."]
            + _paragraphize(
                [
                    f"{rid} — {symptom}. Typical cause: {cause}. First action: {action}."
                    for rid, symptom, cause, action in (
                        ("SUP-R01", "Payments returning HTTP 401", "expired or rotated access token", "confirm token issuance time and refresh"),
                        ("SUP-R02", "Payments returning HTTP 429", "client exceeded the 300 per minute write quota", "inspect X-RateLimit headers and advise backoff"),
                        ("SUP-R03", "Webhook deliveries not received", "endpoint returning non-2xx or timing out", "check delivery attempts and dead letter queue"),
                        ("SUP-R04", "Webhook signature verification failing", "clock skew beyond the 300 second window", "verify customer server time synchronisation"),
                        ("SUP-R05", "Duplicate payments created", "idempotency key omitted on retry", "confirm key usage within the 24 hour retention window"),
                        ("SUP-R06", "Payment stuck in processing", "downstream banking partner delay", "confirm against the 72 hour auto-cancel window"),
                        ("SUP-R07", "Single sign-on login loop", "SAML assertion clock skew or changed certificate", "compare identity provider certificate fingerprint"),
                        ("SUP-R08", "Report export timing out", "export range exceeding recommended size", "advise splitting the range by month"),
                        ("SUP-R09", "Dashboard showing stale figures", "analytics pipeline lag", "check pipeline watermark before escalating"),
                        ("SUP-R10", "Account showing as frozen", "compliance review in progress", "route to the compliance queue, do not advise a timeline"),
                        ("SUP-R11", "Settlement report missing a day", "report generated before cut-off", "regenerate after the 02:00 UTC cut-off"),
                        ("SUP-R12", "API version behaving unexpectedly", "key pinned to a deprecated version", "confirm the Northwind-Version header in use"),
                    )
                ],
                2,
            ),
        ),
        _glossary(
            7,
            "Support terminology used in this playbook and in customer-facing communications.",
            [
                ("First response", "The first substantive human reply to a ticket, excluding automated acknowledgement."),
                ("First-contact resolution", "A ticket resolved in the first reply without further customer exchange."),
                ("Workaround", "A temporary means of restoring function while a permanent fix is developed."),
                ("Duty Manager", "The rotating role accountable for escalations during a shift."),
                ("Follow-the-sun", "Continuous coverage achieved by handing tickets between regional hubs."),
                ("CSAT", "Customer satisfaction score collected on ticket closure, on a 5-point scale."),
                ("Executive escalation", "A customer-requested escalation acknowledged by a Director or above within 2 hours."),
                ("Business hours", "09:00 to 17:30 local time in the customer's assigned support hub, Monday to Friday."),
                ("Service credit", "A contractual remedy for missed uptime, governed by the customer agreement, not by this playbook."),
                ("Known error", "A defect with a documented cause and workaround pending permanent fix."),
            ],
        ),
    ]
    return (
        "Customer_Support_Playbook_and_SLA.pdf",
        "operations",
        "Northwind Systems — Customer Support Playbook and Service Targets",
        sections,
    )


def _model_risk() -> Document:
    sections: list[Section] = [
        (
            "1. Scope and Model Definition",
            [
                "A model is any quantitative method that applies statistical, economic, financial "
                "or machine-learning techniques to input data to produce an output used in a "
                "business decision. Deterministic rule engines are in scope where the rules "
                "materially affect customer outcomes.",
                "All models must be recorded in the model inventory before deployment. Models "
                "deployed without registration are treated as an audit finding and must be "
                "withdrawn within 5 business days.",
            ],
        ),
        (
            "2. Model Tiering",
            [
                "Models are tiered MR-1 to MR-3 by materiality. MR-1 covers models with direct "
                "customer or regulatory impact, such as credit decisioning and fraud scoring. "
                "MR-2 covers models influencing material internal decisions. MR-3 covers "
                "exploratory and internal-productivity models.",
                "MR-1 models require independent validation before deployment and annually "
                "thereafter. MR-2 models require independent validation before deployment and "
                "every two years. MR-3 models require documented peer review only.",
            ],
        ),
        (
            "3. Development Standards",
            [
                "Training data lineage must be fully documented, including source systems, "
                "extraction dates, and any exclusions applied. Personal data used for training "
                "requires a documented lawful basis and a completed compatibility assessment.",
                "Every model must have a documented performance baseline and a champion-challenger "
                "comparison against the incumbent approach. A model may not be promoted on "
                "aggregate accuracy alone; subgroup performance must be reported.",
                "Fairness testing is mandatory for all MR-1 models. Disparate impact is assessed "
                "across protected characteristics, and any adverse impact ratio below 0.80 must be "
                "escalated to the Model Risk Committee before deployment.",
            ],
        ),
        (
            "4. Validation and Approval",
            [
                "Independent validation is performed by the Model Risk function, which does not "
                "report to the model development function. Validation covers conceptual soundness, "
                "data quality, implementation testing, and outcome analysis.",
                "Validation findings are rated Critical, High, Medium or Low. A model with an open "
                "Critical finding may not be deployed. Open High findings require a remediation "
                "plan approved by the Model Risk Committee with a deadline no later than 90 days.",
            ],
        ),
        (
            "5. Monitoring and Retirement",
            [
                "MR-1 models are monitored continuously for input drift, output drift and "
                "performance decay. A population stability index above 0.25 on any material "
                "feature triggers mandatory investigation within 5 business days.",
                "Model performance is formally reviewed quarterly for MR-1 and annually for MR-2. "
                "A model whose primary performance metric degrades by more than 15% from its "
                "approved baseline must be recalibrated or retired.",
                "Retired models must have their inference endpoints disabled and their artefacts "
                "retained for seven years to support regulatory enquiry.",
            ],
        ),
        (
            "6. Generative AI Specific Controls",
            [
                "Large language model applications must implement retrieval grounding with source "
                "citation, and must not present unsourced generated content as fact in a customer "
                "or regulatory context.",
                "Prompt templates are version-controlled and treated as model code subject to "
                "change control. Model temperature for any decisioning or factual retrieval use "
                "case must be set to 0.0 to ensure reproducibility.",
                "Human review is mandatory before any generated content is communicated externally "
                "on the company's behalf. Fully automated external communication generated by a "
                "language model is prohibited.",
            ],
        ),
        (
            "7. Appendix A — Model Inventory Extract",
            ["Extract as at 30 September 2025. The full inventory is maintained by the Model Risk function."]
            + _paragraphize(
                [
                    f"{mid} {name} — tier {tier}, owner {owner}, last validated {validated}, status {status}."
                    for mid, name, tier, owner, validated, status in (
                        ("MDL-101", "Transaction fraud scoring", "MR-1", "Risk Engineering", "June 2025", "approved"),
                        ("MDL-102", "Credit decisioning for merchant advances", "MR-1", "Credit Risk", "April 2025", "approved"),
                        ("MDL-103", "Anti-money-laundering alert prioritisation", "MR-1", "Financial Crime", "August 2025", "approved"),
                        ("MDL-104", "Customer churn propensity", "MR-2", "Growth Analytics", "February 2025", "approved"),
                        ("MDL-105", "Demand forecasting for capacity planning", "MR-2", "Platform Engineering", "January 2025", "approved"),
                        ("MDL-106", "Support ticket triage classifier", "MR-2", "Customer Experience", "May 2025", "approved"),
                        ("MDL-107", "Document retrieval assistant", "MR-2", "Internal Tools", "September 2025", "approved"),
                        ("MDL-108", "Sales lead scoring", "MR-3", "Revenue Operations", "March 2025", "peer reviewed"),
                        ("MDL-109", "Marketing send-time optimisation", "MR-3", "Growth Marketing", "March 2025", "peer reviewed"),
                        ("MDL-110", "Chargeback likelihood estimator", "MR-1", "Risk Engineering", "July 2025", "conditional approval"),
                        ("MDL-111", "Merchant onboarding risk triage", "MR-1", "Financial Crime", "pending", "in validation"),
                        ("MDL-112", "Infrastructure anomaly detection", "MR-3", "Observability", "April 2025", "peer reviewed"),
                    )
                ],
                2,
            ),
        ),
        _glossary(
            8,
            "Model risk terminology aligned to supervisory guidance on model risk management.",
            [
                ("Model", "A quantitative method producing an output used in a business decision."),
                ("Model inventory", "The authoritative register of all models, their tiers and validation status."),
                ("Independent validation", "Review performed by a function organisationally separate from model development."),
                ("Conceptual soundness", "Assessment of whether the modelling approach is appropriate for its purpose."),
                ("Champion-challenger", "Comparison of a candidate model against the incumbent under identical conditions."),
                ("Population stability index", "A measure of distribution shift between the training and live populations."),
                ("Input drift", "Change in the distribution of model inputs relative to the training data."),
                ("Performance decay", "Deterioration of predictive quality over time as conditions change."),
                ("Adverse impact ratio", "The selection rate of a protected group divided by that of the reference group."),
                ("Overlay", "A documented manual adjustment applied to a model output."),
                ("Retrieval grounding", "Constraining generated output to retrieved source documents with citation."),
                ("Model artefact", "The serialised model, its parameters, and the code required to reproduce it."),
            ],
        ),
    ]
    return (
        "AI_Model_Risk_Management_Framework.pdf",
        "compliance",
        "Northwind Systems — Model Risk Management Framework (MRM-2025.1)",
        sections,
    )


def _credit_risk() -> Document:
    sections: list[Section] = [
        (
            "1. Risk Appetite",
            [
                "The board-approved maximum expected credit loss for the merchant advance "
                "portfolio is 3.5% of average outstanding balance per annum. Breach of this "
                "threshold for two consecutive quarters requires immediate suspension of new "
                "originations pending board review.",
                "Single-counterparty exposure may not exceed 2% of total portfolio value at "
                "origination, or 3% on a passive basis following portfolio contraction. Aggregate "
                "exposure to any single industry sector is capped at 20% of portfolio value.",
            ],
        ),
        (
            "2. Origination Standards",
            [
                "All applicants must pass identity verification, sanctions screening and adverse "
                "media screening before assessment. Applications from entities incorporated less "
                "than 12 months prior are declined automatically.",
                "The minimum internal credit score for automatic approval is 620. Applications "
                "scoring between 560 and 619 are referred for manual underwriting. Applications "
                "below 560 are declined.",
                "The maximum advance is the lower of $2,000,000 or four times average monthly "
                "processing volume over the preceding six months. Advances above $500,000 require "
                "dual underwriter approval.",
            ],
        ),
        (
            "3. Internal Rating Scale",
            [
                "Obligors are assigned an internal rating from R1 (strongest) to R9 (default). "
                "Ratings R1 to R5 are performing, R6 and R7 are watch-list, R8 is impaired, and R9 "
                "is default.",
                "Ratings are refreshed monthly using behavioural data, and immediately on any "
                "trigger event such as a missed payment, a material processing volume decline "
                "exceeding 40%, or an adverse media hit.",
            ],
        ),
        (
            "4. Impairment and Provisioning",
            [
                "Expected credit losses are calculated on a three-stage basis. Stage 1 covers "
                "performing exposures with a 12-month expected loss. Stage 2 covers exposures with "
                "a significant increase in credit risk, provisioned at lifetime expected loss. "
                "Stage 3 covers credit-impaired exposures.",
                "A significant increase in credit risk is presumed where a payment is more than 30 "
                "days past due, or where the internal rating has deteriorated by three or more "
                "notches since origination.",
                "Default is defined as 90 days past due, or earlier where the obligor is assessed "
                "as unlikely to pay in full without recourse to security realisation.",
            ],
        ),
        (
            "5. Collections and Recovery",
            [
                "Automated reminders are issued at 1, 7 and 14 days past due. Accounts 30 days "
                "past due are transferred to the collections team. Accounts 60 days past due are "
                "reviewed for restructuring eligibility.",
                "Restructuring is permitted once per obligor in any 24-month period and must not "
                "be used to avoid recognising an impairment. Restructured exposures remain in "
                "Stage 2 for a minimum probation period of 12 months.",
                "Write-off occurs at 180 days past due unless active recovery proceedings are in "
                "progress. Write-off does not extinguish the debt or the right to pursue recovery.",
            ],
        ),
        (
            "6. Stress Testing",
            [
                "The portfolio is stress tested semi-annually against three scenarios: a baseline, "
                "an adverse scenario assuming a 200 basis point rise in default rates, and a "
                "severe scenario assuming a 500 basis point rise combined with a 25% decline in "
                "merchant processing volumes.",
                "Results are reported to the Board Risk Committee within 30 days of the exercise. "
                "Capital adequacy must remain above the internal minimum under all three "
                "scenarios.",
            ],
        ),
        (
            "7. Appendix A — Internal Rating Scale and Expected Loss",
            ["Probability of default is the through-the-cycle twelve-month estimate, recalibrated annually."]
            + _paragraphize(
                [
                    f"{rating} ({descriptor}): probability of default {pd_}%, loss given default {lgd}%, "
                    f"classification {classification}."
                    for rating, descriptor, pd_, lgd, classification in (
                        ("R1", "exceptional", 0.05, 35, "performing"),
                        ("R2", "strong", 0.12, 35, "performing"),
                        ("R3", "good", 0.35, 38, "performing"),
                        ("R4", "satisfactory", 0.90, 40, "performing"),
                        ("R5", "acceptable", 2.10, 42, "performing"),
                        ("R6", "marginal", 4.80, 45, "watch-list"),
                        ("R7", "weak", 9.50, 48, "watch-list"),
                        ("R8", "impaired", 24.00, 55, "impaired"),
                        ("R9", "default", 100.00, 62, "default"),
                    )
                ],
                2,
            ),
        ),
        _glossary(
            8,
            "Credit terminology used in this policy and in portfolio reporting.",
            [
                ("Expected credit loss", "Probability of default multiplied by loss given default and exposure at default."),
                ("Probability of default", "The likelihood an obligor defaults within a stated horizon."),
                ("Loss given default", "The proportion of exposure not recovered following default."),
                ("Exposure at default", "The outstanding amount expected to be owed at the moment of default."),
                ("Significant increase in credit risk", "The deterioration trigger moving an exposure from Stage 1 to Stage 2."),
                ("Watch-list", "Exposures rated R6 or R7 requiring enhanced monitoring."),
                ("Restructuring", "A concession granted to an obligor in financial difficulty."),
                ("Probation period", "The minimum 12 months a restructured exposure remains in Stage 2."),
                ("Write-off", "Derecognition of an exposure from the balance sheet at 180 days past due."),
                ("Through-the-cycle", "An estimate averaged across economic conditions rather than point-in-time."),
                ("Concentration limit", "The cap on exposure to a single counterparty or sector."),
                ("Origination", "The process of assessing and granting a new advance."),
            ],
        ),
    ]
    return (
        "Credit_Risk_Policy_Merchant_Advances.pdf",
        "financial",
        "Northwind Capital — Credit Risk Policy for Merchant Advances (CRP-2025.4)",
        sections,
    )


def _data_governance() -> Document:
    sections: list[Section] = [
        (
            "1. Ownership and Stewardship",
            [
                "Every dataset must have a named data owner accountable for its classification, "
                "quality and access, and a named data steward responsible for day-to-day "
                "curation. Ownership is assigned to a role, never to an individual.",
                "Datasets without a registered owner are quarantined after 30 days: access is "
                "revoked to all but the steward, and the dataset is excluded from downstream "
                "pipelines until ownership is claimed.",
            ],
        ),
        (
            "2. Catalogue and Metadata Standards",
            [
                "All production datasets must be registered in the data catalogue with a "
                "description, classification, owner, steward, refresh cadence, and upstream "
                "lineage. Registration is enforced at pipeline deployment time.",
                "Column-level descriptions are mandatory for datasets classified Confidential or "
                "Restricted. Columns containing personal data must carry a personal-data tag that "
                "drives automated masking in non-production environments.",
                "Lineage must be captured automatically from pipeline definitions rather than "
                "documented manually. Manually asserted lineage is permitted only for external "
                "sources outside the orchestration platform.",
            ],
        ),
        (
            "3. Data Quality Framework",
            [
                "Six quality dimensions are measured: completeness, accuracy, consistency, "
                "timeliness, validity and uniqueness. Every certified dataset must define at least "
                "one test per applicable dimension.",
                "Quality tests run on every pipeline execution. A failed blocking test halts "
                "publication of the affected dataset; a failed non-blocking test raises an alert "
                "to the steward without halting publication.",
                "Certified datasets must maintain a rolling 30-day test pass rate of at least 99%. "
                "Datasets falling below 95% lose certification until remediated.",
            ],
        ),
        (
            "4. Layered Architecture",
            [
                "The lake is organised into three zones. The raw zone holds immutable source "
                "extracts and is retained for seven years. The curated zone holds conformed, "
                "quality-tested entities and is retained for three years. The consumption zone "
                "holds purpose-built aggregates and is retained for 13 months.",
                "Direct consumption from the raw zone is prohibited for reporting and for machine "
                "learning features. Exceptions require Data Governance Council approval and a "
                "documented migration plan.",
            ],
        ),
        (
            "5. Access and Classification",
            [
                "Access is granted to roles, not to individuals, and is provisioned through the "
                "standard access request workflow. Access to Restricted datasets requires data "
                "owner approval and is recertified quarterly.",
                "Dataset classification inherits the highest classification of any column it "
                "contains. Joining two Internal datasets that yields identifiable individuals "
                "produces a Confidential result, and must be reclassified accordingly.",
            ],
        ),
        (
            "6. Appendix A — Certified Dataset Register",
            ["Extract as at 30 September 2025. Certification is reviewed at each quarterly council meeting."]
            + _paragraphize(
                [
                    f"{did} {name} — zone {zone}, classification {classification}, refresh {refresh}, owner {owner}."
                    for did, name, zone, classification, refresh, owner in (
                        ("DS-2001", "payments_transactions", "curated", "restricted", "streaming", "Payments Engineering"),
                        ("DS-2002", "merchant_accounts", "curated", "confidential", "hourly", "Merchant Platform"),
                        ("DS-2003", "settlement_ledger", "curated", "restricted", "daily 02:00 UTC", "Finance Systems"),
                        ("DS-2004", "customer_support_tickets", "curated", "confidential", "hourly", "Customer Experience"),
                        ("DS-2005", "product_events", "curated", "internal", "streaming", "Growth Analytics"),
                        ("DS-2006", "marketing_engagement", "curated", "confidential", "daily 04:00 UTC", "Growth Marketing"),
                        ("DS-2007", "fraud_features", "curated", "restricted", "streaming", "Risk Engineering"),
                        ("DS-2008", "employee_directory", "curated", "confidential", "daily 06:00 UTC", "People Systems"),
                        ("DS-2009", "revenue_daily_summary", "consumption", "internal", "daily 05:00 UTC", "Finance Systems"),
                        ("DS-2010", "executive_kpi_dashboard", "consumption", "internal", "daily 07:00 UTC", "Strategy"),
                        ("DS-2011", "infrastructure_cost_allocation", "consumption", "internal", "daily 03:00 UTC", "Platform Engineering"),
                        ("DS-2012", "regulatory_transaction_report", "consumption", "restricted", "monthly", "Compliance"),
                    )
                ],
                2,
            ),
        ),
        _glossary(
            7,
            "Data governance terminology used across the platform and catalogue.",
            [
                ("Data owner", "The accountable role deciding classification, access and retention for a dataset."),
                ("Data steward", "The role responsible for day-to-day curation and quality of a dataset."),
                ("Raw zone", "Immutable source extracts, retained seven years, not for direct consumption."),
                ("Curated zone", "Conformed, quality-tested entities retained for three years."),
                ("Consumption zone", "Purpose-built aggregates serving reporting, retained for 13 months."),
                ("Lineage", "The recorded path of data from source through transformation to consumption."),
                ("Certification", "The status granted to a dataset meeting the quality and metadata standard."),
                ("Blocking test", "A quality test whose failure prevents publication of the dataset."),
                ("Conformed entity", "A dataset standardised to shared definitions and keys across sources."),
                ("Quarantine", "Restriction applied to a dataset that has no registered owner after 30 days."),
                ("Personal-data tag", "The column marker that drives automated masking outside production."),
            ],
        ),
    ]
    return (
        "Data_Governance_and_Catalog_Standard.pdf",
        "engineering",
        "Northwind Systems — Data Governance and Catalogue Standard (DGS-2025.3)",
        sections,
    )


def _engineering_handbook() -> Document:
    # CONTRADICTION note: section 3 sets a 90-minute production deployment
    # freeze before month end, while the Cloud Architecture Guide implies
    # continuous deployment. Section 2 also sets a 2-reviewer rule that differs
    # from the 1-reviewer rule implied elsewhere - both are intentional, to
    # exercise conflicting-source handling.
    sections: list[Section] = [
        (
            "1. Branching and Code Review",
            [
                "All work is done on short-lived branches cut from main. A branch older than five "
                "working days must be rebased or abandoned. Direct commits to main are blocked by "
                "branch protection.",
                "Every pull request requires approval from two reviewers, at least one of whom "
                "must be a member of the owning team. Pull requests touching payment, "
                "authentication or cryptographic code require an additional approval from the "
                "Security Engineering team.",
                "Pull requests should not exceed 400 changed lines excluding generated files. "
                "Larger changes must be split or accompanied by a written rationale in the "
                "description.",
            ],
        ),
        (
            "2. Testing Requirements",
            [
                "Unit test line coverage must be at least 80% for new and modified code. Coverage "
                "is measured on the diff, not on the whole repository, so legacy gaps do not block "
                "unrelated work.",
                "Every bug fix must include a regression test that fails without the fix. Every "
                "tier-1 service must maintain an end-to-end smoke test suite that runs against "
                "production after each deployment.",
                "Load testing is required before any change expected to alter resource consumption "
                "by more than 20%, and before any change to connection pooling, which is a "
                "documented prior cause of production incidents.",
            ],
        ),
        (
            "3. Deployment and Change Management",
            [
                "Deployments to production use progressive rollout: 10% of instances first, a "
                "30-minute soak with automated metric comparison, then full rollout. Automated "
                "rollback triggers on error rate exceeding 1% or p99 latency exceeding 150% of "
                "baseline.",
                "A change freeze applies from 16:00 UTC on the last business day of each month "
                "until 09:00 UTC on the first business day of the following month, to protect "
                "financial close. Emergency fixes during a freeze require Director approval.",
                "Database configuration changes are treated as tier-1 changes regardless of the "
                "service affected, and must include a capacity calculation validated against the "
                "database limits before approval.",
            ],
        ),
        (
            "4. On-Call Expectations",
            [
                "On-call rotations are weekly and no engineer may be on call for more than one "
                "week in any four. Primary on-call must acknowledge a page within 5 minutes; "
                "secondary is paged automatically after 10 minutes without acknowledgement.",
                "Engineers who are paged outside working hours are entitled to equivalent time off "
                "in lieu. An engineer paged more than three times in a single night must not work "
                "the following day.",
                "Every page must result in either an action or a tuning change to the alert. Alerts "
                "that page without requiring action are treated as defects and must be fixed or "
                "removed within two weeks.",
            ],
        ),
        (
            "5. Documentation and Ownership",
            [
                "Every service must have a service page recording its tier, owner, dependencies, "
                "SLOs, runbook links and escalation path. Service pages are reviewed at each "
                "quarterly architecture review.",
                "A service without a named owning team is treated as an incident: it is escalated "
                "to the Director of Engineering and must be assigned within 10 business days.",
            ],
        ),
        (
            "6. Appendix A — Definition of Done Checklist",
            ["All items must be satisfied before a change is considered complete."]
            + _paragraphize(
                [
                    f"{cid}: {item}"
                    for cid, item in (
                        ("DOD-01", "Acceptance criteria in the ticket are all demonstrably met."),
                        ("DOD-02", "Diff coverage is at or above 80% and the full suite passes."),
                        ("DOD-03", "A regression test exists for every fixed defect."),
                        ("DOD-04", "Two reviewer approvals are recorded, plus security approval where required."),
                        ("DOD-05", "Structured logging is emitted at service boundaries with a correlation identifier."),
                        ("DOD-06", "Metrics and alerts exist for the new code path, and alerts are actionable."),
                        ("DOD-07", "The runbook is updated for any new failure mode introduced."),
                        ("DOD-08", "Feature flags used for rollout have a documented removal date."),
                        ("DOD-09", "Database migrations are backward compatible with the previous release."),
                        ("DOD-10", "No secret, credential or customer identifier appears in code or logs."),
                        ("DOD-11", "Dependency additions are licence-checked and vulnerability-scanned."),
                        ("DOD-12", "The service page is updated if dependencies or SLOs changed."),
                    )
                ],
                3,
            ),
        ),
        _glossary(
            7,
            "Engineering process terminology used across teams.",
            [
                ("Progressive rollout", "Deploying to a small subset first and observing before proceeding."),
                ("Soak period", "The observation window after partial rollout before continuing."),
                ("Diff coverage", "Test coverage measured only on lines changed by the current change."),
                ("Branch protection", "Repository rules preventing direct commits and enforcing review."),
                ("Change freeze", "A period during which non-emergency production changes are prohibited."),
                ("Page", "An alert that wakes a human, as distinct from a ticket or a dashboard signal."),
                ("Time off in lieu", "Compensatory leave granted for out-of-hours incident work."),
                ("Service page", "The canonical record of a service's tier, owner, SLOs and runbooks."),
                ("Backward-compatible migration", "A schema change that the previous release can still operate against."),
                ("Actionable alert", "An alert whose firing always implies a specific human action."),
            ],
        ),
    ]
    return (
        "Engineering_Standards_Handbook.pdf",
        "engineering",
        "Northwind Systems — Engineering Standards Handbook v3.1",
        sections,
    )


EXTRA_BUILDERS = (
    _data_privacy,
    _business_continuity,
    _sales_compensation,
    _procurement,
    _support_playbook,
    _model_risk,
    _credit_risk,
    _data_governance,
    _engineering_handbook,
)
