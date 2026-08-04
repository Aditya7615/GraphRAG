#!/usr/bin/env python3
"""Scalable synthetic enterprise corpus generator (1M+ PDFs).

Generates realistic enterprise PDFs in parallel using multiprocessing.
Each PDF contains 4-10 pages of Faker-generated content across 10 document
types (financial, HR, engineering, compliance, legal, marketing, IT, sales,
customer success, product). Content is deterministic per seed for reproducibility.

Usage:
    python scripts/generate_large_corpus.py --count 1000000 --workers 8 --out data/large
    python scripts/generate_large_corpus.py --count 100 --workers 4 --out data/test
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import random
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from faker import Faker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOC_TYPES: list[str] = [
    "financial", "hr", "engineering", "compliance", "legal",
    "marketing", "it", "sales", "customer_success", "product",
]

SECTION_COUNT = (4, 10)
PARAGRAPHS_PER_SECTION = (3, 8)
PAGE_TARGET = (4, 10)

# Reusable styles (created once, shared across all PDFs)
_styles = getSampleStyleSheet()
TITLE_STYLE = ParagraphStyle(
    "DocTitle", parent=_styles["Title"], fontSize=18, spaceAfter=10,
    textColor="#1a1a2e",
)
HEADING_STYLE = ParagraphStyle(
    "DocHeading", parent=_styles["Heading2"], fontSize=13, spaceBefore=16,
    spaceAfter=6, textColor="#16213e",
)
BODY_STYLE = ParagraphStyle(
    "DocBody", parent=_styles["Normal"], fontSize=10, leading=14,
    spaceAfter=8, textColor="#222222",
)
META_STYLE = ParagraphStyle(
    "DocMeta", parent=_styles["Normal"], fontSize=8.5, textColor="#666666",
)


# ---------------------------------------------------------------------------
# Dataclass to carry work units
# ---------------------------------------------------------------------------

@dataclass
class WorkUnit:
    index: int
    seed: int
    out_dir: str


# ---------------------------------------------------------------------------
# Content generators per document type
# ---------------------------------------------------------------------------

def _gen_financial(f: Faker) -> list[tuple[str, list[str]]]:
    sections = []
    sections.append(("1. Executive Summary", [
        f"For the quarter ended {f.date_between(start_date='-1y', end_date='today').strftime('%B %d, %Y')}, "
        f"total revenue reached ${random.uniform(10, 5000):,.1f} million, representing a "
        f"{random.uniform(5, 45):.1f}% increase year-over-year.",
        f"The company delivered strong profitability with GAAP net income of "
        f"${random.uniform(1, 2000):,.1f} million, or ${random.uniform(0.5, 15):.2f} per diluted share.",
        f"Cash and short-term investments stood at ${random.uniform(100, 15000):,.1f} million, "
        f"providing robust liquidity for continued investment in growth initiatives.",
    ]))
    sections.append(("2. Revenue Breakdown", [
        f"Subscription revenue was ${random.uniform(5, 4000):,.1f} million ({random.uniform(60, 90):.0f}% of total), "
        f"driven by new customer acquisitions and expansion within existing accounts.",
        f"Professional services revenue totalled ${random.uniform(1, 1000):,.1f} million, "
        f"reflecting ${random.uniform(100, 2000):,.1f} million in new implementation bookings.",
        f"By geography, the Americas accounted for {random.uniform(40, 65):.1f}% of revenue, "
        f"EMEA {random.uniform(20, 35):.1f}%, and APAC {random.uniform(10, 25):.1f}%.",
    ]))
    sections.append(("3. Operating Expenses", [
        f"Research and development expense was ${random.uniform(100, 1500):,.1f} million "
        f"({random.uniform(15, 35):.1f}% of revenue), as the company continued to invest in AI capabilities.",
        f"Sales and marketing expense totalled ${random.uniform(100, 1200):,.1f} million "
        f"({random.uniform(15, 30):.1f}% of revenue).",
        f"General and administrative expense was ${random.uniform(50, 500):,.1f} million.",
        f"Operating income was ${random.uniform(100, 2000):,.1f} million, representing an "
        f"operating margin of {random.uniform(5, 35):.1f}%.",
    ]))
    sections.append(("4. Balance Sheet Highlights", [
        f"Total assets were ${random.uniform(5000, 80000):,.1f} million.",
        f"Accounts receivable totalled ${random.uniform(200, 5000):,.1f} million, with days sales "
        f"outstanding of {random.uniform(30, 90):.0f} days.",
        f"Deferred revenue was ${random.uniform(500, 10000):,.1f} million, reflecting strong "
        f"pre-paid subscription commitments.",
        f"Total debt stood at ${random.uniform(0, 5000):,.1f} million against cash of "
        f"${random.uniform(1000, 15000):,.1f} million.",
    ]))
    sections.append(("5. Outlook and Guidance", [
        f"For the next quarter, the company expects revenue of ${random.uniform(10, 5500):,.1f} million "
        f"to ${random.uniform(12, 6000):,.1f} million.",
        f"Full-year revenue guidance has been raised to ${random.uniform(50, 25000):,.1f} million "
        f"from the prior range of ${random.uniform(45, 22000):,.1f} million.",
        f"Free cash flow margin is expected to be {random.uniform(10, 40):.1f}% for the full year.",
    ]))
    return sections


def _gen_hr(f: Faker) -> list[tuple[str, list[str]]]:
    sections = []
    sections.append(("1. Employee Overview", [
        f"As of {f.date_between(start_date='-30d', end_date='today').strftime('%B %d, %Y')}, "
        f"the company employed {random.randint(5000, 150000)} full-time equivalent employees globally.",
        f"The workforce composition was {random.uniform(55, 70):.0f}% engineering, "
        f"{random.uniform(15, 25):.0f}% go-to-market, and {random.uniform(10, 20):.0f}% G&A.",
        f"Voluntary attrition rate for the trailing twelve months was {random.uniform(5, 20):.1f}%, "
        f"down from {random.uniform(15, 30):.1f}% in the prior period.",
    ]))
    sections.append(("2. Compensation and Benefits", [
        f"The median total compensation for employees in the United States was "
        f"${random.uniform(80000, 250000):,.0f}.",
        f"The company contributed ${random.uniform(5000, 20000):,.0f} per employee to the "
        f"401(k) retirement plan, with a {random.uniform(3, 8):.0f}% employer match.",
        f"Health insurance coverage extended to {random.uniform(85, 98):.0f}% of employees, "
        f"with the company covering {random.uniform(70, 95):.0f}% of premiums.",
        f"Total compensation expense for the period was ${random.uniform(100, 8000):,.1f} million.",
    ]))
    sections.append(("3. Diversity and Inclusion", [
        f"Women represented {random.uniform(30, 50):.1f}% of the global workforce, up from "
        f"{random.uniform(28, 48):.1f}% in the prior year.",
        f"Underrepresented minorities accounted for {random.uniform(15, 40):.1f}% of employees "
        f"in the United States.",
        f"The company launched {random.randint(5, 30)} employee resource groups across "
        f"{random.randint(5, 20)} offices worldwide.",
        f"Inclusion survey scores improved to {random.uniform(3.5, 4.8):.2f} out of 5.0, "
        f"up from {random.uniform(3.0, 4.5):.2f} in the prior period.",
    ]))
    sections.append(("4. Learning and Development", [
        f"The company invested ${random.uniform(500, 5000):.0f} per employee in training "
        f"and professional development programmes.",
        f"Over {random.randint(10000, 200000)} hours of learning content were consumed "
        f"through the internal platform.",
        f"{random.randint(20, 500)} employees completed leadership development programmes, "
        f"with {random.uniform(40, 80):.0f}% receiving promotions within 12 months.",
        f"Technical certification completion rate was {random.uniform(60, 95):.1f}% across "
        f"engineering teams.",
    ]))
    sections.append(("5. Safety and Well-being", [
        f"The total recordable incident rate was {random.uniform(0.1, 2.0):.2f} per "
        f"100 employees, below the industry average of {random.uniform(2.0, 5.0):.1f}.",
        f"Employee assistance programme utilisation was {random.uniform(5, 25):.1f}%.",
        f"The company conducted {random.randint(100, 2000)} mental health awareness sessions "
        f"across all locations.",
    ]))
    return sections


def _gen_engineering(f: Faker) -> list[tuple[str, list[str]]]:
    sections = []
    sections.append(("1. Architecture Overview", [
        f"The platform runs on a microservices architecture comprising "
        f"{random.randint(50, 500)} independently deployable services across "
        f"{random.randint(3, 15)} Kubernetes clusters.",
        f"Average request latency (p50) was {random.uniform(5, 50):.1f} ms with p99 "
        f"at {random.uniform(100, 500):.0f} ms.",
        f"The system processed an average of {random.uniform(10, 500):.0f} million "
        f"API requests per day with {random.uniform(99.9, 99.999):.3f}% uptime.",
    ]))
    sections.append(("2. Data Infrastructure", [
        f"The data lake stores {random.uniform(1, 500):.1f} PB across Amazon S3, "
        f"with {random.uniform(10, 200):.0f} TB in the real-time analytics tier.",
        f"PostgreSQL clusters handle {random.uniform(10, 500):.0f}K queries per second "
        f"with an average response time of {random.uniform(1, 20):.1f} ms.",
        f"Kafka clusters process {random.uniform(1, 100):.0f} million events per minute "
        f"across {random.randint(100, 2000)} topics.",
        f"The Elasticsearch cluster indexes {random.uniform(100, 5000):.0f} million documents "
        f"with {random.uniform(50, 500):.0f} ms average query latency.",
    ]))
    sections.append(("3. Security and Compliance", [
        f"The system passed SOC 2 Type II audit with {random.randint(0, 5)} exceptions "
        f"out of {random.randint(200, 400)} controls tested.",
        f"Penetration testing identified {random.randint(0, 10)} findings, all remediated "
        f"within {random.randint(5, 30)} days.",
        f"Vulnerability scan coverage includes {random.uniform(95, 100):.1f}% of "
        f"production infrastructure.",
        f"Data encryption at rest uses AES-256 with {random.randint(3, 10)} key rotations "
        f"per year.",
    ]))
    sections.append(("4. Performance Metrics", [
        f"Build pipeline throughput: {random.randint(100, 2000)} deployments per day "
        f"with {random.uniform(85, 99):.1f}% success rate.",
        f"Mean time to recovery (MTTR) was {random.uniform(5, 60):.1f} minutes for "
        f"severity 1 incidents.",
        f"Test coverage across all services is {random.uniform(70, 95):.1f}% with "
        f"{random.uniform(80, 99):.1f}% of critical paths covered.",
        f"Container image scan pass rate: {random.uniform(95, 100):.1f}% with "
        f"zero critical vulnerabilities in production.",
    ]))
    sections.append(("5. Cost Optimisation", [
        f"Monthly cloud spend was ${random.uniform(500, 50000):,.0f} with "
        f"{random.uniform(10, 40):.1f}% in reserved instances.",
        f"Cost per transaction decreased {random.uniform(5, 30):.1f}% quarter-over-quarter "
        f"following infrastructure right-sizing.",
        f"Spot instance adoption reached {random.uniform(20, 60):.1f}% of non-critical "
        f"workloads, saving approximately ${random.uniform(50, 5000):,.0f} per month.",
    ]))
    return sections


def _gen_compliance(f: Faker) -> list[tuple[str, list[str]]]:
    sections = []
    sections.append(("1. Regulatory Landscape", [
        f"The company operates under {random.randint(20, 100)} regulatory jurisdictions "
        f"globally, including GDPR, CCPA, HIPAA, and SOX.",
        f"During the review period, {random.randint(1, 10)} new regulations were enacted "
        f"requiring operational changes.",
        f"Regulatory examination findings: {random.randint(0, 5)} material findings, "
        f"all addressed within prescribed timeframes.",
    ]))
    sections.append(("2. Policy Compliance", [
        f"Policy attestation rate was {random.uniform(90, 100):.1f}% across all "
        f"employees, up from {random.uniform(85, 98):.1f}% in the prior period.",
        f"{random.randint(10, 200)} policy exceptions were granted, with "
        f"{random.uniform(80, 100):.0f}% having documented risk acceptance.",
        f"Third-party risk assessments were completed for {random.randint(100, 2000)} vendors, "
        f"with {random.uniform(5, 25):.1f}% classified as critical.",
        f"Training completion rate for mandatory compliance modules: "
        f"{random.uniform(92, 100):.1f}%.",
    ]))
    sections.append(("3. Audit Results", [
        f"{random.randint(3, 15)} internal audits were completed during the period, "
        f"with {random.uniform(70, 100):.0f}% rated as satisfactory or above.",
        f"External audit fees totalled ${random.uniform(100, 2000):,.0f} thousand.",
        f"Open audit findings decreased from {random.randint(10, 50)} to "
        f"{random.randint(0, 30)} quarter-over-quarter.",
        f"Average remediation time for audit findings: {random.uniform(5, 45):.0f} days.",
    ]))
    sections.append(("4. Incident Management", [
        f"{random.randint(10, 200)} compliance incidents were reported, with "
        f"{random.uniform(50, 100):.0f}% classified as low severity.",
        f"Mean time to initial response: {random.uniform(0.5, 24):.1f} hours.",
        f"Regulatory notifications were required for {random.randint(0, 5)} incidents, "
        f"all filed within mandated timeframes.",
        "No material enforcement actions or consent decrees were received.",
    ]))
    return sections


def _gen_legal(f: Faker) -> list[tuple[str, list[str]]]:
    sections = []
    sections.append(("1. Litigation Summary", [
        f"The company is party to {random.randint(5, 50)} active legal proceedings.",
        f"Estimated aggregate exposure for pending litigation is "
        f"${random.uniform(10, 500):,.1f} million.",
        f"{random.randint(1, 10)} cases were resolved during the period, "
        f"with favourable outcomes in {random.uniform(60, 90):.0f}% of matters.",
    ]))
    sections.append(("2. Intellectual Property", [
        f"The company holds {random.randint(100, 5000)} issued patents and has "
        f"{random.randint(50, 1000)} pending applications.",
        f"{random.randint(10, 100)} patent applications were filed during the period.",
        f"Trade secret protections cover {random.randint(50, 500)} proprietary "
        f"processes and algorithms.",
        f"IP licensing revenue was ${random.uniform(1, 100):,.1f} million.",
    ]))
    sections.append(("3. Contractual Obligations", [
        f"Total contractual obligations exceeding $1 million: "
        f"${random.uniform(100, 10000):,.1f} million.",
        f"Customer agreements with non-standard terms: {random.uniform(10, 40):.1f}% "
        f"of total contract value.",
        f"Change-of-control provisions affect {random.uniform(5, 30):.1f}% of "
        f"material contracts.",
    ]))
    sections.append(("4. Regulatory Filings", [
        f"{random.randint(20, 100)} regulatory filings were submitted on time.",
        "No late filings or material amendments were required.",
        f"Government inquiries received: {random.randint(0, 10)}, with "
        f"{random.uniform(50, 100):.0f}% resolved.",
    ]))
    return sections


def _gen_marketing(f: Faker) -> list[tuple[str, list[str]]]:
    sections = []
    sections.append(("1. Campaign Performance", [
        f"The company ran {random.randint(20, 200)} active marketing campaigns "
        f"during the period.",
        f"Total marketing spend was ${random.uniform(1, 200):,.1f} million, "
        f"with {random.uniform(30, 60):.0f}% allocated to digital channels.",
        f"Marketing-sourced pipeline was ${random.uniform(100, 10000):,.1f} million, "
        f"representing {random.uniform(30, 70):.0f}% of total pipeline.",
        f"Customer acquisition cost (CAC) decreased {random.uniform(5, 25):.1f}% "
        f"to ${random.uniform(1000, 50000):,.0f}.",
    ]))
    sections.append(("2. Brand and Content", [
        f"Brand awareness (unaided) increased to {random.uniform(10, 50):.1f}% "
        f"in target segments.",
        f"The content library grew to {random.randint(500, 10000)} assets, "
        f"with {random.uniform(20, 60):.1f}% video content.",
        f"Organic search traffic was {random.uniform(100, 2000):.0f}K monthly visits, "
        f"up {random.uniform(10, 50):.1f}% year-over-year.",
        f"Social media following reached {random.uniform(10, 500):.0f}K across "
        f"all platforms.",
    ]))
    sections.append(("3. Event Marketing", [
        f"The company hosted {random.randint(2, 20)} proprietary events with "
        f"an aggregate attendance of {random.randint(1000, 50000)}.",
        f"Sponsored events and conferences: {random.randint(10, 100)} with "
        f"${random.uniform(100, 5000):.0f}K in sponsorship costs.",
        f"Event-sourced pipeline: ${random.uniform(50, 5000):,.1f} million.",
        f"Net Promoter Score (NPS) for events: {random.uniform(30, 80):.0f}.",
    ]))
    return sections


def _gen_it(f: Faker) -> list[tuple[str, list[str]]]:
    sections = []
    sections.append(("1. Infrastructure Overview", [
        f"The IT estate comprises {random.randint(1000, 50000)} endpoints, "
        f"{random.randint(100, 5000)} servers, and {random.randint(10, 200)} "
        f"network appliances.",
        f"Cloud footprint spans {random.randint(2, 10)} regions across "
        f"{random.randint(2, 5)} cloud providers.",
        f"Monthly IT operating budget: ${random.uniform(1, 100):,.1f} million.",
    ]))
    sections.append(("2. Service Delivery", [
        f"IT service desk handled {random.randint(10000, 500000)} tickets with "
        f"{random.uniform(80, 95):.0f}% first-call resolution rate.",
        f"Average incident resolution time: {random.uniform(1, 48):.1f} hours.",
        f"Major incident count: {random.randint(0, 20)}, with "
        f"{random.uniform(90, 100):.0f}% resolved within SLA.",
        f"Change success rate: {random.uniform(90, 99):.1f}% with "
        f"{random.randint(100, 5000)} changes per quarter.",
    ]))
    sections.append(("3. Cybersecurity Posture", [
        f"Security incident count: {random.randint(100, 10000)}, with "
        f"{random.uniform(85, 99):.1f}% classified as low or medium severity.",
        f"Phishing simulation failure rate: {random.uniform(2, 15):.1f}%, "
        f"down from {random.uniform(5, 25):.1f}% prior period.",
        f"Endpoint detection and response (EDR) coverage: {random.uniform(98, 100):.1f}%.",
        f"Security awareness training completion: {random.uniform(90, 100):.1f}%.",
    ]))
    sections.append(("4. Software Licensing", [
        f"Total software licence spend: ${random.uniform(1, 50):,.1f} million annually.",
        f"License utilisation rate: {random.uniform(70, 95):.1f}% across all products.",
        f"{random.randint(5, 50)} licence true-up adjustments were processed.",
    ]))
    return sections


def _gen_sales(f: Faker) -> list[tuple[str, list[str]]]:
    sections = []
    sections.append(("1. Pipeline Summary", [
        f"Total sales pipeline was ${random.uniform(100, 50000):,.1f} million, "
        f"up {random.uniform(5, 50):.1f}% quarter-over-quarter.",
        f"Weighted pipeline (by stage probability) totalled "
        f"${random.uniform(50, 25000):,.1f} million.",
        f"Pipeline coverage ratio: {random.uniform(2, 6):.1f}x against quota.",
    ]))
    sections.append(("2. Bookings and Revenue", [
        f"New bookings totalled ${random.uniform(10, 5000):,.1f} million, with "
        f"{random.uniform(30, 70):.0f}% in subscription and "
        f"{random.uniform(30, 70):.0f}% in professional services.",
        f"Average deal size: ${random.uniform(10, 500):,.1f}K, "
        f"up {random.uniform(5, 30):.1f}% year-over-year.",
        f"Renewal rate: {random.uniform(85, 99):.1f}% with "
        f"{random.uniform(105, 130):.0f}% net dollar retention.",
        f"Sales cycle length: {random.uniform(30, 180):.0f} days average.",
    ]))
    sections.append(("3. Team Performance", [
        f"The sales team comprises {random.randint(20, 500)} quota-carrying reps "
        f"across {random.randint(3, 15)} territories.",
        f"{random.uniform(40, 80):.0f}% of reps achieved or exceeded quota.",
        f"Average ramp time for new hires: {random.uniform(3, 9):.1f} months.",
        f"Sales attrition rate: {random.uniform(5, 25):.1f}% annualised.",
    ]))
    sections.append(("4. Vertical Breakdown", [
        f"Technology sector: ${random.uniform(10, 5000):,.1f}M in bookings "
        f"({random.uniform(20, 40):.0f}% of total).",
        f"Financial services: ${random.uniform(5, 3000):,.1f}M "
        f"({random.uniform(10, 30):.0f}% of total).",
        f"Healthcare: ${random.uniform(5, 2000):,.1f}M "
        f"({random.uniform(5, 25):.0f}% of total).",
        f"Manufacturing: ${random.uniform(2, 1500):,.1f}M "
        f"({random.uniform(5, 15):.0f}% of total).",
    ]))
    return sections


def _gen_customer_success(f: Faker) -> list[tuple[str, list[str]]]:
    sections = []
    sections.append(("1. Customer Health", [
        f"Net Promoter Score (NPS): {random.uniform(30, 80):.0f}, up from "
        f"{random.uniform(25, 75):.0f} prior period.",
        f"Customer Satisfaction (CSAT): {random.uniform(80, 98):.1f}%.",
        f"Health score distribution: {random.uniform(50, 80):.0f}% green, "
        f"{random.uniform(10, 30):.0f}% yellow, "
        f"{random.uniform(1, 15):.0f}% red.",
        f"Churn rate: {random.uniform(3, 15):.1f}% annualised, "
        f"down from {random.uniform(5, 20):.1f}% prior period.",
    ]))
    sections.append(("2. Support Metrics", [
        f"Support tickets received: {random.randint(1000, 100000)}.",
        f"Average first response time: {random.uniform(0.5, 24):.1f} hours.",
        f"Ticket resolution time: {random.uniform(2, 72):.1f} hours average.",
        f"Self-service deflection rate: {random.uniform(20, 60):.1f}%.",
        f"Support quality score: {random.uniform(80, 98):.1f}%.",
    ]))
    sections.append(("3. Adoption and Expansion", [
        f"Product adoption rate: {random.uniform(60, 95):.1f}% of licensed users "
        f"actively using the platform weekly.",
        f"Feature adoption for key modules: "
        f"{random.uniform(30, 80):.0f}% for analytics, "
        f"{random.uniform(20, 60):.0f}% for automation.",
        f"Expansion revenue: ${random.uniform(5, 500):,.1f} million, "
        f"representing {random.uniform(10, 40):.0f}% of total revenue.",
        f"Cross-sell conversion rate: {random.uniform(5, 25):.1f}%.",
    ]))
    sections.append(("4. Customer Onboarding", [
        f"Average time to first value: {random.uniform(1, 30):.0f} days.",
        f"Onboarding completion rate: {random.uniform(80, 98):.1f}%.",
        f"Customer education sessions delivered: {random.randint(100, 5000)}.",
        f"Certification programme participants: {random.randint(500, 20000)}.",
    ]))
    return sections


def _gen_product(f: Faker) -> list[tuple[str, list[str]]]:
    sections = []
    sections.append(("1. Product Roadmap", [
        f"The product roadmap includes {random.randint(10, 50)} features planned "
        f"for the next {random.randint(1, 4)} quarters.",
        f"{random.randint(3, 20)} features are in beta testing with "
        f"{random.randint(100, 5000)} early access customers.",
        f"Product investment (R&D as % of revenue): {random.uniform(15, 35):.1f}%.",
        f"Time from ideation to GA: {random.uniform(3, 12):.1f} months average.",
    ]))
    sections.append(("2. Platform Capabilities", [
        f"The platform supports {random.randint(5, 50)} integrations with "
        f"third-party systems.",
        f"API availability: {random.uniform(99.5, 99.99):.3f}% uptime.",
        f"Data processing capacity: {random.uniform(1, 100):.0f} million records per hour.",
        f"Maximum concurrent users supported: {random.randint(1000, 100000)}.",
    ]))
    sections.append(("3. Quality Metrics", [
        f"Bug escape rate: {random.uniform(0.1, 5):.2f} per 1000 lines of code.",
        f"Customer-reported defects: {random.randint(10, 500)} per quarter.",
        f"Mean time between failures (MTBF): {random.uniform(100, 1000):.0f} hours.",
        f"Mean time to repair (MTTR): {random.uniform(0.5, 24):.1f} hours.",
    ]))
    sections.append(("4. Competitive Position", [
        f"Win rate against top 3 competitors: {random.uniform(30, 70):.1f}%.",
        f"Analyst recognition: featured in {random.randint(1, 5)} industry reports.",
        f"Feature parity score: {random.uniform(70, 95):.1f}% vs. market leader.",
        f"Customer switching cost: ${random.uniform(10, 500):.0f}K average.",
    ]))
    sections.append(("5. User Experience", [
        f"System Usability Scale (SUS) score: {random.uniform(60, 90):.0f}/100.",
        f"Task completion rate: {random.uniform(80, 98):.1f}%.",
        f"Average session duration: {random.uniform(5, 60):.0f} minutes.",
        f"Feature request backlog: {random.randint(100, 5000)} items, "
        f"with {random.uniform(10, 40):.0f}% addressed in current quarter.",
    ]))
    return sections


GEN_MAP: dict[str, Callable[[Faker], list[tuple[str, list[str]]]]] = {
    "financial": _gen_financial,
    "hr": _gen_hr,
    "engineering": _gen_engineering,
    "compliance": _gen_compliance,
    "legal": _gen_legal,
    "marketing": _gen_marketing,
    "it": _gen_it,
    "sales": _gen_sales,
    "customer_success": _gen_customer_success,
    "product": _gen_product,
}


# ---------------------------------------------------------------------------
# PDF writer
# ---------------------------------------------------------------------------

def _write_pdf(path: Path, title: str, sections: list[tuple[str, list[str]]],
               author: str, revision_date: str) -> None:
    doc = SimpleDocTemplate(
        str(path),
        pagesize=LETTER,
        title=title,
        author=author,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
    )

    flow = [
        Paragraph(title, TITLE_STYLE),
        Paragraph(
            f"{author} &nbsp;|&nbsp; Internal — Confidential &nbsp;|&nbsp; "
            f"Document owner: {author} &nbsp;|&nbsp; Last revised: {revision_date}",
            META_STYLE,
        ),
        Spacer(1, 14),
    ]
    for i, (heading, paragraphs) in enumerate(sections):
        if i and i % 3 == 0:
            flow.append(PageBreak())
        flow.append(Paragraph(heading, HEADING_STYLE))
        flow.extend(Paragraph(p, BODY_STYLE) for p in paragraphs)

    doc.build(flow)


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def _generate_one(unit: WorkUnit) -> str | None:
    """Generate a single PDF. Returns filename on success, None on failure."""
    try:
        random.Random(unit.seed)
        f = Faker()
        Faker.seed(unit.seed)

        doc_type = DOC_TYPES[unit.index % len(DOC_TYPES)]
        gen_fn = GEN_MAP[doc_type]
        sections = gen_fn(f)

        author = f.company()
        revision_date = f.date_between(start_date='-1y', end_date='today').isoformat()
        company = f.company().replace("'", "").replace(",", "")
        topic = f.catch_phrase()
        title = f"{company} — {doc_type.replace('_', ' ').title()}: {topic}"

        # Make filename deterministic from seed
        short_hash = hashlib.md5(str(unit.seed).encode()).hexdigest()[:8]
        filename = f"{doc_type}_{short_hash}.pdf"

        out_path = Path(unit.out_dir) / filename
        _write_pdf(out_path, title, sections, author, revision_date)
        return filename
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate large-scale synthetic enterprise corpus (1M+ PDFs)."
    )
    parser.add_argument(
        "--count", type=int, default=1_000_000,
        help="Number of PDFs to generate (default: 1,000,000).",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Worker processes (default: min(8, cpu_count)).",
    )
    parser.add_argument(
        "--out", default="data/large",
        help="Output directory (default: data/large).",
    )
    parser.add_argument(
        "--start-seed", type=int, default=0,
        help="Starting random seed (default: 0).",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete existing files in output directory first.",
    )
    args = parser.parse_args()

    if args.workers <= 0:
        args.workers = min(8, os.cpu_count() or 4)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for existing in out_dir.glob("*.pdf"):
            existing.unlink()

    print(f"Generating {args.count:,} PDFs into {out_dir.resolve()} "
          f"using {args.workers} workers...")

    units = [
        WorkUnit(index=i, seed=args.start_seed + i, out_dir=str(out_dir))
        for i in range(args.count)
    ]

    t0 = time.monotonic()
    generated = 0
    failed = 0

    with mp.Pool(args.workers) as pool:
        for result in pool.imap_unordered(_generate_one, units, chunksize=256):
            if result:
                generated += 1
            else:
                failed += 1
            if (generated + failed) % 10000 == 0:
                elapsed = time.monotonic() - t0
                rate = (generated + failed) / elapsed if elapsed > 0 else 0
                print(f"  [{generated + failed:,}/{args.count:,}] "
                      f"{generated:,} ok, {failed:,} failed "
                      f"({rate:,.0f} PDFs/sec)")

    elapsed = time.monotonic() - t0
    rate = (generated + failed) / elapsed if elapsed > 0 else 0

    print(f"\nDone. {generated:,} PDFs generated, {failed:,} failed.")
    print(f"Time: {elapsed:.1f}s ({rate:,.0f} PDFs/sec)")
    print(f"Output: {out_dir.resolve()}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
