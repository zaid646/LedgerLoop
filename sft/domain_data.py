"""Deterministic synthetic expense-coding instructions for LedgerLoop.

The examples mirror the GL taxonomy used by ``ai-service/app/schemas.py`` and the
merchant catalog in ``ai-service/app/catalog.py``. They are generated locally so
the SFT pipeline has a reproducible, license-free domain-adaptation corpus.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

GL_CATEGORIES: tuple[str, ...] = (
    "TRAVEL",
    "MEALS",
    "OFFICE_SUPPLIES",
    "SOFTWARE",
    "HARDWARE",
    "MARKETING",
    "PROFESSIONAL_SERVICES",
    "UTILITIES",
    "RENT",
    "INSURANCE",
    "TRAINING",
    "ENTERTAINMENT",
    "OTHER",
)

ALLOWED = ", ".join(GL_CATEGORIES)

_MERCHANTS: dict[str, list[tuple[str, str]]] = {
    "TRAVEL": [
        ("Uber", "Ride from the airport to the client office"),
        ("Lyft", "Late-night ride home after a customer visit"),
        ("Delta Air Lines", "Round-trip flight to the Chicago sales meeting"),
        ("United Airlines", "One-way airfare to the Denver conference"),
        ("Airbnb", "Two-night stay near the manufacturing plant"),
        ("Marriott Hotels", "Hotel room for the annual partner summit"),
        ("Shell", "Fuel for the rental car during a site visit"),
        ("Hertz", "Three-day rental car for the regional roadshow"),
        ("Enterprise Rent-A-Car", "Weekly rental for the field audit"),
        ("Southwest Airlines", "Flight change fee for the Austin trip"),
        ("British Airways", "International airfare to the London office"),
        ("Hilton", "Overnight stay before the early client workshop"),
        ("Chevron", "Diesel for the company van on a site inspection"),
        ("ParkWhiz", "Airport parking for a four-day business trip"),
    ],
    "MEALS": [
        ("Starbucks", "Coffee and breakfast during the client kickoff"),
        ("DoorDash", "Team lunch ordered for the quarterly planning day"),
        ("McDonald's", "Quick dinner while traveling between meetings"),
        ("Uber Eats", "Working lunch for the engineering stand-up"),
        ("Chipotle", "Catered burritos for the onboarding session"),
        ("Pret a Manger", "Sandwich and coffee before the board meeting"),
        ("Shake Shack", "Dinner with the visiting vendor team"),
        ("Blue Bottle Coffee", "Coffee for a candidate interview panel"),
        ("Panera Bread", "Catering for the department all-hands"),
        ("Grubhub", "Late dinner during the release cutover"),
        ("Subway", "Lunch for the warehouse inventory count"),
        ("Local Cafe", "Breakfast meeting with an external accountant"),
        ("Pizza Palace", "Team dinner after the product launch"),
        ("Freshii", "Working lunch for the audit fieldwork team"),
    ],
    "OFFICE_SUPPLIES": [
        ("Staples", "Printer paper and toner for the finance team"),
        ("Office Depot", "Notebooks, pens and file folders"),
        ("MR D.I.Y.", "Shelving and storage bins for the records room"),
        ("Amazon Business", "Whiteboard markers and sticky notes"),
        ("Uline", "Shipping boxes and packing tape"),
        ("Quill", "Desk organizers for the new hires"),
        ("Paperchase", "Folders and labels for the year-end archive"),
        ("B&H Photo", "Webcam and USB hub for the conference room"),
        ("Walmart Business", "Cleaning supplies for the office kitchen"),
        ("Target", "Kitchen supplies for the break room"),
        ("FedEx Office", "Printing and binding for the board pack"),
        ("Local Stationery", "Envelopes and letterhead for client mail"),
        ("Costco Business Center", "Bulk coffee and cups for the office"),
        ("Home Depot", "Extension cords for the desk refit"),
    ],
    "SOFTWARE": [
        ("GitHub", "Annual Team plan subscription for engineering"),
        ("Amazon Web Services", "Monthly cloud compute and storage usage"),
        ("Slack", "Business+ workspace subscription"),
        ("Microsoft", "Microsoft 365 licenses for the finance team"),
        ("Adobe", "Creative Cloud licenses for the design team"),
        ("Salesforce", "CRM seats for the revenue organization"),
        ("Notion", "Team workspace plan for product operations"),
        ("Zoom", "Annual webinar and meeting licenses"),
        ("Atlassian", "Jira and Confluence cloud subscription"),
        ("Datadog", "Observability platform usage for production"),
        ("Snowflake", "Data warehouse compute credits"),
        ("Figma", "Design collaboration seats for the product team"),
        ("Asana", "Project management licenses for delivery"),
        ("1Password", "Password manager seats for the company"),
    ],
    "HARDWARE": [
        ("Apple", "MacBook Pro for a new software engineer"),
        ("Dell", "Replacement monitors for the support desk"),
        ("Best Buy", "Docking stations for the hybrid work policy"),
        ("IKEA", "Standing desks for the office expansion"),
        ("Logitech", "Wireless keyboards and mice for new hires"),
        ("Lenovo", "ThinkPad laptops for the sales team"),
        ("HP", "Network switch for the branch office"),
        ("Samsung", "Tablets for the field inspection crew"),
        ("Apple Store", "iPad and keyboard case for the account team"),
        ("CDW", "Server memory upgrade for the data lab"),
        ("B&H Photo", "Monitors for the video production desk"),
        ("Newegg", "SSD drives for developer workstations"),
        ("Framework", "Laptop for the remote contractor"),
        ("Razer", "Headsets for the customer support team"),
    ],
    "MARKETING": [
        ("Google Ads", "Search campaign for the Q3 product launch"),
        ("Meta Ads", "Retargeting campaign for the webinar"),
        ("LinkedIn Ads", "Sponsored posts for the hiring campaign"),
        ("HubSpot", "Marketing automation subscription"),
        ("Mailchimp", "Email campaign plan for the newsletter"),
        ("Canva", "Brand templates for the sales enablement team"),
        ("Eventbrite", "Ticketing fees for the customer summit"),
        ("Hootsuite", "Social media scheduling subscription"),
        ("Semrush", "SEO research and rank tracking"),
        ("Vimeo", "Video hosting for the product demo"),
        ("Typeform", "Lead capture forms for the campaign"),
        ("Brandwatch", "Social listening for the launch"),
        ("AdRoll", "Display advertising for the retargeting funnel"),
        ("PR Newswire", "Press release distribution for the announcement"),
    ],
    "PROFESSIONAL_SERVICES": [
        ("Deloitte", "External audit support for the fiscal year"),
        ("FedEx", "Overnight courier of signed contracts"),
        ("PwC", "Tax advisory engagement for the quarter"),
        ("McKinsey", "Operations benchmark study"),
        ("Accenture", "Implementation consulting for the ERP rollout"),
        ("UPS", "Express shipping for legal documents"),
        ("DHL", "International courier for the vendor agreement"),
        ("Legal Counsel", "Contract review for the new supplier"),
        ("Notary Service", "Notarization of the lease amendment"),
        ("Translation Agency", "Localization of the investor deck"),
        ("Recruiting Agency", "Placement fee for the controller role"),
        ("Accounting Firm", "Bookkeeping support for the month-end close"),
        ("Background Check Provider", "Pre-employment screening for new hires"),
        ("Courier Service", "Same-day delivery of the signed proposal"),
    ],
    "UTILITIES": [
        ("Con Edison", "Electricity for the New York office"),
        ("PG&E", "Monthly electricity bill for the San Francisco office"),
        ("AT&T", "Business internet and phone service"),
        ("Verizon", "Mobile data plan for the field team"),
        ("Comcast", "Office broadband service"),
        ("City Water Utility", "Quarterly water and sewer charges"),
        ("Waste Management", "Office waste collection service"),
        ("National Grid", "Natural gas for the Boston office"),
        ("Duke Energy", "Electricity for the Charlotte branch"),
        ("T-Mobile", "Mobile hotspot plan for the audit team"),
        ("Spectrum", "Cable internet for the remote office"),
        ("Local Water District", "Water service for the warehouse"),
        ("Republic Services", "Recycling pickup for the office"),
        ("Pacific Gas", "Gas bill for the customer support center"),
    ],
    "RENT": [
        ("WeWork", "Monthly hot-desk membership for the sales team"),
        ("Regus", "Private office lease for the London team"),
        ("Office Lease Co", "Monthly rent for the headquarters floor"),
        ("Coworking Hub", "Meeting-room credits for the quarter"),
        ("Industrial Properties", "Warehouse rent for the inventory team"),
        ("Main Street Offices", "Satellite office rent for the support team"),
        ("FlexSpace", "Short-term office lease for the launch team"),
        ("Downtown Workspace", "Monthly coworking desks for contractors"),
        ("Storage Facility", "Archive storage unit for finance records"),
        ("Innovation Center", "Lab space rental for the hardware team"),
        ("Harbor Offices", "Client meeting suite rental"),
        ("Summit Coworking", "Day passes for the visiting engineering team"),
        ("City Center Properties", "Rent for the customer experience center"),
        ("Parkview Offices", "Monthly suite rental for the legal team"),
    ],
    "INSURANCE": [
        ("Aetna", "Employee health insurance premium"),
        ("Blue Cross", "Dental and vision coverage for the quarter"),
        ("State Farm", "Commercial general liability premium"),
        ("Hiscox", "Cyber liability insurance for the year"),
        ("Travelers", "Workers compensation premium"),
        ("Allianz", "Business travel insurance for the sales team"),
        ("Chubb", "Property insurance for the office"),
        ("Zurich", "Professional liability coverage"),
        ("Geico", "Commercial auto insurance for the fleet"),
        ("Nationwide", "Business owner policy premium"),
        ("MetLife", "Group life insurance premium"),
        ("The Hartford", "Umbrella liability coverage"),
        ("AXA", "International employee travel coverage"),
        ("Liberty Mutual", "Equipment insurance for the field team"),
    ],
    "TRAINING": [
        ("Coursera", "Data engineering course for the analytics team"),
        ("Udemy", "Python automation course for finance staff"),
        ("Pluralsight", "Cloud certification training seats"),
        ("Conference Registration", "Registration for the AI engineering summit"),
        ("O'Reilly", "Technical books and online learning subscription"),
        ("DataCamp", "SQL and analytics training for the audit team"),
        ("LinkedIn Learning", "Leadership course for new managers"),
        ("AWS Training", "Solutions architect certification bootcamp"),
        ("SANS Institute", "Security awareness training for engineers"),
        ("Local Training Co", "Excel modeling workshop for the FP&A team"),
        ("Product Management Institute", "Product discovery certification"),
        ("CFA Institute", "Financial analysis course materials"),
        ("Workshop Series", "Negotiation skills workshop for procurement"),
        ("Certification Body", "Project management exam fee"),
    ],
    "ENTERTAINMENT": [
        ("Netflix", "Team movie night subscription"),
        ("Spotify", "Office music subscription"),
        ("Cinema City", "Tickets for the customer appreciation event"),
        ("Ticketmaster", "Concert tickets for the partner event"),
        ("Steam", "Game licenses for the user research lab"),
        ("Bowling Alley", "Team-building bowling night"),
        ("Escape Room", "Department social event"),
        ("Museum Tickets", "Client entertainment at the art museum"),
        ("Local Theatre", "Sponsorship of the charity performance"),
        ("Sports Club", "Box seats for the client hospitality event"),
        ("Arcade Bar", "Team celebration after the product launch"),
        ("Comedy Club", "Customer dinner and show"),
        ("Wine Tasting", "Partner networking event"),
        ("Festival Passes", "Industry festival tickets for marketing"),
    ],
    "OTHER": [
        ("ATM Withdrawal", "Cash withdrawal for petty expenses"),
        ("Bank Fee", "Monthly business account fee"),
        ("Donation", "Charitable contribution approved by the committee"),
        ("Postage Meter", "Postage refill for the mailroom"),
        ("Miscellaneous Vendor", "Unclassified office purchase"),
        ("Currency Exchange", "Foreign exchange fee for the trip"),
        ("Late Fee", "Late payment charge on a supplier invoice"),
        ("Parking Fine", "Parking violation during a client visit"),
        ("Petty Cash", "Replenishment of the office petty cash box"),
        ("Bank Wire Fee", "International wire transfer fee"),
        ("Refund Adjustment", "Supplier refund reconciliation"),
        ("Uncategorized Purchase", "Item awaiting finance review"),
        ("Service Charge", "Bank service charge for the quarter"),
        ("Adjustment", "Manual accounting adjustment"),
    ],
}

_INSTRUCTIONS: tuple[str, ...] = (
    "You are LedgerLoop's expense coding assistant. Assign the transaction to exactly one "
    "general-ledger category from the allowed list. Return only the category code.\n\n"
    "Allowed categories: {allowed}\n\nTransaction:\n{transaction}",
    "Classify this business transaction into the correct GL account. Answer with one code "
    "from the allowed list and nothing else.\n\nAllowed categories: {allowed}\n\n"
    "Transaction:\n{transaction}",
    "Map the following expense to the company's chart of accounts. Use exactly one of the "
    "allowed codes.\n\nAllowed categories: {allowed}\n\nTransaction:\n{transaction}",
    "As an accounts-payable assistant, choose the best GL category for this expense. "
    "Respond with only the category code.\n\nAllowed categories: {allowed}\n\n"
    "Transaction:\n{transaction}",
)

_CURRENCIES: tuple[str, ...] = ("USD", "EUR", "GBP", "CAD", "AUD")


@dataclass(frozen=True)
class DomainExample:
    id: str
    category: str
    merchant: str
    description: str
    amount: float
    currency: str
    date: str
    instruction: str
    response: str


def _transaction_text(merchant: str, description: str, amount: float, currency: str, date: str) -> str:
    return (
        f"Merchant: {merchant}\n"
        f"Amount: {currency} {amount:.2f}\n"
        f"Date: {date}\n"
        f"Description: {description}"
    )


def build_domain_examples(seed: int = 42) -> list[DomainExample]:
    """Build 182 unique, deterministic instruction-response pairs."""
    rng = random.Random(seed)
    examples: list[DomainExample] = []
    for category, merchants in _MERCHANTS.items():
        for merchant, description in merchants:
            amount = round(rng.uniform(8.0, 1800.0), 2)
            currency = rng.choice(_CURRENCIES)
            date = f"2026-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}"
            transaction = _transaction_text(merchant, description, amount, currency, date)
            instruction = rng.choice(_INSTRUCTIONS).format(
                allowed=ALLOWED, transaction=transaction
            )
            examples.append(
                DomainExample(
                    id=f"ledgerloop-{len(examples):03d}",
                    category=category,
                    merchant=merchant,
                    description=description,
                    amount=amount,
                    currency=currency,
                    date=date,
                    instruction=instruction,
                    response=category,
                )
            )
    rng.shuffle(examples)
    return examples


def split_domain_examples(
    examples: list[DomainExample], train_size: int = 152
) -> tuple[list[DomainExample], list[DomainExample]]:
    if not 0 < train_size < len(examples):
        raise ValueError(f"train_size must be between 1 and {len(examples) - 1}")
    return examples[:train_size], examples[train_size:]


if __name__ == "__main__":
    data = build_domain_examples()
    train, val = split_domain_examples(data)
    print(f"domain examples: {len(data)} (train={len(train)}, eval={len(val)})")
    print(f"categories: {len({x.category for x in data})}")
    print(f"sample instruction:\n{train[0].instruction}")
    print(f"sample response: {train[0].response}")
