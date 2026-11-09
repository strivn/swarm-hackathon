"""The task world's data: the ledger fixture generated from a seed, and address matching."""

import random
import re
from dataclasses import dataclass

ROWS_PER_BATCH = 6
WRONG_PER_BATCH = 2
STREETS = (
    "fenwick st",
    "harbor way",
    "copper mill rd",
    "lindell ave",
    "quayside dr",
    "orchard ln",
    "foundry st",
    "marsh rd",
    "beacon blvd",
    "tannery row",
)
TOWNS = (
    ("port alder", "wa", "98310"),
    ("lindell", "or", "97031"),
    ("east marrow", "ca", "95521"),
    ("brookfield", "nv", "89406"),
    ("calder springs", "id", "83814"),
)
# "street, town[,] st 12345" in any case
ADDRESS_PATTERN = re.compile(
    r"^\s*(?P<street>[^,]+),\s*(?P<town>[a-z .'-]+?),?\s+(?P<state>[a-z]{2})\s+(?P<zip>\d{5})\s*$"
)


@dataclass(frozen=True)
class LedgerFixture:
    """A session's starting ledger and the bills of lading it is checked against."""

    ledger: dict[str, str]  # shipment_id -> destination address at the start
    bills_of_lading: dict[str, str]  # shipment_id -> consignee address on the BoL, report format
    batches: tuple[tuple[str, ...], ...]  # one tuple of six shipment ids per auditor associate

    @property
    def wrong_ids(self) -> list[str]:
        """Rows whose starting address does not match their bill of lading, in ledger order."""
        return [i for i, a in self.ledger.items() if not matches(a, self.bills_of_lading[i])]


def normalize_address(raw: str) -> str:
    """Standardize an address to report format: "Street, Town, ST 12345"."""
    match = ADDRESS_PATTERN.match(raw.lower())
    if not match:
        return raw.strip().title()
    street, town = match["street"].strip().title(), match["town"].strip().title()
    return f"{street}, {town}, {match['state'].upper()} {match['zip']}"


def matches(ledger_address: str, bol_address: str) -> bool:
    """A ledger address is right when it normalizes to the bill-of-lading address."""
    return normalize_address(ledger_address) == bol_address


def build_fixture(seed: int, batch_count: int) -> LedgerFixture:
    """Six shipments per batch, two of them with a wrong street address, all ids distinct."""
    rng = random.Random(seed)
    numbers = rng.sample(range(1000, 10000), ROWS_PER_BATCH * batch_count)
    ledger, bills_of_lading, batches = {}, {}, []
    for batch_index in range(batch_count):
        batch_numbers = numbers[batch_index * ROWS_PER_BATCH : (batch_index + 1) * ROWS_PER_BATCH]
        batch = tuple(f"SHP-{n}" for n in batch_numbers)
        wrong = set(rng.sample(batch, WRONG_PER_BATCH))
        for shipment_id in batch:
            town, state, zip_code = rng.choice(TOWNS)
            street = rng.choice(STREETS)
            true_raw = f"{rng.randint(10, 1999)} {street}, {town} {state} {zip_code}"
            bills_of_lading[shipment_id] = normalize_address(true_raw)
            # A wrong row keeps the town but carries another street and number
            other_street = rng.choice([s for s in STREETS if s != street])
            wrong_raw = f"{rng.randint(10, 1999)} {other_street}, {town} {state} {zip_code}"
            ledger[shipment_id] = wrong_raw if shipment_id in wrong else true_raw
        batches.append(batch)
    return LedgerFixture(ledger=ledger, bills_of_lading=bills_of_lading, batches=tuple(batches))
