"""A synthetic PDF with distinct facts on every page, for end-to-end tests."""
import random
import fitz

SUBJECTS = ["Aurora reactor", "Kestrel bridge", "Lumen protocol", "Orchid vaccine", "Tundra rover",
            "Vesper telescope", "Cobalt ledger", "Marrow algorithm", "Halcyon dam", "Quartz compiler",
            "Saffron canal", "Juniper satellite"]
PEOPLE = ["Mira Kovacs", "Tomas Ruel", "Anya Petrov", "Idris Okafor", "Lena Varga", "Paulo Mendes",
          "Hana Sato", "Ravi Menon", "Elif Demir", "Jonas Berg", "Nadia Haddad", "Owen Price"]
CITIES = ["Valdris", "Port Emery", "Kessington", "Lowmarsh", "Arden Falls", "Tirravale",
          "Brightwater", "Cindral", "Oskarhavn", "Mellow Bay", "Redcliff", "Zurrow"]
FILLER = ["Engineers reviewed the design several times before approval.",
          "Funding came from a mix of public grants and private partners.",
          "Local newspapers covered the project extensively during construction.",
          "Several smaller studies followed the initial announcement.",
          "Critics questioned the cost but supporters pointed to long term savings.",
          "The team published progress reports every quarter.",
          "Maintenance schedules were adjusted after the first winter.",
          "Training programs were organised for new operators."]


def make_pdf(path: str, pages=16, seed=3):
    rng = random.Random(seed)
    doc = fitz.open()
    for p in range(pages):
        s, person, city = SUBJECTS[p % 12], PEOPLE[p % 12], CITIES[p % 12]
        year, num = 1950 + 6 * p, 40 + 17 * p
        facts = [
            f"The {s} was designed by chief engineer {person} in the city of {city}.",
            f"Construction of the {s} finished in {year} after {num} months of work.",
            f"The {s} uses a {['magnetic', 'hydraulic', 'optical', 'thermal'][p % 4]} control system "
            f"that {person} patented.",
            f"Its annual operating budget reached {num * 3} million credits by {year + 10}.",
        ]
        body = []
        for f in facts:
            body.append(f)
            body += rng.sample(FILLER, 3)
        body += rng.sample(FILLER, 8) + rng.sample(FILLER, 8)
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 800), " ".join(body), fontsize=8)
    doc.save(path)
    return path
