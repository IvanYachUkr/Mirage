"""
V15 -- Generate additional movie industry persons via Gemini Flash Lite
======================================================================
Rich, rotating prompts per batch. Self-adapts based on gaps in
generated data (underrepresented nationalities, roles, stages).
Incremental save after every batch. Never deletes existing data.

E2 scaling: run with --target 24000 to expand the person pool 4x.
generate_latent_vars_api.py handles the incremental latent var step.

Usage:
    python test15/generate_persons_llm.py --target 24000
    python test15/generate_persons_llm.py --target 24000 --batch-size 50
    python test15/generate_persons_llm.py --target 500   # small test
"""
import os, json, time, sys, argparse, random, re
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, os.path.dirname(__file__))
from contracts import (
    GENRES, STYLE_TAGS, CAREER_STAGES,
    NATIONALITIES, MARKETS, ENTITY_COUNTS,
)
from llm_provider import get_llm_client
try:
    from contracts import DIRECTOR_STYLES
except ImportError:
    DIRECTOR_STYLES = []

try:
    from contracts import MODEL_TIERS
    MODEL = MODEL_TIERS.get("person_gen", MODEL_TIERS.get("latent_vars", "gemini-3-flash-preview"))
except ImportError:
    MODEL = "gemini-3-flash-preview"

PRICING = {"input": 0.25, "output": 1.50}  # gemini-3-flash-preview: $0.25/$1.50 per 1M tokens
BASE_DIR = Path(__file__).parent
ENTITY_DIR = BASE_DIR / "entities"

# ─── Target distributions ────────────────────────────────────────────

# V15: 4x scaling goal = MORE ACTORS for better Gini coefficient.
# We already have 1244 directors (enough for 7500 movies at 5-6 films/director).
# New batches push actors to ~65% and slash directors to ~5%.
# Each multi-role string encodes array roles: "actor,director" -> ["actor", "director"]
ROLE_DISTRIBUTION = {
    "actor": 0.65,          # ^ primary goal: more actors for Gini
    "director": 0.05,       # v already over-represented (20.9% existing)
    "actor,director": 0.04, # v keep dual-role actors
    "producer": 0.06,       # v slightly
    "writer": 0.05,         # v already over-represented
    "writer,director": 0.02,
    "cinematographer": 0.05, # ^ need 1/movie x 7500 movies
    "editor": 0.04,          # ^ need 1/movie x 7500 movies
    "composer": 0.04,        # ^ need 1/movie x 7500 movies
}
assert abs(sum(ROLE_DISTRIBUTION.values()) - 1.0) < 0.001, "ROLE_DISTRIBUTION must sum to 1.0"

# Expanded to 80+ micro-nationalities / ethnic sub-groups.
# Each has a DISTINCT naming tradition -> near-zero cross-collision.
# Saturated pools (American, British, Korean) intentionally low-weighted
# so remaining 15k persons come from fresh, uncolliding name spaces.
NATIONALITY_WEIGHTS = {
    # North America (already have ~1800 American names -> reduced)
    "American": 0.04, "African-American": 0.02, "Chicano/Mexican-American": 0.02,
    "Puerto Rican": 0.01, "Native American/Choctaw": 0.01,
    "Cajun/French-Creole American": 0.01, "Hawaiian": 0.01,
    "Quebecois": 0.02, "Acadian-Canadian": 0.01, "British-Columbian Canadian": 0.01,
    # Europe (already have many British/French/German -> reduced)
    "British": 0.02, "Welsh": 0.01, "Scottish Gaelic": 0.01, "Northern Irish": 0.01,
    "French": 0.02, "Breton": 0.01, "Provencal French": 0.01,
    "German": 0.02, "Bavarian": 0.01,
    "Italian": 0.02, "Sicilian": 0.01, "Venetian": 0.01, "Neapolitan": 0.01,
    "Spanish": 0.01, "Basque": 0.02, "Catalan": 0.02, "Galician": 0.01, "Andalusian": 0.01,
    "Portuguese": 0.01, "Lusophone African": 0.01,
    "Swedish": 0.01, "Norwegian": 0.01, "Finnish": 0.01, "Saami": 0.01,
    "Danish": 0.01, "Faroese": 0.01,
    "Polish": 0.02, "Czech": 0.01, "Slovak": 0.01,
    "Hungarian": 0.01, "Romanian": 0.01, "Bulgarian": 0.01,
    "Serbian": 0.01, "Croatian": 0.01, "Slovenian": 0.01,
    "Greek": 0.01, "Albanian": 0.01, "Macedonian": 0.01,
    "Russian": 0.02, "Ukrainian": 0.01, "Belarusian": 0.01,
    "Georgian": 0.01, "Armenian": 0.01, "Azerbaijani": 0.01,
    "Irish": 0.01, "Maltese": 0.01,
    # Middle East / Central Asia
    "Iranian/Persian": 0.02, "Kurdish": 0.01, "Turkish": 0.02,
    "Egyptian Arab": 0.01, "Levantine Arab": 0.01, "Gulf Arab": 0.01,
    "Moroccan Amazigh": 0.01, "Tunisian": 0.01, "Algerian": 0.01,
    "Uzbek": 0.01, "Kazakh": 0.01, "Tajik": 0.01, "Kyrgyz": 0.01,
    "Afghan/Pashtun": 0.01, "Pakistani/Urdu": 0.01,
    # South & Southeast Asia (expand India sub-groups)
    "Tamil Indian": 0.02, "Bengali Indian": 0.02, "Punjabi Indian": 0.02,
    "Marathi Indian": 0.01, "Kannada Indian": 0.01, "Telugu Indian": 0.01,
    "Malayali Indian": 0.01, "Odia Indian": 0.01, "Assamese Indian": 0.01,
    "Sri Lankan Tamil": 0.01, "Sinhalese": 0.01,
    "Bangladeshi": 0.01, "Nepali": 0.01,
    "Burmese": 0.01, "Thai": 0.02, "Lao": 0.01, "Khmer": 0.01,
    "Vietnamese": 0.02, "Filipino/Tagalog": 0.02, "Cebuano Filipino": 0.01,
    "Indonesian/Javanese": 0.02, "Sundanese": 0.01, "Balinese": 0.01,
    "Malaysian Malay": 0.01, "Malaysian Chinese": 0.01,
    "Singaporean": 0.01,
    # East Asia (already have many Korean/Chinese -> reduced)
    "South Korean": 0.01, "North Korean-diaspora": 0.01,
    "Cantonese Chinese": 0.02, "Mandarin Northern Chinese": 0.01,
    "Hokkien/Fujian Chinese": 0.01, "Sichuan Chinese": 0.01,
    "Tibetan": 0.01, "Mongolian": 0.01, "Uyghur": 0.01,
    "Japanese": 0.02, "Okinawan": 0.01, "Ainu Japanese": 0.01,
    "Taiwanese Hakka": 0.01,
    # Sub-Saharan Africa (massively underrepresented)
    "Yoruba Nigerian": 0.02, "Igbo Nigerian": 0.02, "Hausa Nigerian": 0.01,
    "Ghanaian Akan": 0.02, "Ghanaian Ewe": 0.01,
    "Kenyan Kikuyu": 0.01, "Kenyan Luo": 0.01, "Kenyan Maasai": 0.01,
    "Ethiopian Amhara": 0.01, "Ethiopian Oromo": 0.01, "Eritrean": 0.01,
    "Tanzanian Swahili": 0.01, "Ugandan": 0.01, "Rwandan": 0.01,
    "South African Zulu": 0.01, "South African Xhosa": 0.01, "Afrikaner": 0.01,
    "Senegalese Wolof": 0.01, "Malian Bambara": 0.01, "Ivorian": 0.01,
    "Congolese": 0.01, "Cameroonian": 0.01, "Mozambican": 0.01,
    # Latin America / Caribbean
    "Brazilian Portuguese": 0.02, "Brazilian Nordestino": 0.01,
    "Argentine Rioplatense": 0.02, "Argentine Andean": 0.01,
    "Mexican mestizo": 0.02, "Mexican indigenous/Nahuatl": 0.01,
    "Colombian": 0.01, "Venezuelan": 0.01, "Peruvian": 0.01,
    "Chilean": 0.01, "Cuban": 0.01, "Jamaican": 0.01, "Haitian": 0.01,
    # Oceania
    "Maori New Zealander": 0.01, "Aboriginal Australian": 0.01,
    "Fijian": 0.01, "Samoan": 0.01, "Tongan": 0.01, "Papua New Guinean": 0.01,
}
# Note: weights don't need to sum to 1.0 -- random.choices normalizes them.

# Per-batch name constraint themes. Rotate through these to force rare/unique patterns.
# These go into the Step A prompt to mechanically prevent the LLM from using
# common patterns it's already generated 100s of times.
NAME_CONSTRAINTS = [
    "All FIRST NAMES must start with letters: X, Y, Z, Q, W, V (rare initials only)",
    "Use HYPHENATED or COMPOUND first names (Marie-Claude, Jean-Baptiste, Sun-Li, Ki-won)",
    "All persons born between 1935-1955: use period-authentic first names for that era and culture",
    "All persons born between 1985-2000: use modern, often anglicized or hybrid first names",
    "Use PATRONYMIC/MATRONYMIC surname patterns (Eriksson, Petrov, O'Brien, bin Abdullah, -ova, -idze)",
    "All surnames must be 3+ syllables and end in a vowel (Navarrete, Okonkwo, Kimura, Papadimitriou)",
    "Use LITERARY or PLACE-DERIVED surnames (names derived from landmarks, rivers, plants)",
    "Use names with DIACRITICS: accents, umlauts, cedillas (Björn, Ñoño, Fiévet, Ğüneş)",
    "All persons have DOUBLE-BARRELED surnames (Van der Berg, Diaz-Morales, Santos-Ferreira)",
    "Use DIMINUTIVE or NICKNAME as given name (Bea, Zeke, Tavi, Nix, Bram, Kit, Suki)",
]

STAGE_TARGETS = {"rising": 0.30, "prime": 0.35, "veteran": 0.20, "legend": 0.10, "retired": 0.05}

# ─── Rotating prompt themes ──────────────────────────────────────────
# Each batch picks a different creative angle to maximize diversity

BATCH_THEMES = [
    {
        "focus": "Award-winning indie cinema professionals",
        "bio_hint": "Focus on festival circuit careers, art-house sensibilities, breakthrough at Sundance/Cannes/Venice/TIFF",
        "example_bio": "After studying documentary filmmaking at CalArts, she pivoted to narrative features with a lo-fi aesthetic that caught the eye of Sundance programmers. Her debut feature, shot on 16mm in her grandmother's house, won the Grand Jury Prize and launched a career defined by intimate, observational storytelling.",
    },
    {
        "focus": "Mainstream blockbuster talent",
        "bio_hint": "Focus on commercial cinema, franchise work, box office appeal, action training",
        "example_bio": "A former professional stunt performer who transitioned to acting after a viral short film showcased his charisma. He trained at the Stella Adler Studio and quickly landed ensemble roles in franchise action films, becoming known for the rare ability to deliver witty dialogue mid-fight sequence.",
    },
    {
        "focus": "International arthouse and festival darlings",
        "bio_hint": "Focus on non-English language cinema, cultural specificity, national film movements",
        "example_bio": "Emerged from Tehran's underground theatre scene during a period of intense censorship, developing a spare, poetic performance style. Her collaboration with banned directors produced some of Iranian cinema's most quietly devastating works, earning recognition at Berlin and Locarno.",
    },
    {
        "focus": "TV-to-film crossover talent",
        "bio_hint": "Focus on prestige TV backgrounds, streaming era careers, showrunner experience",
        "example_bio": "Cut his teeth writing for late-night sketch comedy before creating a critically acclaimed limited series about small-town corruption. His signature blend of dark humor and social commentary attracted A-list talent and Netflix bidding wars, establishing him as a voice of the streaming generation.",
    },
    {
        "focus": "Genre specialists - horror, sci-fi, and fantasy",
        "bio_hint": "Focus on genre craftsmanship, practical effects knowledge, cult followings",
        "example_bio": "A makeup effects artist turned director who brings a practical craftsman's eye to body horror. Trained under legendary SFX houses, she understands creature design from the inside out, creating films where the monsters feel disturbingly biological. Her work has a devoted midnight-movie following.",
    },
    {
        "focus": "Behind-the-camera talent - DPs, editors, composers",
        "bio_hint": "Focus on technical artistry, collaboration style, visual/sonic signatures",
        "example_bio": "A classically trained pianist from the Moscow Conservatory who discovered film scoring through an accidental collaboration with a student filmmaker. His lush orchestral scores evoke golden-age Hollywood while incorporating electronic textures, making him the go-to composer for period dramas with a modern edge.",
    },
    {
        "focus": "Rising newcomers and debut filmmakers",
        "bio_hint": "Focus on fresh voices, social media discovery, first features, youth perspective",
        "example_bio": "A self-taught filmmaker from Lagos whose debut web series about motorcycle couriers went viral on YouTube, attracting Nollywood producers. At 24, she secured funding for her first feature, bringing raw energy and a documentary eye to street-level stories that major studios are now desperate to distribute.",
    },
    {
        "focus": "Veteran character actors and industry legends",
        "bio_hint": "Focus on long careers, iconic supporting roles, mentorship, body of work",
        "example_bio": "With over 200 film credits spanning five decades, he is the kind of actor whose face audiences recognize immediately but whose name they can never quite place. Trained at the Yale School of Drama, he brings meticulous preparation to even the smallest role, turning two-scene parts into the moments audiences remember.",
    },
    {
        "focus": "Documentary and non-fiction specialists",
        "bio_hint": "Focus on investigative work, social justice, vérité style, war correspondents",
        "example_bio": "A former war correspondent for Al Jazeera who turned her camera toward long-form documentary after witnessing events she felt needed more than a news segment. Her unflinching verité approach and ability to build trust with subjects in crisis zones has produced three Emmy-nominated documentaries.",
    },
    {
        "focus": "Animation and voice-over talent",
        "bio_hint": "Focus on voice range, motion capture, anime, Pixar/Studio Ghibli style work",
        "example_bio": "A theatrical voice coach turned prolific voice actor, bringing distinct character voices to over 50 animated features across Japanese anime and Western studio animation. Known for creating voices so lived-in that audiences forget they're hearing a performance.",
    },
]


def compute_gaps(all_persons, target_total):
    """Compute what nationalities, stages, and roles are underrepresented."""
    if not all_persons:
        return {}

    # Nationality gaps
    nat_counts = Counter(p["nationality"] for p in all_persons)
    nat_gaps = {}
    for nat, target_pct in NATIONALITY_WEIGHTS.items():
        expected = target_pct * target_total
        actual = nat_counts.get(nat, 0)
        if actual < expected * 0.7:  # more than 30% under target
            nat_gaps[nat] = int(expected - actual)

    # Stage gaps
    stage_counts = Counter(p["career_stage"] for p in all_persons)
    stage_gaps = {}
    for stage, target_pct in STAGE_TARGETS.items():
        expected = target_pct * target_total
        actual = stage_counts.get(stage, 0)
        if actual < expected * 0.7:
            stage_gaps[stage] = int(expected - actual)

    # Role gaps -- track which primary roles are over/under target.
    # Counts unique persons having each role (not total role slots).
    role_counts = Counter()
    for p in all_persons:
        for r in p.get("roles", []):
            role_counts[r] += 1
    # Which simple roles are over-represented vs ROLE_DISTRIBUTION target?
    # Down-weight roles that are already at >130% of target.
    role_over = {}  # role -> fraction over target (suppress in prompt)
    for role_combo, target_pct in ROLE_DISTRIBUTION.items():
        for r in role_combo.split(","):
            r = r.strip()
            actual_pct = role_counts.get(r, 0) / len(all_persons)
            if actual_pct > target_pct * 1.3:  # >30% over target
                role_over[r] = round(actual_pct / target_pct, 2)

    return {
        "nationality_gaps": nat_gaps,
        "stage_gaps": stage_gaps,
        "top_nationalities": dict(nat_counts.most_common(8)),
        "top_stages": dict(stage_counts.most_common()),
        "role_over": role_over,  # roles that are over-represented
    }




def parse_json_response(text):
    """Parse JSON with truncation recovery and bare-object handling."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Bare objects {..}{..} without array brackets
    if text.startswith("{"):
        wrapped = "[" + re.sub(r'\}\s*\{', '},{', text) + "]"
        try:
            return json.loads(wrapped)
        except json.JSONDecodeError:
            last_brace = wrapped.rfind("}")
            if last_brace > 0:
                candidate = wrapped[:last_brace + 1].rstrip().rstrip(",") + "]"
                try:
                    data = json.loads(candidate)
                    print(f"  (Recovered {len(data)} from bare objects)")
                    return data
                except json.JSONDecodeError:
                    pass

    # Truncated array recovery
    last_brace = text.rfind("}")
    if last_brace > 0:
        candidate = text[:last_brace + 1].rstrip().rstrip(",") + "\n]"
        try:
            data = json.loads(candidate)
            print(f"  (Recovered {len(data)} from truncated array)")
            return data
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON ({len(text)} chars)")


def validate_person(p, existing_names):
    """Validate and normalize. Returns (person, None) or (None, reason)."""
    name = p.get("name", "").strip()
    if not name or len(name) < 3:
        return None, f"SHORT_NAME:{name!r}"
    if name.lower() in existing_names:
        return None, f"DUPE:{name}"

    p["name"] = name
    p["nationality"] = p.get("nationality", "American")

    # Normalize gender -- model often returns full words
    GENDER_MAP = {"M": "M", "F": "F", "NB": "NB",
                  "Male": "M", "male": "M", "Female": "F", "female": "F",
                  "Non-binary": "NB", "non-binary": "NB", "Nonbinary": "NB",
                  "Non-Binary": "NB", "Other": "NB"}
    raw_gender = p.get("gender", "M")
    p["gender"] = GENDER_MAP.get(raw_gender, "M")

    p["bio"] = p.get("bio", "").strip()
    if len(p["bio"]) < 30:
        return None, f"SHORT_BIO:{len(p['bio'])}chars:{name}"

    st = p.get("style_tags", [])
    if isinstance(st, str):
        st = [s.strip() for s in st.split(",")]
    p["style_tags"] = st[:4]

    ga = p.get("genre_affinity", [])
    if isinstance(ga, str):
        ga = [g.strip() for g in ga.split(",")]
    p["genre_affinity"] = [g for g in ga if g in GENRES][:3] or ["Drama"]

    # Normalize career_stage -- model uses its own terms
    STAGE_MAP = {s: s for s in CAREER_STAGES}  # identity for valid ones
    STAGE_MAP.update({
        "Emerging": "rising", "emerging": "rising", "early": "rising",
        "Early-career": "rising", "early-career": "rising", "newcomer": "rising",
        "Rising Star": "rising", "Rising star": "rising", "rising star": "rising",
        "Mid-career": "prime", "mid-career": "prime", "Mid-Career": "prime",
        "Established": "prime", "established": "prime",
        "Peak": "prime", "peak": "prime",
        "Late-career": "veteran", "late-career": "veteran", "Late-Career": "veteran",
        "Senior": "veteran", "senior": "veteran", "Veteran": "veteran",
        "Legend": "legend", "Legendary": "legend", "legendary": "legend",
        "Icon": "legend", "icon": "legend",
        "Retired": "retired", "Semi-retired": "retired", "semi-retired": "retired",
    })
    cs = p.get("career_stage", "prime")
    p["career_stage"] = STAGE_MAP.get(cs) or STAGE_MAP.get(cs.lower(), "prime")

    roles = p.get("roles", ["actor"])
    if isinstance(roles, str):
        roles = [r.strip() for r in roles.split(",")]
    p["roles"] = roles

    mf = p.get("market_fit", ["Regional"])
    if isinstance(mf, str):
        mf = [m.strip() for m in mf.split(",")]
    p["market_fit"] = mf

    return p, None


def _call_api_with_timeout(llm, model, prompt, config, timeout=70):
    """Call LLM with timeout — delegates to llm.generate()."""
    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("max_output_tokens", 8192)
    return llm.generate(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout,
        max_attempts=1,
    )


def build_name_only_prompt(n_persons, batch_num, existing_names, all_persons, target_total):
    """Step A: Generate ONLY names+nationality+gender+role -- cheap, fast, dedup-friendly.

    Bios and style details are expensive (long output). By separating name generation
    we can dedup cheaply before spending tokens on descriptions.
    """
    rng = random.Random(batch_num * 137)
    theme = BATCH_THEMES[batch_num % len(BATCH_THEMES)]
    gaps = compute_gaps(all_persons, target_total)
    role_over = gaps.get("role_over", {})

    nat_gaps = gaps.get("nationality_gaps", {})
    if nat_gaps:
        gap_nats = list(nat_gaps.keys())
        n_gap = int(n_persons * 0.6)
        n_normal = n_persons - n_gap
        nat_targets = rng.choices(gap_nats, k=n_gap) + rng.choices(
            list(NATIONALITY_WEIGHTS.keys()),
            weights=list(NATIONALITY_WEIGHTS.values()),
            k=n_normal,
        )
    else:
        nat_targets = rng.choices(
            list(NATIONALITY_WEIGHTS.keys()),
            weights=list(NATIONALITY_WEIGHTS.values()),
            k=n_persons,
        )
    nat_counts = Counter(nat_targets)
    nat_breakdown = ", ".join(f"{v}x {k}" for k, v in nat_counts.most_common(10))

    # Build adjusted weights -- halve weight for any over-represented primary role
    adjusted_weights = []
    for role_combo, w in ROLE_DISTRIBUTION.items():
        primary = role_combo.split(",")[0].strip()
        if primary in role_over:
            w = w * 0.5  # suppress: less likely to generate more of this role
        adjusted_weights.append(w)

    roles_needed = rng.choices(
        list(ROLE_DISTRIBUTION.keys()),
        weights=adjusted_weights,
        k=n_persons,
    )
    role_counts = Counter(roles_needed)
    role_breakdown = ", ".join(f"{v}x {k}" for k, v in role_counts.most_common())
    if role_over:
        role_breakdown += f"  [suppressed: {list(role_over)}]"

    stage_gaps = gaps.get("stage_gaps", {})
    if stage_gaps:
        stage_line = ", ".join(f"NEED MORE {k} (gap: {v})" for k, v in stage_gaps.items())
    else:
        stage_targets = {k: max(1, int(v * n_persons)) for k, v in STAGE_TARGETS.items()}
        stage_line = ", ".join(f"{v}x {k}" for k, v in stage_targets.items())

    # Pass ALL existing names to the LLM as a blacklist.
    # Input is $0.25/1M tokens vs output $1.50/1M -- 6x cheaper.
    # Paying ~$0.012 in input to show all 9k+ names beats wasting $0.017 in
    # output tokens on duplicate names that get rejected post-generation.
    # Names sorted alphabetically so LLM can efficiently check with binary search.
    all_names_sorted = sorted(existing_names)  # already lowercased
    # Format as one name per line for maximum LLM readability at scale
    name_list_str = "\n".join(all_names_sorted)
    names_block = (
        f"\nFORBIDDEN -- these {len(all_names_sorted)} names already exist. "
        f"Every name you generate MUST NOT appear in this list:\n"
        f"<existing_names>\n{name_list_str}\n</existing_names>\n"
    )

    # Rotating name constraint -- forces rare/unique naming patterns each batch
    name_constraint = NAME_CONSTRAINTS[(batch_num // 2) % len(NAME_CONSTRAINTS)]

    return f"""Generate {n_persons} COMPLETELY FICTIONAL movie industry person stubs.

BATCH THEME: {theme['focus']}

CRITICAL: ALL NAMES MUST BE INVENTED. NO REAL CELEBRITIES.
  WRONG: "Joaquin Phoenix", "Meryl Streep"
  RIGHT: "Joaquin Delgado", "Margaret Ashford"
{names_block}
NAME CONSTRAINT FOR THIS BATCH (MANDATORY): {name_constraint}

Role breakdown: {role_breakdown}
Nationalities: {nat_breakdown}
Career stages: {stage_line}

Output a JSON array of EXACTLY {n_persons} objects with ONLY these 5 fields:
  name, nationality, gender (M/F/NB), roles (array), career_stage

Names MUST match nationality. No markdown. Only JSON array starting with ["""


def build_bio_prompt(name_stubs, batch_num):
    """Step B: Given deduplicated name stubs, generate full profiles with bios."""
    theme = BATCH_THEMES[batch_num % len(BATCH_THEMES)]

    lines = []
    for s in name_stubs:
        roles_str = ",".join(s.get("roles", ["actor"]))
        lines.append(
            f'[{s["name"]} | {s.get("nationality","?")} | {s.get("gender","M")} | '
            f'{roles_str} | {s.get("career_stage","prime")}]'
        )
    stubs_block = "\n".join(lines)

    return f"""Complete these {len(name_stubs)} movie industry persons with full profiles.

BATCH THEME: {theme['focus']}
{theme['bio_hint']}

For EACH person below, output a JSON object with their EXACT name plus all fields.

PERSONS TO COMPLETE:
{stubs_block}

For each person output:
  name (EXACT as given), nationality (EXACT as given), gender (EXACT as given),
  roles (EXACT as given), career_stage (EXACT as given),
  bio (2-3 vivid sentences, Wikipedia-style),
  style_tags (2-4 from: {', '.join(STYLE_TAGS[:20])}),
  genre_affinity (1-3 from: {', '.join(GENRES)}),
  market_fit (1-2 from: Local, Regional, North America, Europe, Asia, Global)

EXAMPLE BIO:
"{theme['example_bio']}"

No markdown. Only JSON array starting with ["""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--target", type=int, default=5000)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--timeout", type=int, default=70,
                        help="API call hang timeout in seconds (default: 70)")
    args = parser.parse_args()

    llm = get_llm_client()

    # ALWAYS resume from existing data -- never delete
    out_path = ENTITY_DIR / "persons.json"
    all_persons = []
    existing_names = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            all_persons = json.load(f)
        existing_names = {p["name"].lower() for p in all_persons}
        print(f"Resuming from {len(all_persons)} existing persons")

    target = args.target
    batch_size = args.batch_size
    api_timeout = args.timeout
    total_input = 0
    total_output = 0
    total_cost = 0.0
    consecutive_failures = 0
    acceptance_history = []  # rolling acceptance rate guard

    remaining = max(0, target - len(all_persons))
    n_batches = (remaining + batch_size - 1) // batch_size
    est_cost = n_batches * batch_size * 200 / 1e6 * PRICING["output"]
    print(f"Target: {target}, batch size: {batch_size}")
    print(f"Model: {args.model}  |  API timeout: {api_timeout}s")
    print(f"Already have: {len(all_persons)}, need: {remaining}")
    print(f"Est. batches: {n_batches}, est. cost: ${est_cost:.2f}")
    print(f"2-step approach: Step A=names only (cheap), Step B=full bios (after dedup)")

    base_config = {
        "temperature": 0.9,
        "thinking_config": {"thinking_budget": 0},
    }

    batch_num = len(all_persons) // batch_size
    while len(all_persons) < target:
        batch_num += 1
        need = min(batch_size, target - len(all_persons))
        theme_idx = batch_num % len(BATCH_THEMES)

        print(f"\n  Batch {batch_num} ({len(all_persons)}/{target}) "
              f"[Theme: {BATCH_THEMES[theme_idx]['focus'][:40]}]")

        # ---- STEP A: Generate names only -- slight oversampling for safety ----
        # With full name blacklist in prompt, LLM should achieve 70-90% hit rate.
        # 1.3x oversampling (ask 130 to get ~100) is plenty vs old 2.5x.
        n_ask = min(int(need * 1.3), 150)
        name_prompt = build_name_only_prompt(
            n_ask, batch_num, existing_names, all_persons, target
        )
        name_stubs = None
        for retry in range(5):
            try:
                resp = _call_api_with_timeout(
                    llm, args.model, name_prompt,
                    {**base_config, "max_output_tokens": n_ask * 45},
                    timeout=api_timeout,
                )
            except TimeoutError as e:
                print(f"  STEP-A TIMEOUT (retry {retry+1}): {e}")
                time.sleep(5 * (retry + 1))
                continue
            except Exception as e:
                err_str = str(e)
                is_503 = "503" in err_str or "unavailable" in err_str.lower()
                print(f"  STEP-A {'503 ' if is_503 else ''}API ERROR (retry {retry+1}): {e}")
                time.sleep(5 * (retry + 1))
                continue

            inp = resp.input_tokens
            out = resp.output_tokens
            cost = resp.cost_usd
            total_input += inp; total_output += out; total_cost += cost
            print(f"  [A] {inp:,}in+{out:,}out=${cost:.4f}")

            try:
                raw = parse_json_response(resp.text)
                if isinstance(raw, list) and len(raw) > 0:
                    name_stubs = raw
                    break
                print(f"  [A] Empty response, retrying...")
            except Exception as e:
                print(f"  [A] PARSE ERROR (retry {retry+1}): {e}")
                raw_path = BASE_DIR / "_dev" / f"names_batch{batch_num}_r{retry}_raw.txt"
                os.makedirs(BASE_DIR / "_dev", exist_ok=True)
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                time.sleep(2)

        if name_stubs is None:
            consecutive_failures += 1
            print(f"  [A] FAILED -- skipping batch {batch_num}")
            if consecutive_failures >= 3:
                print("  3 consecutive failures -- stopping")
                break
            continue

        # ---- DEDUP: O(1) set lookup on lowercased names ----------------------
        unique_stubs = []
        for stub in name_stubs:
            name = str(stub.get("name", "")).strip()
            if not name or len(name) < 3:
                continue
            if name.lower() in existing_names:
                continue  # fast O(1) reject
            existing_names.add(name.lower())  # claim now
            unique_stubs.append(stub)
            if len(unique_stubs) >= need:  # got enough -- stop early
                break

        n_asked = len(name_stubs)
        n_dupes = n_asked - len(unique_stubs)
        pct_unique = len(unique_stubs) / n_asked * 100 if n_asked > 0 else 0
        print(f"  [A] {n_asked} asked -> {len(unique_stubs)} unique "
              f"({n_dupes} dupes, {pct_unique:.0f}% hit rate, need={need})")

        if not unique_stubs:
            print("  All names were dupes -- refetching")
            continue

        # ── STEP B: Full profiles for unique names only ────────────────────────
        bio_prompt = build_bio_prompt(unique_stubs, batch_num)
        persons = None
        for retry in range(5):
            try:
                resp = _call_api_with_timeout(
                    llm, args.model, bio_prompt,
                    {**base_config, "max_output_tokens": len(unique_stubs) * 200},
                    timeout=api_timeout,
                )
            except TimeoutError as e:
                print(f"  STEP-B TIMEOUT (retry {retry+1}): {e}")
                time.sleep(5 * (retry + 1))
                continue
            except Exception as e:
                err_str = str(e)
                is_503 = "503" in err_str or "unavailable" in err_str.lower()
                print(f"  STEP-B {'503 ' if is_503 else ''}API ERROR (retry {retry+1}): {e}")
                time.sleep(5 * (retry + 1))
                continue

            inp = resp.input_tokens
            out = resp.output_tokens
            cost = resp.cost_usd
            total_input += inp; total_output += out; total_cost += cost
            print(f"  [B] {inp:,}in+{out:,}out=${cost:.4f}")

            try:
                raw_persons = parse_json_response(resp.text)
                if isinstance(raw_persons, list) and len(raw_persons) > 0:
                    persons = raw_persons
                    break
                print(f"  [B] Empty response, retrying...")
            except Exception as e:
                print(f"  [B] PARSE ERROR (retry {retry+1}): {e}")
                raw_path = BASE_DIR / "_dev" / f"persons_batch{batch_num}_r{retry}_raw.txt"
                os.makedirs(BASE_DIR / "_dev", exist_ok=True)
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                time.sleep(2)

        if persons is None:
            consecutive_failures += 1
            print(f"  [B] FAILED batch {batch_num}")
            # Remove claimed names back from the set (we won't have their bios)
            for stub in unique_stubs:
                existing_names.discard(stub.get("name", "").lower())
            if consecutive_failures >= 3:
                print("  3 consecutive failures -- stopping")
                break
            continue

        consecutive_failures = 0

        # Validate and add.
        # IMPORTANT: pass a snapshot of confirmed_names (before Step A claimed these names)
        # for the bio-validation name check. Step A already guaranteed uniqueness;
        # if we pass existing_names here every result would be rejected as DUPE
        # because we pre-added all unique_stubs names to existing_names in Step A.
        confirmed_names_snapshot = existing_names - {s.get("name","").lower() for s in unique_stubs}
        added = 0
        rejected = 0
        reject_reasons = []
        for p in persons:
            result, reason = validate_person(p, confirmed_names_snapshot)
            if result:
                all_persons.append(result)
                added += 1
            else:
                # Name wasn't in the bio response or bio was malformed -- release the claim
                rejected += 1
                reject_reasons.append(reason)
                bad_name = str(p.get("name", "")).strip().lower()
                existing_names.discard(bad_name)

        acceptance_pct = (added / (added + rejected) * 100) if (added + rejected) > 0 else 100
        print(f"  +{added} persons (rejected {rejected}, {acceptance_pct:.0f}%) "
              f"-> {len(all_persons)} total | run cost ${total_cost:.3f}")
        if reject_reasons:
            reason_types = Counter(r.split(':')[0] for r in reject_reasons)
            print(f"    Reject breakdown: {dict(reason_types)}")

        # Rolling acceptance rate guard (over last 5 batches)
        acceptance_history.append(acceptance_pct)
        if len(acceptance_history) >= 5:
            rolling_avg = sum(acceptance_history[-5:]) / 5
            if rolling_avg < 60:
                print(f"\n  STOPPING: Rolling Step-B acceptance {rolling_avg:.0f}% < 60% threshold")
                print(f"  Last 5 batches: {[f'{x:.0f}%' for x in acceptance_history[-5:]]}")
                print(f"  All data saved ({len(all_persons)} persons). Re-run to resume.")
                break

        # Show 1 sample
        if added > 0:
            s = all_persons[-1]
            safe_name = s['name'].encode('ascii', 'replace').decode()
            safe_bio = s.get('bio', '')[:140].encode('ascii', 'replace').decode()
            print(f"    [{s['career_stage']:<8}] {safe_name} ({s['nationality']}, {s['gender']})")
            print(f"    {safe_bio}...")

        # INCREMENTAL SAVE -- always
        os.makedirs(ENTITY_DIR, exist_ok=True)
        for i, p in enumerate(all_persons):
            if "person_id" not in p:
                p["person_id"] = i + 1
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_persons, f, indent=2, ensure_ascii=False)

        time.sleep(0.5)

    # ─── Final report ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PERSON GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total:    {len(all_persons)}")
    print(f"  Tokens:   {total_input:,} in + {total_output:,} out")
    print(f"  Cost:     ${total_cost:.4f}  (${total_cost / max(len(all_persons),1) * 1000:.2f} per 1k persons)")

    genders = Counter(p["gender"] for p in all_persons)
    stages = Counter(p["career_stage"] for p in all_persons)
    roles = Counter()
    for p in all_persons:
        for r in p.get("roles", []):
            roles[r] += 1
    nats = Counter(p["nationality"] for p in all_persons)

    print(f"\n  Gender: {dict(genders.most_common())}")
    print(f"  Stages: {dict(stages.most_common())}")
    print(f"  Roles:  {dict(roles.most_common(10))}")
    print(f"  Nationalities: {dict(nats.most_common(10))}")
    # Show targets vs actuals for key roles
    n_total = len(all_persons)
    print(f"\n  Role targets at {n_total} persons:")
    for role_combo, target_pct in ROLE_DISTRIBUTION.items():
        primary = role_combo.split(",")[0].strip()
        actual = roles.get(primary, 0)
        target_n = int(target_pct * n_total)
        pct = actual / n_total * 100
        flag = " !! OVER" if actual > target_n * 1.3 else (" !! UNDER" if actual < target_n * 0.7 else "")
        print(f"    {primary:<20} actual={actual:>5} ({pct:>4.1f}%)  target~{target_n}{flag}")
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
